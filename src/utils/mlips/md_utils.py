import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MDStopIteration(Exception):
    """Exception raised to stop MD simulation from a callback."""
    pass

class ExplosionMonitor:
    """
    Monitor for unphysical instabilities in MD simulations.
    Stops the simulation if temperature exceeds a threshold or becomes NaN.
    """
    def __init__(self, atoms=None, temp_threshold: float = 10000.0):
        self.atoms = atoms
        self.temp_threshold = temp_threshold

    def __call__(self, *args, **kwargs):
        dyn = kwargs.get("dyn")
        atoms = self.atoms
        for arg in args:
            if hasattr(arg, "get_temperature"):
                if hasattr(arg, "atoms"):
                    dyn = arg
                    atoms = arg.atoms
                else:
                    atoms = arg
                break
        
        if atoms is None and dyn is not None and hasattr(dyn, "atoms"):
            atoms = dyn.atoms
            
        if atoms is None:
            return
            
        temp = atoms.get_temperature()
        if np.isnan(temp) or temp > self.temp_threshold:
            msg = f"Explosion detected: Temperature = {temp:.1f}K"
            logger.error(msg)
            raise MDStopIteration(msg)

class EquilibrationMonitor:
    """
    Monitor for simulation equilibration/stability.
    Detects when temperature and potential energy have converged.
    """
    def __init__(
        self, 
        atoms=None, 
        timestep_fs: float | None = None, 
        log_interval: int | None = None, 
        window_ps: float = 1.0, 
        stability_ps: float = 5.0, 
        temp_std_threshold: float = 50.0,
        **kwargs
    ):
        self.atoms = atoms
        self.timestep_fs = timestep_fs
        self.log_interval = log_interval
        self.window_ps = window_ps
        self.stability_ps = stability_ps
        self.temp_std_threshold = temp_std_threshold
        
        self.window_entries = None
        self.stability_entries = None
        self.history = {"temps": [], "epots": [], "stds": []}

    def _resolve_params(self, atoms, dyn=None):
        """Resolve simulation parameters from atoms or dynamics object."""
        if self.atoms is None:
            self.atoms = atoms
            
        if dyn is not None:
            # Try to get timestep from Dynamics object (ASE units)
            if self.timestep_fs is None:
                from ase import units
                # ASE Dynamics object has dt or get_time_step()
                if hasattr(dyn, "dt"):
                    self.timestep_fs = dyn.dt / units.fs
                elif hasattr(dyn, "get_time_step"):
                    self.timestep_fs = dyn.get_time_step() / units.fs
            
            # Log interval is usually not in dyn directly for the callback,
            # but we pass it during attach if we want.
            # However, matcalc uses its own loginterval.
            # We can try to guess it or pass it. 
        
        # Fallback to defaults if still None
        ts = self.timestep_fs if self.timestep_fs is not None else 1.0
        li = self.log_interval if self.log_interval is not None else 10
        
        if self.window_entries is None:
            self.window_entries = max(2, int(self.window_ps * 1000 / ts / li))
        if self.stability_entries is None:
            self.stability_entries = max(2, int(self.stability_ps * 1000 / ts / li))

    def __call__(self, *args, **kwargs):
        # find atoms and potentially dyn
        dyn = kwargs.get("dyn")
        atoms = self.atoms
        
        for arg in args:
            if hasattr(arg, "get_temperature"):
                if hasattr(arg, "atoms"): # Likely a Dynamics object
                    dyn = arg
                    atoms = arg.atoms
                else:
                    atoms = arg
                break
        
        if atoms is None and dyn is not None and hasattr(dyn, "atoms"):
            atoms = dyn.atoms
            
        if atoms is None:
            return

        # Resolve parameters on the first call
        self._resolve_params(atoms, dyn)
        
        curr_temp = atoms.get_temperature()
        curr_epot = atoms.get_potential_energy() / len(atoms)
        
        self.history["temps"].append(curr_temp)
        self.history["epots"].append(curr_epot)
        
        if len(self.history["temps"]) >= self.window_entries:
            # We check the standard deviation of the most recent window
            recent_temps = self.history["temps"][-self.window_entries:]
            std_t = np.std(recent_temps)
            self.history["stds"].append(std_t)
            
            # Condition 1: Absolute threshold
            if std_t < self.temp_std_threshold:
                # Still check minimal duration? Original code checked stability_entries against history length
                # User said: "alternatively ... smaller than threshold"
                # Let's keep a sanity check that we have at least one window
                msg = f"Equilibration reached: std(T)={std_t:.2f} < {self.temp_std_threshold}"
                logger.info(msg)
                raise MDStopIteration(msg)

            # Condition 2: Fluctuation stability (plateau)
            # Check if stds have been recorded for stability_ps duration
            # "if it doesn't decrease (or remain at similar value) for stability_ps time"
            
            # We need enough std history to cover stability_ps
            # stability_entries is count of points (assuming 1 point per call effectively?)
            # Wait, stability_entries is calculated as (time / ts / li)
            # self.history["stds"] grows by 1 every call.
            # So looking back `stability_entries` in `stds` corresponds to `stability_ps`.
            
            if len(self.history["stds"]) >= self.stability_entries:
                recent_stds = self.history["stds"][-self.stability_entries:]
                
                # Check trend. Simple linear regression slope.
                x = np.arange(len(recent_stds))
                slope, _ = np.polyfit(x, recent_stds, 1)
                
                # If slope >= -epsilon (it is not decreasing significantly)
                # We assume decreasing is negative slope. 
                # "doesn't decrease" -> slope >= ~0
                # "remain at similar value" -> slope ~ 0
                
                # Let's say if slope > -0.005 (very slightly decreasing or flat/increasing)
                # It means we are not improving much anymore.
                
                # Also check relative change?
                # Let's stick to slope.
                
                if slope > -0.01: # Threshold for "not decreasing"
                    avg_std = np.mean(recent_stds)
                    msg = f"Equilibration reached (stable fluctuation): slope={slope:.4f}, mean_std={avg_std:.2f}"
                    logger.info(msg)
                    raise MDStopIteration(msg)

class OvershootMonitor:
    """
    Monitor for temperature overshoot or thermostat failure.
    Stops if temperature deviates too far from target.
    """
    def __init__(self, atoms=None, target_temp: float = 300.0, tolerance: float = 500.0, delay_steps: int = 100):
        self.atoms = atoms
        self.target_temp = target_temp
        self.tolerance = tolerance
        self.delay_steps = delay_steps  # Allow some steps for initial ramp
        self.step_count = 0

    def __call__(self, *args, **kwargs):
        self.step_count += 1
        if self.step_count < self.delay_steps:
            return
            
        dyn = kwargs.get("dyn")
        atoms = self.atoms
        for arg in args:
            if hasattr(arg, "get_temperature"):
                if hasattr(arg, "atoms"):
                    dyn = arg
                    atoms = arg.atoms
                else:
                    atoms = arg
                break
        
        if atoms is None and dyn is not None and hasattr(dyn, "atoms"):
            atoms = dyn.atoms
            
        if atoms is None:
            return
            
        temp = atoms.get_temperature()
        if abs(temp - self.target_temp) > self.tolerance:
            msg = f"Temperature overshoot detected: T={temp:.1f}K, Target={self.target_temp}K, Tolerance={self.tolerance}K"
            logger.error(msg)
            raise MDStopIteration(msg)

class VolumeMonitor:
    """
    Monitor for volume expansion/contraction in NPT simulations.
    Stops if volume exceeds upper or lower limit ratios.
    """
    def __init__(self, atoms=None, lower_limit_ratio: float = 0.2, upper_limit_ratio: float = 2.0):
        self.atoms = atoms
        self.initial_volume = atoms.get_volume() if atoms else None
        self.lower_limit_ratio = lower_limit_ratio
        self.upper_limit_ratio = upper_limit_ratio

    def __call__(self, *args, **kwargs):
        dyn = kwargs.get("dyn")
        atoms = self.atoms
        for arg in args:
            if hasattr(arg, "get_volume"):
                if hasattr(arg, "atoms"):
                    dyn = arg
                    atoms = arg.atoms
                else:
                    atoms = arg
                break
        
        if atoms is None and dyn is not None and hasattr(dyn, "atoms"):
            atoms = dyn.atoms
            
        if atoms is None:
            return
        
        if self.initial_volume is None:
            self.initial_volume = atoms.get_volume()
            
        current_volume = atoms.get_volume()
        if current_volume > self.initial_volume * self.upper_limit_ratio:
            msg = f"Volume expansion detected: Volume {current_volume:.1f} A^3 > {self.upper_limit_ratio}x initial ({self.initial_volume:.1f} A^3)"
            logger.error(msg)
            raise MDStopIteration(msg)
        
        if current_volume < self.initial_volume * self.lower_limit_ratio:
            msg = f"Volume contraction detected: Volume {current_volume:.1f} A^3 < {self.lower_limit_ratio}x initial ({self.initial_volume:.1f} A^3)"
            logger.error(msg)
            raise MDStopIteration(msg)

def get_md_callback(
    monitor_type: str, 
    atoms=None, 
    timestep_fs: Optional[float] = None, 
    log_interval: Optional[int] = None, 
    **kwargs
):
    """Factory function to create MD callbacks."""
    # Note: timestep_fs and log_interval can now be None for auto-detection in Monitors
    
    if monitor_type == "explosion":
        threshold = kwargs.get("temp_threshold", 10000.0)
        return ExplosionMonitor(atoms, threshold), log_interval
    elif monitor_type == "equilibration" or monitor_type == "melting":
        # 'melting' is kept for backward compatibility and is a specific case of equilibration
        return EquilibrationMonitor(
            atoms=atoms, 
            timestep_fs=timestep_fs, 
            log_interval=log_interval, 
            **kwargs
        ), log_interval
    elif monitor_type == "overshoot":
        target_temp = kwargs.get("temperature", 300.0)
        tolerance = kwargs.get("tolerance", 500.0)
        return OvershootMonitor(atoms, target_temp, tolerance), log_interval
    elif monitor_type == "volume":
        lower = kwargs.get("lower_limit_ratio", 0.2)
        upper = kwargs.get("upper_limit_ratio", 2.0)
        return VolumeMonitor(atoms, lower, upper), log_interval
    
    return None
