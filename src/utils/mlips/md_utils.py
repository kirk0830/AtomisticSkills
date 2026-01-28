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
    def __init__(self, atoms, temp_threshold: float = 10000.0):
        # logger.info(f"DEBUG: ExplosionMonitor init with {type(atoms)}")
        self.atoms = atoms
        self.temp_threshold = temp_threshold

    def __call__(self, *args, **kwargs):
        # ASE Callbacks can be called in different ways. 
        # Usually it's (bool, atoms) or just (atoms) or ()
        atoms = self.atoms
        for i, arg in enumerate(args):
            # logger.info(f"DEBUG: arg[{i}] type={type(arg)}")
            # Check if it has the required method AND is not a float
            if not isinstance(arg, (float, int, bool)) and hasattr(arg, "get_temperature"):
                atoms = arg
                break
        
        if isinstance(atoms, (float, int)):
             logger.error(f"CORRUPTION: atoms is {type(atoms)}. self.atoms was {type(self.atoms)}")
             # Emergency fallback: if self.atoms was corrupted, we are in trouble.
             # But how could self.atoms become a float?
        
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
        atoms, 
        timestep_fs: float, 
        log_interval: int, 
        window_ps: float = 1.0, 
        stability_ps: float = 5.0, 
        temp_std_threshold: float = 50.0
    ):
        self.atoms = atoms
        self.window_entries = max(2, int(window_ps * 1000 / timestep_fs / log_interval))
        self.stability_entries = max(2, int(stability_ps * 1000 / timestep_fs / log_interval))
        self.temp_std_threshold = temp_std_threshold
        
        self.history = {"temps": [], "epots": []}

    def __call__(self, *args, **kwargs):
        atoms = self.atoms
        for arg in args:
            if hasattr(arg, "get_temperature"):
                atoms = arg
                break
                
        curr_temp = atoms.get_temperature()
        curr_epot = atoms.get_potential_energy() / len(atoms)
        
        self.history["temps"].append(curr_temp)
        self.history["epots"].append(curr_epot)
        
        if len(self.history["temps"]) >= self.window_entries:
            # We check the standard deviation of the most recent window
            recent_temps = self.history["temps"][-self.window_entries:]
            std_t = np.std(recent_temps)
            
            # Simple stability: if temp std is low enough
            if std_t < self.temp_std_threshold:
                # Optional: check if it's been stable for enough time
                if len(self.history["temps"]) >= self.stability_entries:
                    msg = f"Equilibration reached: std(T)={std_t:.2f} < {self.temp_std_threshold}"
                    logger.info(msg)
                    raise MDStopIteration(msg)

class OvershootMonitor:
    """
    Monitor for temperature overshoot or thermostat failure.
    Stops if temperature deviates too far from target.
    """
    def __init__(self, atoms, target_temp: float, tolerance: float = 500.0, delay_steps: int = 100):
        self.atoms = atoms
        self.target_temp = target_temp
        self.tolerance = tolerance
        self.delay_steps = delay_steps  # Allow some steps for initial ramp
        self.step_count = 0

    def __call__(self, *args, **kwargs):
        self.step_count += 1
        if self.step_count < self.delay_steps:
            return
            
        atoms = self.atoms
        for arg in args:
            if hasattr(arg, "get_temperature"):
                atoms = arg
                break
                
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
    def __init__(self, atoms, lower_limit_ratio: float = 0.2, upper_limit_ratio: float = 2.0):
        self.atoms = atoms
        self.initial_volume = atoms.get_volume()
        self.lower_limit_ratio = lower_limit_ratio
        self.upper_limit_ratio = upper_limit_ratio

    def __call__(self, *args, **kwargs):
        atoms = self.atoms
        for arg in args:
            if hasattr(arg, "get_volume"):
                atoms = arg
                break
        
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
    atoms, 
    timestep_fs: float, 
    log_interval: int, 
    **kwargs
):
    """Factory function to create MD callbacks."""
    if monitor_type == "explosion":
        threshold = kwargs.get("temp_threshold", 10000.0)
        return ExplosionMonitor(atoms, threshold), log_interval
    elif monitor_type == "equilibration" or monitor_type == "melting":
        # 'melting' is kept for backward compatibility and is a specific case of equilibration
        return EquilibrationMonitor(
            atoms, 
            timestep_fs, 
            log_interval, 
            **kwargs
        ), log_interval
    elif monitor_type == "overshoot":
        target_temp = kwargs.get("temperature", 300.0)
        tolerance = kwargs.get("tolerance", 500.0) # Default large tolerance
        return OvershootMonitor(atoms, target_temp, tolerance), log_interval
    elif monitor_type == "volume":
        lower = kwargs.get("lower_limit_ratio", 0.2)
        upper = kwargs.get("upper_limit_ratio", 2.0)
        return VolumeMonitor(atoms, lower, upper), log_interval
    
    return None
