"""
Utility module for running VASP calculations using atomate2.
"""

import os
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read

logger = logging.getLogger(__name__)

class Atomate2Handler:
    """Handles atomate2 VASP flows and job execution."""

    def __init__(self, output_dir: str):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.output_path / "results"
        self.results_dir.mkdir(exist_ok=True)

    def check_environment(self) -> Dict[str, Any]:
        """Check if VASP, POTCAR, and atomate2 are available."""
        checks = {
            "atomate2": False,
            "vasp": False,
            "potcar": False,
            "error": None
        }

        # Check atomate2
        import atomate2
        import jobflow
        checks["atomate2"] = True

        # Check VASP cmd
        vasp_cmd = os.environ.get("ATOMATE2_VASP_CMD") or os.environ.get("VASP_CMD")
        if vasp_cmd:
            checks["vasp"] = True
        else:
            # Check common location
            vasp_bin = Path("/home/bdeng/Packages/vasp.6.4.2/bin/vasp_std")
            if vasp_bin.exists():
                checks["vasp"] = True
                os.environ["ATOMATE2_VASP_CMD"] = f"mpirun -np 1 {vasp_bin}"
            else:
                checks["error"] = "VASP_CMD or ATOMATE2_VASP_CMD not set."

        # Check POTCAR
        if os.environ.get("PMG_VASP_PSP_DIR"):
            checks["potcar"] = True
        else:
            checks["error"] = "PMG_VASP_PSP_DIR (POTCAR directory) not set."

        return checks

    def load_structures(self, structures_path: str) -> List[Structure]:
        """Load structures from path."""
        path = Path(structures_path)
        structures = []

        if path.is_dir():
            files = sorted(list(path.glob("*.cif")) + list(path.glob("*.xyz")) + list(path.glob("POSCAR*")))
            for f in files:
                if f.suffix == ".cif":
                    structures.append(Structure.from_file(str(f)))
                else:
                    atoms = read(str(f))
                    structures.append(AseAtomsAdaptor.get_structure(atoms))
        elif path.is_file():
            if path.suffix == ".cif":
                structures.append(Structure.from_file(str(path)))
            else:
                atoms = read(str(path))
                structures.append(AseAtomsAdaptor.get_structure(atoms))

        return structures

    def get_flow_maker(self, preset_type: str, calculation_type: str, config: Optional[Dict[str, Any]] = None):
        """Create atomate2 flow maker based on preset."""
        from atomate2.vasp.flows.matpes import MatPesStaticFlowMaker
        from atomate2.vasp.jobs.matpes import MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker
        from atomate2.vasp.jobs.core import StaticMaker, RelaxMaker

        preset = preset_type.lower()
        user_incar = config or {}
        
        if preset == "matpes-r2scan":
            # For MatPES, we apply custom settings to the relevant makers
            # Note: MatPesStaticFlowMaker doesn't easily expose nested config in init
            # but we can try to wrap it or just use standard makers if config is provided
            return MatPesStaticFlowMaker(
                static1=None,
                static2=MatPesMetaGGAStaticMaker(
                    input_set_generator=MatPesMetaGGAStaticMaker().input_set_generator.default_factory()
                ).update_incar(user_incar) if user_incar else MatPesMetaGGAStaticMaker(),
                static3=None
            )
        elif preset == "matpes-pbe":
            return MatPesStaticFlowMaker(
                static1=MatPesGGAStaticMaker(
                    input_set_generator=MatPesGGAStaticMaker().input_set_generator.default_factory()
                ).update_incar(user_incar) if user_incar else MatPesGGAStaticMaker(),
                static2=None,
                static3=None
            )
        elif preset in ["mp", "omat"]:
            if calculation_type == "static":
                maker = StaticMaker()
            else:
                maker = RelaxMaker()
            
            if user_incar:
                maker.input_set_generator.user_incar_settings.update(user_incar)
            return maker
        else:
            raise ValueError(f"Unknown preset_type: {preset_type}")

    def check_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a job."""
        # Try status file first (local or cached remote)
        status_file = self.output_path / f"{job_id}_status.json"
        
        result = {"job_id": job_id, "status": "unknown"}

        if status_file.exists():
            with open(status_file) as f:
                result.update(json.load(f))
            return result
        
        # Check if it's a remote job by checking jf project
        try:
            from jobflow_remote.jobs.jobcontroller import JobController
            
            # Load controller to check status
            jc = JobController.from_project_name("remote_test")
            
            if job_id.isdigit():
                flows_info = jc.get_flows_info(db_ids=[job_id])
            else:
                flows_info = jc.get_flows_info(flow_ids=[job_id])

            if flows_info:
                flow_info = flows_info[0]
                result["status"] = flow_info.state.name
                result["remote"] = True
                # Cache status
                with open(status_file, "w") as f:
                    json.dump(result, f)
                return result
        except Exception as e:
            logger.debug(f"Remote status check failed: {e}")

        return result

    def _check_sshproxy(self, worker_name: str) -> Tuple[bool, str]:
        """
        Check if sshproxy is configured for NERSC/Perlmutter workers.
        
        Returns:
            Tuple of (is_configured, message)
        """
        if "perlmutter" not in worker_name.lower():
            return True, ""

        # Check for NERSC key file
        nersc_key = Path.home() / ".ssh" / "nersc"
        if not nersc_key.exists():
            return False, f"NERSC SSH key not found at {nersc_key}. Please run 'sshproxy -u <username>' to generate keys."
            
        # Optional: Check if key is expired (naive check based on file modification time > 24h?)
        # For now, existence is the primary check requested.
        
        return True, "SSHProxy appears configured."

    def run_remote(self, flow: Any, project_name: str = "remote_test", worker_name: str = None) -> str:
        """Submit a flow remotely using jobflow-remote."""
        
        # Check sshproxy if applicable
        if worker_name:
            is_configured, msg = self._check_sshproxy(worker_name)
            if not is_configured:
                raise RuntimeError(f"SSHProxy check failed: {msg}")

        from jobflow_remote import submit_flow
        
        db_ids = submit_flow(flow, project=project_name, worker=worker_name)
        # Use the first db_id as the job_id for tracking
        job_id = db_ids[0]
        
        status_file = self.output_path / f"{job_id}_status.json"
        with open(status_file, "w") as f:
            json.dump({
                "job_id": job_id,
                "status": "SUBMITTED",
                "remote": True,
                "project": project_name,
                "worker": worker_name,
                "submitted_at": datetime.now().isoformat()
            }, f)
            
        return job_id

    def extract_results(self, job_id: str) -> Dict[str, Any]:
        """Extract energy, forces, and stress from completed job using idiomatic Atomate2 parsing."""
        responses_file = self.output_path / f"{job_id}_responses.pkl"
        all_responses = {}
        
        if responses_file.exists():
            with open(responses_file, 'rb') as f:
                all_responses = pickle.load(f)
        else:
            # Check if it's a remote job and fetch from JobStore
            status_info = self.check_status(job_id)
            if status_info.get("remote"):
                try:
                    from jobflow_remote.jobs.jobcontroller import JobController
                    jc = JobController.from_project_name(status_info.get("project", "remote_test"))
                    jobstore = jc.jobstore
                    
                    if job_id.isdigit():
                        flows_info = jc.get_flows_info(db_ids=[job_id])
                    else:
                        flows_info = jc.get_flows_info(flow_ids=[job_id])
                    
                    if not flows_info:
                        return {"error": "Remote flow not found."}

                    flow_info = flows_info[0]
                    # Fetch outputs from jobstore
                    results = []
                    # flow_info.jobs gives basic info about jobs in the flow
                    for job_info in flow_info.jobs:
                        jid = job_info.uuid
                        # We use jobstore.get_output directly if possible, or jc.get_job_output
                        doc = jobstore.get_output(jid)
                        if doc:
                            # doc is usually the output of the job (e.g. TaskDoc or similar)
                            # We need to handle both MSONable and standard dicts
                            from monty.json import jsanitize
                            doc_dict = jsanitize(doc)
                            
                            # Try to extract standard VASP result fields
                            # doc might be a TaskDoc-like object
                            output = doc_dict.get("output", {})
                            
                            entry = {
                                "flow_uuid": flow_info.uuid,
                                "job_uuid": jid,
                                "energy": doc_dict.get("energy", output.get("energy", output.get("final_energy"))),
                                "forces": doc_dict.get("forces", output.get("forces")),
                                "stress": doc_dict.get("stress", output.get("stress")),
                                "structure": doc_dict.get("structure")
                            }
                            # Convert numpy to list if needed
                            if entry["forces"] is not None:
                                entry["forces"] = np.array(entry["forces"]).tolist()
                            if entry["stress"] is not None:
                                entry["stress"] = np.array(entry["stress"]).tolist()
                            results.append(entry)
                    
                    return {"job_id": job_id, "results": results, "remote": True}
                except Exception as e:
                    return {"error": f"Failed to fetch remote results: {e}"}
            else:
                return {"error": "Results not yet available."}

        from atomate2.vasp.jobs.base import get_vasp_task_document
        from atomate2.vasp.drones import VaspDrone

        drone = VaspDrone()
        extracted = []

        for flow_uuid, data in all_responses.items():
            run_dir = data.get("dir")
            if not run_dir or not Path(run_dir).exists():
                continue

            # Find all valid VASP directories within this flow's run directory
            valid_paths = []
            for root, dirs, files in os.walk(run_dir):
                valid_paths.extend(drone.get_valid_paths((root, dirs, files)))

            for path in valid_paths:
                doc = get_vasp_task_document(path)
                output = doc.output
                
                entry = {
                    "flow_uuid": flow_uuid,
                    "job_uuid": getattr(doc, "task_id", None),
                    "dir_name": getattr(doc, "dir_name", str(path)),
                    "energy": getattr(output, 'energy', getattr(output, 'final_energy', None)),
                    "forces": getattr(output, 'forces', None),
                    "stress": getattr(output, 'stress', None),
                    "structure": doc.structure.as_dict() if hasattr(doc, 'structure') else None
                }
                
                # Convert numpy to list if needed
                if entry["forces"] is not None:
                    entry["forces"] = np.array(entry["forces"]).tolist()
                if entry["stress"] is not None:
                    entry["stress"] = np.array(entry["stress"]).tolist()
                
                extracted.append(entry)

        return {"job_id": job_id, "results": extracted}
