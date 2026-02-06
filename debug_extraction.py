from smol.moca import SampleContainer
from smol.cofe import ClusterExpansion
import numpy as np

ce_file = "research/2026-02-05_smol_refactor_verification/smol/cluster_expansion.json"
traj_file = "research/2026-02-05_smol_refactor_verification/mc_trajectory.h5"

print("Loading CE...")
ce = ClusterExpansion.load(ce_file)
print("ClusterSubspace attributes:", dir(ce.cluster_subspace))

print("Loading Trajectory...")
try:
    container = SampleContainer.from_hdf5(traj_file)
    print("SampleContainer attributes:", dir(container))
    print("SampleContainer metadata:", container.metadata)
    if hasattr(container, "ensemble"):
        print("Container has ensemble.")
        print("Ensemble processor:", container.ensemble.processor)
    else:
        print("Container has NO ensemble.")
        
    occupancies = container.get_occupancies()
    print("Occupancy shape:", occupancies.shape)
    
except Exception as e:
    print("Error loading trajectory:", e)
