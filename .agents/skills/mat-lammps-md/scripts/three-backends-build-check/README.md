# LAMMPS MLIP Backend Matrix Check

## Goal
Verify that each MLIP stack has its own working LAMMPS binary and matching runtime environment.

## Steps

1. Build MACE-linked binary:
```bash
# Env: base
KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE86 \
bash pixi.toml (feature: mace) / install_lammps.sh
```

2. Build MatGL-linked binary:
```bash
# Env: base
KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE86 \
bash pixi.toml (feature: matgl) / install_lammps.sh
```

3. Build FairChem-linked binary:
```bash
# Env: base
KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE86 \
bash pixi.toml (feature: fairchem) / install_lammps.sh
```

4. Verify each binary with its matching environment:
```bash
# Env: mace
pixi shell -e mace
./lammps/mace/lmp -h

# Env: matgl
pixi shell -e matgl
./lammps/matgl/lmp -h

# Env: fairchem
pixi shell -e fairchem
./lammps/fairchem/lmp -h
```

## Expected Output
- All three binaries exist and start without Python embedding errors.
- Package summaries include Python and Kokkos support.
- No backend is run with a non-matching pixi environment.
