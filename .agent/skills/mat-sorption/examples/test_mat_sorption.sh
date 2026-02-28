#!/usr/bin/env bash

cd /home/alyssenko/AtomisticSkills

conda run -n fairchem-agent python .agent/skills/mat-sorption/scripts/run_widom_uma.py \
  --structure /home/alyssenko/AtomisticSkills/.agent/skills/mat-sorption/examples/Hb-DBD-AA_supercell.relaxed.cif \
  --name Hb-DBD-AA \
  --weights /home/shared/uma-s-1p1.pt \
  --task-name omol \
  --gas CO2 \
  --temperature 298 \
  --num-insertions 5000 \
  --output-dir .agent/skills/mat-sorption/examples

conda run -n fairchem-agent python .agent/skills/mat-sorption/scripts/run_gcmc_uma.py \
  --cif /home/alyssenko/AtomisticSkills/.agent/skills/mat-sorption/examples/Hb-DBD-AA_supercell.relaxed.cif \
  --output-dir .agent/skills/mat-sorption/examples \
  --weights /home/shared/uma-s-1p1.pt \
  --task-name odac \
  --steps 10000 \
  --temperature-K 298 \
  --pressure-bar 1.0 \
  --adsorbate CO2 \
  --scheme gcmc

conda run -n fairchem-agent python .agent/skills/mat-sorption/scripts/run_gcmc_uma_multi.py \
  --cif /home/alyssenko/AtomisticSkills/.agent/skills/mat-sorption/examples/Hb-DBD-AA_supercell.relaxed.cif \
  --output-dir .agent/skills/mat-sorption/examples \
  --weights /home/shared/uma-s-1p1.pt \
  --task-name odac \
  --steps 10000 \
  --temperature-K 298 \
  --gases CO2 N2 \
  --y 0.15 0.85 \
  --p-total-bar 1.0 \
  --scheme gcmc

