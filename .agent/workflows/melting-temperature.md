---
description: How to calculate the melting temperature of a material using MLIP and DFT.
---

# Workflow: Calculating Melting Temperature

This workflow describes the steps to calculate the melting temperature of a material by fine-tuning a foundation potential with DFT data.

1. **Research and Reference**: Perform a web search for relevant academic papers to obtain an approximate melting temperature for the target material.
2. **Structure Retrieval**: Query the material structure. If unavailable in exisisting database, construct the structure and relax it using a foundation MLIP. Save the relaxed structure as a `.cif` file locally.
3. **Benchmark and fine-tune**: Select a relevant foundation potential based on foundation-potential.md, and follow benchmark-fine-tune-mlip.md to benchmark and optionally fine-tune the foundation potential if error is high.
4. **Melting the material**: runing a high-temperature MD with the MLIP and melt the structure, make sure the structure have been melted by checking volume change and termostat log. The MD needs to be run with a orthorhombic supercell with at least 15A long in the long side.
5. **Create an interface**: Concatenate the melted supercell and the un-melted supercell in the long side. Use relax_structure mcp tool to relax the combined cell such that the interface is ionically relaxed.
6. **NVE MD**: Run a NVE-ensemble MD simulation of the combined supercell for 20 ps. The temperature of the NVE simulation needs to be around the melting temperature of the material, such that after the NVE MD equilibriates, a solid-liquid interface remains in the simulation cell. Check carefully that not everything solidifies or melts, otherwise raise or lower the NVE-MD temperature. If both solid and liquid coexist after the temperature equilibriates in the NVE simulation, the final temperature of the cell correspond to the melting temperature of the material.