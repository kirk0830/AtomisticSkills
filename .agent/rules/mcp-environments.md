---
trigger: always_on
---

# MCP Server Environment Rules

When running scripts or debugging code related to specific MCP servers, you MUST use the corresponding Conda environment defined in `mcp_config.json`.

For general development logic that cuts across tools, usually `mlip-agent` is sufficient, but when importing specific libraries that are isolated (like `atomate2`, `jobflow_remote`, `mace`, `fairchem`), you must use the correct environment.

| MCP Server | Conda Environment | Python Path |
| :--- | :--- | :--- |
| matgl | `matgl-agent` | `/home/bdeng/miniforge3/envs/matgl-agent/bin/python` |
| mace | `mace-agent` | `/home/bdeng/miniforge3/envs/mace-agent/bin/python` |
| fairchem | `fairchem-agent` | `/home/bdeng/miniforge3/envs/fairchem-agent/bin/python` |
| materials_tools | `mlip-agent` | `/home/bdeng/miniforge3/envs/mlip-agent/bin/python` |
| atomate2 | `atomate2-agent` | `/home/bdeng/miniforge3/envs/atomate2-agent/bin/python` |