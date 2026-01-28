---
trigger: model_decision
description: Rules to implement a skill under `.agent/skills/`
---

# Skill Standards

All modular capabilities in this project should be implemented as "Skills" within the `.agent/skills/` directory. This rule ensures consistency and discoverability for the agent.

## Directory Structure

Each skill must reside in its own subdirectory:
- `.agent/skills/<skill-name>/`
  - `SKILL.md` (Required)
  - `scripts/` (Optional: Helper Python/Bash scripts)
  - `examples/` (Optional: Reference input/output files)
  - `resources/` (Optional: Configuration, data files, or templates)

## SKILL.md Format

The `SKILL.md` file must follow this structure:

### 1. YAML Frontmatter
```yaml
---
name: human-readable-unique-id
description: Concise one-sentence summary of the skill's purpose.
---
```

### 2. Main Body
The body should be organized using the following headers:

- **Goal**: What this skill specifically achieves.
- **Instructions**: Step-by-step logic for the agent to follow. Use links to scripts in the `scripts/` directory. You must specify which conda-env is required for each script.
- **Examples**: Short snippets or scenarios showing how to use the skill.
- **Constraints**: Limitations or safety rules (e.g., "Only run on supercells with >50 atoms").

## Best Practices

1. **Focus**: Each skill should target a single, well-defined task.
2. **Progressive Disclosure**: Keep the `SKILL.md` concise. Move detailed data or complex logic into `resources/` or `scripts/`.
3. **Relative Paths**: Always reference scripts and resources using relative paths from the `SKILL.md` file (e.g., `[plot.py](scripts/plot.py)`).
4. **Environment Awareness**: Specify which Conda environment is required for the scripts if they depend on specific packages (like `pymatgen` or `ase`).