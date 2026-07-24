"""AstrBot chatbot framework configuration."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from ..base import (
    PROJECT_ROOT,
    get_jinja_env,
    symlink_target,
    is_relative_to,
    symlink_dir_contents,
)
from ..env_block_rewriter import transform_env_blocks
from ..mcp_loader import load_mcp_servers

SKILLS_SRC = PROJECT_ROOT / ".agents" / "skills"
WORKFLOWS_SRC = PROJECT_ROOT / ".agents" / "workflows"
RULES_SRC = PROJECT_ROOT / ".agents" / "rules"
MCP_CONFIG_SRC = PROJECT_ROOT / "mcp_config.json"
MCP_SERVERS_SRC = PROJECT_ROOT / "src" / "mcp_server"

INDEX_SKILL_NAME = "atomisticskills"
PERSONA_FILE_NAME = "persona.md"


def detect_astrbot_data_dir(cli_path: str | None = None) -> Path:
    """Return the AstrBot data directory, or raise if not found."""
    if cli_path:
        return Path(cli_path).expanduser().resolve()

    env_path = None
    raw_env = os.environ.get("ASTRBOT_DATA_DIR")
    if raw_env:
        env_path = Path(raw_env).expanduser()

    candidates = [
        env_path,
        Path("data").resolve(),
        Path("..").resolve() / "astrbot" / "data",
        Path.home() / "astrbot" / "data",
    ]

    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not detect AstrBot data directory. "
        "Please pass it explicitly with --data-dir."
    )


def link_skills_to_astrbot(
    data_dir: Path, project_root: Path | None = None
) -> dict[str, int | list[str]]:
    """Symlink project skills into <data_dir>/skills and write the index skill."""
    if project_root is None:
        project_root = PROJECT_ROOT

    project_skills_dir = project_root / ".agents" / "skills"
    skills_dir = data_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    removed = _remove_stale_project_skill_dirs(skills_dir, project_skills_dir)

    linked = 0
    refreshed = 0
    skipped: list[str] = []
    conflicts: list[str] = []
    rewritten_skills = 0

    for project_skill in sorted(project_skills_dir.iterdir()):
        if not project_skill.is_dir():
            continue
        name = project_skill.name
        if name.startswith(".") or name.startswith("private-"):
            skipped.append(name)
            continue

        dst_path = skills_dir / name

        # Handle existing entries (including old symlinks from prior configs)
        if dst_path.exists() or dst_path.is_symlink():
            if dst_path.is_symlink():
                dst_path.unlink()
                linked += 1
            elif dst_path.is_dir():
                shutil.rmtree(dst_path)
                refreshed += 1
            else:
                conflicts.append(name)
        else:
            linked += 1

        # Copy entire skill directory tree (excluding SKILL.md — handled below)
        def _ignore_skill_md(d, files):
            return ["SKILL.md"] if "SKILL.md" in files else []

        shutil.copytree(
            project_skill,
            dst_path,
            ignore=_ignore_skill_md,
            symlinks=False,
            dirs_exist_ok=True,
        )

        # Rewrite and copy SKILL.md for AstrBot sandbox
        skill_src = project_skill / "SKILL.md"
        if skill_src.exists():
            _copy_and_rewrite(skill_src, dst_path / "SKILL.md", "skill")
            rewritten_skills += 1

    write_index_skill(skills_dir, project_root)

    return {
        "linked": linked,
        "refreshed": refreshed,
        "skipped": len(skipped),
        "removed_stale": removed,
        "conflicts": conflicts,
        "rewritten_skills": rewritten_skills,
    }


def _rewrite_for_sandbox(text: str, context: str) -> str:
    """Rewrite project-relative paths to AstrBot sandbox-relative paths.

    Converts ``@.agents/...`` syntax (Claude Code/Cursor) and ``.agents/...``
    paths into paths reachable within AstrBot's ``data/`` sandbox.

    Parameters:
        text: The markdown content to transform.
        context: Where the content lives in the sandbox —
            ``"rules"`` (``skills/atomisticskills/rules/``),
            ``"workflows"`` (``skills/atomisticskills/workflows/``),
            ``"skill"`` (``skills/<skill-name>/``).
    """
    # Map context → relative paths to other sandbox areas
    _up_to_skills = {"rules": "../..", "workflows": "../..", "skill": ".."}
    _up_to_index = {"rules": "..", "workflows": "..", "skill": "atomisticskills"}
    _rules_ref = {"rules": "", "workflows": "../rules/", "skill": "atomisticskills/rules/"}
    _workflows_ref = {"rules": "../workflows/", "workflows": "", "skill": "atomisticskills/workflows/"}

    up = _up_to_skills[context]
    up_index = _up_to_index[context]

    # 1. file:// absolute paths FIRST (before .agents/skills/ rule consumes them)
    text = re.sub(
        r'file://[^ )]*?\.agents/skills/([\w\-.]+)/([^ )]+)',
        lambda m: f"{up}/{m.group(1)}/{m.group(2)}",
        text,
    )

    # 2. @.agents/rules/xxx.md  →  xxx.md (same dir) or ../rules/xxx.md
    text = re.sub(
        r'@\.agents/rules/([\w\-.]+\.md)',
        lambda m: f"{_rules_ref[context]}{m.group(1)}",
        text,
    )

    # 3. @.agents/skills/  →  ../../ or ../
    text = re.sub(
        r'@\.agents/skills/',
        f"{up}/",
        text,
    )

    # 4. .agents/skills/<skill>/  →  ../<skill>/  (cross-skill references)
    text = re.sub(
        r'\.agents/skills/([\w\-.]+)/',
        lambda m: f"{up}/{m.group(1)}/",
        text,
    )

    # 5. .agents/workflows/<file>  →  workflows/<file> or ../workflows/<file>
    text = re.sub(
        r'\.agents/workflows/([\w\-.]+\.md)',
        lambda m: f"{_workflows_ref[context]}{m.group(1)}",
        text,
    )

    # 6. .agents/rules/<file>  →  rules/<file> or ../rules/<file>
    #    (non-@ references, e.g. in prose like "see .agents/rules/...")
    text = re.sub(
        r'\.agents/rules/([\w\-.]+\.md)',
        lambda m: f"{_rules_ref[context]}{m.group(1)}",
        text,
    )

    # 7. .agents/templates/  →  remove (templates are build-time only)
    text = re.sub(r'\.agents/templates/', '', text)

    # 8. ../skills/<skill>/  →  ../../<skill>/  (workflow context only)
    #    Workflows at data/skills/atomisticskills/workflows/ reference skills via
    #    ../skills/<name>/SKILL.md. In the sandbox layout, skills are siblings under
    #    data/skills/, so the correct path is ../../<name>/SKILL.md.
    if context == "workflows":
        text = re.sub(
            r'\.\./skills/([\w\-.]+)/',
            r'../../\1/',
            text,
        )

    # 9. docs/<file>  →  remove (external docs not available in sandbox)
    #    References like docs/hpc_job_submission.md are not copied into the
    #    sandbox. Replace with a note to consult the project site.
    text = re.sub(
        r'`docs/([^`]+)`',
        r'(see project documentation for `docs/\1`)',
        text,
    )
    text = re.sub(
        r'\[([^\]]+)\]\(docs/([^)]+)\)',
        r'\1 (project documentation)',
        text,
    )

    # 10. ~/.atomistic_skills.yaml  →  keep as-is
    #     In local mode with administrator access, ~ resolves to the real home
    #     directory, so this reference remains valid. No rewrite needed.

    # 11. # Env: code blocks → rewrite as pre-filled mcp_pixi_run() calls.
    #     The Agent in astrbot (non-admin) has no shell/Python — every
    #     ``# Env:`` command must go through the pixi MCP server.  Instead
    #     of asking the LLM to parse command text and guess parameters, we
    #     rewrite the block so the Agent sees a ready-to-copy MCP call with
    #     all fields pre-filled.  Script paths are restored from sandbox-
    #     relative (``../...``) back to project-relative (``.agents/skills/...``)
    #     so the pixi server can find them.
    text = transform_env_blocks(text)

    return text


def _copy_and_rewrite(src: Path, dst: Path, context: str) -> None:
    """Copy *src* to *dst*, rewriting paths for the AstrBot sandbox."""
    content = src.read_text(encoding="utf-8")
    rewritten = _rewrite_for_sandbox(content, context)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(rewritten, encoding="utf-8")


def _remove_stale_project_skill_dirs(
    skills_dir: Path,
    project_skills_dir: Path,
) -> int:
    """Remove directories (or old symlinks) whose project skill no longer exists."""
    removed = 0
    if not skills_dir.exists():
        return removed

    for entry in skills_dir.iterdir():
        if entry.name == INDEX_SKILL_NAME:
            continue

        if not entry.is_symlink():
            # Real directory — remove if the source skill is gone
            if entry.is_dir() and not (project_skills_dir / entry.name).is_dir():
                shutil.rmtree(entry)
                removed += 1
            continue

        # Old symlink-based entry — clean up
        target = symlink_target(entry)
        if is_relative_to(target, project_skills_dir) and not target.exists():
            entry.unlink()
            removed += 1

    return removed


def write_index_skill(skills_dir: Path, project_root: Path) -> None:
    """Write the atomisticskills index SKILL.md and sandbox-safe sub-resources.

    Rules and workflows are **copied** (not symlinked) so that project-relative
    paths (``@.agents/...``, ``.agents/...``) can be rewritten into sandbox-
    relative paths that AstrBot can actually follow.
    """
    index_dir = skills_dir / INDEX_SKILL_NAME
    index_dir.mkdir(parents=True, exist_ok=True)

    # Workflows: copy + rewrite paths
    workflows_dir = index_dir / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    if WORKFLOWS_SRC.exists():
        for wf_file in WORKFLOWS_SRC.iterdir():
            if wf_file.suffix == ".md":
                _copy_and_rewrite(wf_file, workflows_dir / wf_file.name, "workflows")

    # Rules: copy + rewrite paths
    rules_dir = index_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    if RULES_SRC.exists():
        for rule_file in RULES_SRC.iterdir():
            if rule_file.suffix == ".md":
                _copy_and_rewrite(rule_file, rules_dir / rule_file.name, "rules")

    mcpservers_dir = index_dir / "mcpservers"
    mcpservers_dir.mkdir(parents=True, exist_ok=True)
    _symlink_mcp_resources(mcpservers_dir, project_root)

    skill_file = index_dir / "SKILL.md"
    skill_file.write_text(
        _build_index_skill_md(project_root, index_dir),
        encoding="utf-8",
    )


def _symlink_mcp_resources(mcpservers_dir: Path, project_root: Path) -> None:
    """Copy MCP-related resources into the mcpservers/ subdirectory.

    Uses physical copies (not symlinks) because AstrBot's sandbox resolves
    symlinks and then blocks access to paths outside the workspace.
    """
    # --- mcp_config.json ---
    config_dst = mcpservers_dir / "mcp_config.json"
    if MCP_CONFIG_SRC.exists():
        # Remove any existing entry (symlink, file, or directory)
        if config_dst.exists() or config_dst.is_symlink():
            config_dst.unlink()
        shutil.copy2(MCP_CONFIG_SRC, config_dst)

    # --- servers/ ---
    servers_dst = mcpservers_dir / "servers"
    if MCP_SERVERS_SRC.exists():
        # Remove any existing entry
        if servers_dst.exists() or servers_dst.is_symlink():
            if servers_dst.is_dir() and not servers_dst.is_symlink():
                shutil.rmtree(servers_dst)
            else:
                servers_dst.unlink()
        shutil.copytree(
            MCP_SERVERS_SRC,
            servers_dst,
            symlinks=False,
            dirs_exist_ok=True,
        )


def _build_index_skill_md(project_root: Path, index_dir: Path) -> str:
    """Return the content of the atomisticskills index SKILL.md."""
    env = get_jinja_env()
    if env is not None:
        template = env.get_template("skill/astrbot_skill.md.j2")
        return template.render(
            project_root=str(project_root),
            index_dir=str(index_dir),
        )

    return _build_index_skill_fallback(project_root, index_dir)


def _build_index_skill_fallback(project_root: Path, index_dir: Path) -> str:
    """Fallback implementation when Jinja2 is not available."""
    return f"""\
---
name: atomisticskills
description: Use AtomisticSkills for atomistic research, materials simulation, molecular modeling, spectroscopy, MLIP, drug discovery, and scientific workflow tasks. Contains rules, workflows, MCP server info, and all skill references. This is the PRIMARY entry point — always read this first.
---

# AtomisticSkills — Primary Reference

> **🔴 CRITICAL: This is the ROOT skill. Read this first, always, before any other skill or action.**
>
> This skill is the **single entry point** into the AtomisticSkills framework. It sits
> **above** all other skills and must be consulted before any task. It contains the
> **rules**, **workflows**, **MCP server documentation**, and a categorized index of
> all individual research skills.

## 📖 Progressive Disclosure — Read in This Order

When processing any user request, follow this strict reading order:

```
1. THIS SKILL (atomisticskills)     ← you are here
2. Rules (linked below)             ← protocols, coding standards, environments
3. Workflows (linked below)         ← end-to-end research blueprints
4. Individual Skills                ← use only when no workflow matches
```

**Never skip a level.** Do not jump directly to an individual skill without first
checking whether a workflow covers the goal. Do not start any research task without
first reading the rules.

The AtomisticSkills skills are installed in your skills directory.

---

## 🔴 Before Doing Anything — Read These Rules

These rules define how you operate as an AtomisticSkills research agent.
**Read them now and re-read whenever you are unsure about protocol.**

| Rule File | Purpose |
|-----------|---------|
| [research-standards.md](research-standards.md) | **Core research protocol** — intent classification, research plan workflow, artifact rules |
| [coding-standards.md](coding-standards.md) | Coding conventions, error handling, MCP stability rules |
| [mcp-environments.md](mcp-environments.md) | Which Pixi environment to use for each MCP server |
| [skill-standards.md](skill-standards.md) | How to read and follow a skill |
| [workflow-standards.md](workflow-standards.md) | How to execute a multi-step workflow |
| [plot-standards.md](plot-standards.md) | Plotting and visualization conventions |
| [hpc-standards.md](hpc-standards.md) | HPC/Slurm job submission rules — never run heavy computation locally |

All rule files are also available as symlinks in the `rules/` subdirectory
next to this SKILL.md:
```
{index_dir / 'rules'}
```

---

## 📋 Workflows — End-to-End Research Protocols

Workflows are complete research blueprints that chain multiple skills.
**Always check if a workflow matches the user's goal before assembling steps
from individual skills.**

Available workflows (also symlinked in `workflows/`):

| Workflow | Description |
|----------|-------------|
| [materials-discovery.md](workflows/materials-discovery.md) | General materials discovery campaign using MLIP + DFT validation |
| [sorption-discovery.md](workflows/sorption-discovery.md) | Gas sorption material screening in porous frameworks |
| [mof-co2-dac-screening.md](workflows/mof-co2-dac-screening.md) | MOF screening for CO₂ direct air capture |
| [drug-hit-finding-htvs.md](workflows/drug-hit-finding-htvs.md) | High-throughput virtual screening for drug hit discovery |
| [generative-halide-discovery.md](workflows/generative-halide-discovery.md) | Generative AI + MLIP for halide perovskite discovery |
| [mlip-benchmark-finetune.md](workflows/mlip-benchmark-finetune.md) | MLIP benchmarking and fine-tuning workflow |
| [nmr-reaction-kinetics.md](workflows/nmr-reaction-kinetics.md) | NMR-based reaction kinetics analysis |
| [reaction-to-nmr-quantification.md](workflows/reaction-to-nmr-quantification.md) | Reaction analysis with NMR quantification |
| [image-to-xrd-phase.md](workflows/image-to-xrd-phase.md) | XRD phase identification from digitized plot images |

All workflow files are also available as symlinks in the `workflows/`
subdirectory next to this SKILL.md:
```
{index_dir / 'workflows'}
```

---

## 🛠 MCP Servers — Computational Tools

MCP servers provide the low-level tools (relaxation, MD, database queries, etc.)
that skills and workflows orchestrate. Server configs must be added separately
in AstrBot's WebUI (see the `mcpservers/` directory for reference).

| Server | Environment | Capabilities |
|--------|-------------|--------------|
| `base` | base | Structure handling, Materials Project, PubChem, ChEMBL, PDB, literature, HPC submission |
| `mace` | mace | MACE MLIP: relaxation, MD, phonons, energy/force predictions |
| `matgl` | matgl | MatGL (CHGNet, M3GNet, TensorNet): relaxation, MD, properties |
| `fairchem` | fairchem | FairChem (UMA/ESEN) MLIP predictions |
| `smol` | smol | Cluster expansion, Monte Carlo simulations |
| `drugdisc` | drugdisc | Molecular docking, protein prep, ADMET, fingerprints |
| `adit` | adit | ADiT all-atom diffusion transformer generation |
| `diffcsp` | diffcsp | DiffCSP++ crystal structure generation |
| `mattergen` | mattergen | MatterGen generative crystal design |

MCP reference files are in the `mcpservers/` subdirectory next to this SKILL.md:
```
{index_dir / 'mcpservers'}
```
- `mcpservers/mcp_config.json` — full server config template
- `mcpservers/servers/` — source code of each MCP server

---

## 🧪 Individual Skills

All individual skills are symlinked directly in AstrBot's skills directory:
```
{index_dir.parent}
```

Scan `SKILL.md` files to find the right skill for each sub-task. Each skill
contains step-by-step instructions, helper scripts, and examples.

**Skill categories:**
- **Materials (mat-*)**: Structure, stability, phonons, diffusion, defects, surfaces, phase diagrams, XRD, etc.
- **Chemistry (chem-*)**: Molecular DFT, bonding, conformers, spectroscopy, sorption, solution MD, etc.
- **Drug Discovery (drug-*)**: Docking, ADMET, MD, retrosynthesis, fingerprints, etc.
- **Machine Learning (ml-*)**: MLIP benchmarking, fine-tuning, generative models, property prediction.
- **General (general-*)**: Literature search, peer review, plotting, presentations, etc.

---

## Operating Principles

1. **Rules first** — always know and follow the research standards and coding standards.
2. **Workflow before skills** — check if a workflow covers the goal before assembling individual skills.
3. **Skills before custom code** — prefer existing skills and MCP tools; only write custom scripts when nothing fits.
4. **Plan before acting** — for computational research tasks, create a research plan artifact and get user approval.
5. **Use the right tool** — pick the correct MCP server and environment for each operation.

If the AtomisticSkills skills are already available in your skills directory,
use them directly instead of searching for external copies.
"""


def generate_astrbot_mcp_configs(
    project_root: Path,
    use_uv: bool,
) -> dict[str, dict[str, Any]]:
    """Load MCP configs and format them for AstrBot."""
    servers = load_mcp_servers(pixi_root=str(project_root))
    result: dict[str, dict[str, Any]] = {}

    for name, cfg in servers.items():
        if not _python_exists_for_server(cfg):
            print(
                f"  [WARN] {name}: interpreter not found ({cfg.get('command')}). "
                f"Run 'pixi install -e {name}' first.",
                file=sys.stderr,
            )

        if use_uv:
            result[name] = _to_uv_form(name, cfg)
        else:
            result[name] = dict(cfg)

    return result


def _python_exists_for_server(server_cfg: dict[str, Any]) -> bool:
    """Return True if the server's configured Python interpreter exists."""
    cmd = server_cfg.get("command", "")
    return cmd and Path(cmd).exists()


def _to_uv_form(server_name: str, server_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert a direct-path server config to AstrBot's env + uv form."""
    python_path = server_cfg["command"]
    args = list(server_cfg.get("args", []))

    env_args = []
    for key, value in server_cfg.get("env", {}).items():
        env_args.append(f"{key}={value}")

    uv_args = [
        *env_args,
        "uv",
        "run",
        "--python",
        str(python_path),
    ]

    uv_args.append("python")
    uv_args.extend(args)

    return {"command": "env", "args": uv_args}


def print_mcp_configs(servers: dict[str, dict[str, Any]]) -> None:
    """Print per-server JSON blocks for copy-paste into AstrBot WebUI."""
    print("\n=== AstrBot MCP Server Configs ===")
    print("Paste each block into AstrBot WebUI -> MCP -> Add MCP Server.\n")

    for name, cfg in servers.items():
        print(f"--- {name} ---")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        print()


def write_mcp_config_file(
    data_dir: Path, servers: dict[str, dict[str, Any]]
) -> Path:
    """Save all MCP configs to <data_dir>/config/atomisticskills_mcp.json."""
    config_dir = data_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "atomisticskills_mcp.json"
    config_file.write_text(
        json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_file


def build_persona_md(project_root: Path) -> str:
    """Return the content of persona.md for AstrBot personality settings."""
    env = get_jinja_env()
    if env is not None:
        template = env.get_template("persona/astrbot_persona.md.j2")
        return template.render(project_root=str(project_root))

    return _build_persona_fallback(project_root)


def _build_persona_fallback(project_root: Path) -> str:
    """Fallback implementation when Jinja2 is not available."""
    return f"""\
# AtomisticSkills Research Agent Persona

## Role

你是一名专业的**原子尺度材料研究智能体 (AtomisticSkills Research Agent)**，基于 AtomisticSkills 框架为用户提供材料科学、计算化学、药物发现和分子模拟领域的专业研究支持。

## Core Identity

- **专业领域**: 材料科学、计算化学、凝聚态物理、药物发现、分子模拟
- **方法论**: 基于第一性原理 (DFT)、机器学习原子间势 (MLIP)、分子动力学 (MD)、蒙特卡洛 (MC) 方法
- **工作方式**: 遵循严格的研究规范，先规划后执行，优先使用已有 Skills 和 Workflows
- **语言风格**: 专业、严谨、清晰，适当使用科学术语，确保可复现性

## Knowledge Domains

### 材料科学 (Materials Science)
- 晶体结构、缺陷、表面、界面、晶界
- 热力学稳定性、相图、形成能、凸包分析
- 弹性模量、力学性能、热膨胀
- 晶格热导率、声子谱
- 离子扩散、激活能、NEB 势垒
- 电化学稳定性窗口、嵌入电压
- XRD 图谱计算、物相分析、Rietveld 精修
- 拉曼光谱、振动分析
- 合成路线推荐、文献数据挖掘

### 计算化学 (Computational Chemistry)
- 分子 DFT 计算 (ORCA)
- 键解离能 (BDE)、均裂/异裂
- 构象搜索与生成
- 过渡态搜索与验证 (IRC)
- 溶液相分子动力学
- 热力学量计算 (H, S, G)
- NMR 谱预测与分析
- MS/MS 谱预测
- 红外光谱匹配

### 药物发现 (Drug Discovery)
- 蛋白结构准备与预处理
- 结合位点定义与口袋检测
- 分子对接 (AutoDock Vina)
- 打分函数与结合自由能计算 (MM-GBSA/PBSA)
- ADMET 性质预测
- 分子指纹与相似性搜索
- 虚拟筛选 (HTVS)
- 蛋白-配体分子动力学 (OpenMM)
- 轨迹分析 (RMSD、RMSF、氢键、接触频率)
- 逆合成分析
- 生物活性数据查询 (ChEMBL, PubChem, PDB)

### 机器学习原子间势 (MLIP)
- MACE, MatGL (CHGNet, M3GNet, TensorNet), FairChem (UMA/ESEN)
- 结构弛豫、单点能计算、分子动力学
- MLIP 基准测试与精度评估
- 迁移学习与微调
- 不确定性量化 (Committee models)
- 生成式模型: MatterGen, DiffCSP++, ADiT
- 性质预测器训练

### 高级模拟方法
- 经典分子动力学 (LAMMPS)
- 元动力学、伞形采样
- 相场模拟 (Allen-Cahn, Cahn-Hilliard)
- 动力学蒙特卡洛 (kMC)
- 巨正则蒙特卡洛 (GCMC)
- 聚类展开 + SMOL
- CALPHAD 相图计算
- 固体自由能计算 (Frenkel-Ladd)
- 熔点计算 (固液共存法)

## Operating Principles

### 1. 研究流程规范
- **意图分类优先**: 先判断用户请求是直接数据查询、计算研究任务，还是广泛文献综述
- **直接查询**: 直接调用 MCP 工具或数据库，无需创建研究目录和计划
- **计算研究**: 必须先创建研究计划 (research_plan.md artifact)，经用户确认后执行
- **文献综述**: 提供详细的技术综述，最后询问是否需要启动正式研究项目

### 2. 工具使用层级
- **优先使用 Workflow**: 检查是否有现成的工作流覆盖用户目标
- **其次使用 Skill**: 使用针对性的 Skill 解决子任务
- **最后自定义代码**: 当没有合适的 Skill/Tool 时才编写自定义脚本

### 3. 质量保证
- 所有计算结果注明方法和参数
- 关键结果建议验证方法
- 生成的图像必须进行视觉检查
- 保持代码清洁、模块化，遵循编码规范
- MCP 工具调用出现问题时，优先调试工具而非绕过

### 4. 沟通风格
- 用中文回复用户，专业术语可保留英文
- 步骤清晰，结果明确
- 主动报告进展和遇到的问题
- 提供合理的后续建议
- 对于长周期任务 (如 HPC 计算)，及时告知用户任务已提交并说明如何查询状态

## Framework Reference

本智能体基于 AtomisticSkills 框架运行:
- **Skills 目录**: `skills/`（所有技能模块在此目录下）
- **主入口**: `skills/atomisticskills/SKILL.md`（包含规则、Workflows、MCP 服务器文档）
- **Skills**: 100+ 个专业研究技能模块
- **Workflows**: 端到端研究方案
- **MCP Servers**: base, mace, matgl, fairchem, smol, drugdisc, adit, diffcsp, mattergen 等
- **环境管理**: Pixi 隔离环境

## 启动检查清单

每次对话开始时，确保你已经:
1. 阅读了 atomisticskills skill 中的规则文件 (research-standards, coding-standards, mcp-environments)
2. 了解了可用的 Workflows
3. 知道如何调用 MCP 工具执行计算
4. 准备好根据用户请求选择合适的研究路径
"""


def write_persona_file(data_dir: Path, project_root: Path) -> Path:
    """Write persona.md to the data directory for copy-paste into AstrBot."""
    persona_path = data_dir / PERSONA_FILE_NAME
    persona_path.write_text(build_persona_md(project_root), encoding="utf-8")
    return persona_path


def print_persona_prompt(persona_path: Path) -> None:
    """Print instructions for setting the persona in AstrBot."""
    print("\n" + "=" * 60)
    print("=== AstrBot 人格设定 (Persona) ===")
    print("=" * 60)
    print(f"已生成人格文件: {persona_path}")
    print()
    print("请按以下步骤设置 AstrBot 人格:")
    print("  1. 打开 AstrBot WebUI")
    print("  2. 进入 人格 / Persona 设置页面")
    print("  3. 将上述文件中的内容复制粘贴到人格设定框中")
    print("  4. 保存并重启 AstrBot 以生效")
    print()
    print("该人格设定将使智能体以专业的原子材料研究人员身份工作，")
    print("遵循研究规范，优先使用 Skills 和 Workflows，并保持专业的沟通风格。")
    print("=" * 60)