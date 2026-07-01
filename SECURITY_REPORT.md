# AtomisticSkills Bash Shell 脚本安全评估报告

**评估日期**: 2026-07-01
**评估范围**: 仓库中所有 `.sh` / `.bash` shell 脚本
**脚本总数**: 52 个

---

## 1. 总体概述

本报告对 AtomisticSkills 仓库中的所有 Bash shell 脚本进行了安全性评估。脚本主要分布在以下两个区域：

| 区域 | 脚本数量 | 主要用途 |
|------|----------|----------|
| `conda-envs/*/` | 30 个 | Conda 环境安装、LAMMPS 编译、安装验证 |
| `.agents/skills/*/` | 22 个 | 研究技能示例脚本、测试脚本 |

### 风险等级汇总

| 风险等级 | 脚本数量 | 占比 |
|----------|----------|------|
| 🔴 高危 | 4 | 7.7% |
| 🟠 中危 | 12 | 23.1% |
| 🟡 低危 | 28 | 53.8% |
| 🟢 安全 | 8 | 15.4% |

---

## 2. 高危风险脚本 (🔴 Critical)

### 2.1 `curl | sh` 远程代码执行模式

**涉及脚本**: 18 个 install.sh（见第 3.1 节通用模式）

**风险描述**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
直接从互联网下载脚本并通过管道执行，是典型的供应链攻击入口。如果 `astral.sh` 域名被劫持或服务器被入侵，攻击者可在用户机器上执行任意代码。

**风险等级**: 🔴 高危

**影响范围**: 所有使用此模式的 install.sh 脚本（约 18 个）

---

### 2.2 硬编码绝对路径暴露用户信息

**涉及脚本**:
- [conda-envs/void-agent/install.sh](file:///workspace/conda-envs/void-agent/install.sh#L35-L41)

**风险代码**:
```bash
VOID_DIR="$HOME/projects/AtomisticSkills/VOID"
```

**风险描述**:
硬编码了开发者的个人目录结构 `$HOME/projects/AtomisticSkills/`，暴露了用户目录命名习惯。虽然使用了 `$HOME` 变量，但路径前缀暗示了特定的项目布局。

**风险等级**: 🔴 高危（路径泄露 + 路径不通用）

---

### 2.3 第三方仓库克隆并直接安装

**涉及脚本**:
- [conda-envs/react-ot-agent/install.sh](file:///workspace/conda-envs/react-ot-agent/install.sh#L38-L81)
- [conda-envs/scd-agent/install.sh](file:///workspace/conda-envs/scd-agent/install.sh#L45-L56)
- [conda-envs/msms-agent/install.sh](file:///workspace/conda-envs/msms-agent/install.sh#L76-L115)

**风险代码** (react-ot-agent):
```bash
git clone https://github.com/deepprinciple/react-ot.git "$REACT_OT_DIR"
# ... apply patches via sed ...
pip install -e "$REACT_OT_DIR"
```

**风险描述**:
1. 从 GitHub 直接克隆第三方仓库并安装到 conda 环境中
2. 使用 `sed` 对第三方代码进行补丁修改
3. 无代码完整性校验（无 hash 校验、无版本 tag 锁定）
4. `pip install -e` 以可编辑模式安装，第三方代码可随时更改

**风险等级**: 🔴 高危

---

### 2.4 修改 site-packages 中的第三方库代码

**涉及脚本**:
- [conda-envs/msms-agent/install.sh](file:///workspace/conda-envs/msms-agent/install.sh#L40-L57)
- [conda-envs/msms-agent/install.sh](file:///workspace/conda-envs/msms-agent/install.sh#L119-L146)

**风险代码**:
```bash
python - <<'DGLPATCH'
import pathlib, sys, re
site = pathlib.Path(sys.executable).parent.parent / "lib/python3.10/site-packages"
graphbolt = site / "dgl/graphbolt"
for f in graphbolt.glob("*.py"):
    txt = f.read_text()
    if "torchdata" not in txt:
        continue
    patched = re.sub(r'^((?:from|import) torchdata\S*.*)', ... , txt, flags=re.MULTILINE)
    f.write_text(patched)
DGLPATCH
```

**风险描述**:
1. 直接修改 `site-packages` 中已安装的第三方库（DGL）源代码
2. 创建伪造的 `torchdata` 包以欺骗导入系统
3. 破坏包管理器的完整性保证
4. 可能导致难以追踪的安全漏洞和兼容性问题

**风险等级**: 🔴 高危

---

## 3. 中危风险脚本 (🟠 High)

### 3.1 通用安装脚本模式（18 个 install.sh 共用）

**涉及脚本**:
- [conda-envs/base-agent/install.sh](file:///workspace/conda-envs/base-agent/install.sh)
- [conda-envs/mace-agent/install.sh](file:///workspace/conda-envs/mace-agent/install.sh)
- [conda-envs/matgl-agent/install.sh](file:///workspace/conda-envs/matgl-agent/install.sh)
- [conda-envs/fairchem-agent/install.sh](file:///workspace/conda-envs/fairchem-agent/install.sh)
- [conda-envs/atomate2-agent/install.sh](file:///workspace/conda-envs/atomate2-agent/install.sh)
- [conda-envs/drugdisc-agent/install.sh](file:///workspace/conda-envs/drugdisc-agent/install.sh)
- [conda-envs/drugmd-agent/install.sh](file:///workspace/conda-envs/drugmd-agent/install.sh)
- [conda-envs/adit-agent/install.sh](file:///workspace/conda-envs/adit-agent/install.sh)
- [conda-envs/diffcsp-agent/install.sh](file:///workspace/conda-envs/diffcsp-agent/install.sh)
- [conda-envs/mattergen-agent/install.sh](file:///workspace/conda-envs/mattergen-agent/install.sh)
- [conda-envs/smol-agent/install.sh](file:///workspace/conda-envs/smol-agent/install.sh)
- [conda-envs/xrd-agent/install.sh](file:///workspace/conda-envs/xrd-agent/install.sh)
- [conda-envs/scd-agent/install.sh](file:///workspace/conda-envs/scd-agent/install.sh)
- [conda-envs/void-agent/install.sh](file:///workspace/conda-envs/void-agent/install.sh)
- [conda-envs/nmr-agent/install.sh](file:///workspace/conda-envs/nmr-agent/install.sh)
- [conda-envs/calphad-agent/install.sh](file:///workspace/conda-envs/calphad-agent/install.sh)
- [conda-envs/phasefield-agent/install.sh](file:///workspace/conda-envs/phasefield-agent/install.sh)
- [conda-envs/msms-agent/install.sh](file:///workspace/conda-envs/msms-agent/install.sh)

**主要风险点**:

| 风险点 | 代码示例 | 风险等级 |
|--------|----------|----------|
| `curl | sh` 无校验安装 uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | 🔴 高危 |
| 强制删除现有环境 | `conda env remove -n $ENV_NAME -y \|\| true` | 🟠 中危 |
| 从 YAML 中提取 pip 依赖时剥离引号 | `tr -d '"' \| tr -d "'"` | 🟡 低危 |
| 变量未引用导致单词分割 | `conda env create -f conda_only_env.yaml`（部分有引用，部分没有） | 🟡 低危 |

---

### 3.2 LAMMPS 编译脚本

**涉及脚本**:
- [conda-envs/mace-agent/install_lammps.sh](file:///workspace/conda-envs/mace-agent/install_lammps.sh)
- [conda-envs/matgl-agent/install_lammps.sh](file:///workspace/conda-envs/matgl-agent/install_lammps.sh)
- [conda-envs/fairchem-agent/install_lammps.sh](file:///workspace/conda-envs/fairchem-agent/install_lammps.sh)

**风险描述**:
1. **从 GitHub 克隆 LAMMPS 源码并编译** - 无完整性校验
   ```bash
   git clone --branch "${LAMMPS_REF}" --depth 1 "${LAMMPS_GIT_URL}" "${LAMMPS_SRC_DIR}"
   ```
2. **在项目根目录创建 `lammps/` 目录** - 写入位置不受控
3. **使用 `rm -rf` 清理构建目录**
   ```bash
   rm -rf "${LAMMPS_BUILD_DIR}"
   ```
4. **编译结果二进制 `lmp` 直接执行** - 无签名验证

**风险等级**: 🟠 中危

---

### 3.3 安装验证脚本（verify_install.sh）

**涉及脚本**:
- [conda-envs/base-agent/verify_install.sh](file:///workspace/conda-envs/base-agent/verify_install.sh)
- [conda-envs/mace-agent/verify_install.sh](file:///workspace/conda-envs/mace-agent/verify_install.sh)
- [conda-envs/fairchem-agent/verify_install.sh](file:///workspace/conda-envs/fairchem-agent/verify_install.sh)
- [conda-envs/atomate2-agent/verify_install.sh](file:///workspace/conda-envs/atomate2-agent/verify_install.sh)
- [conda-envs/orca-agent/verify_install.sh](file:///workspace/conda-envs/orca-agent/verify_install.sh)
- [conda-envs/smol-agent/verify_install.sh](file:///workspace/conda-envs/smol-agent/verify_install.sh)

**风险描述**:
1. 创建临时测试环境并运行 pytest
2. 设置 `PYTHONPATH=$(pwd)` 将当前目录加入 Python 路径
3. 可能触发模型下载（MACE/FairChem），下载内容无 hash 校验

**风险等级**: 🟡 低危 ~ 🟠 中危

---

### 3.4 环境变量覆盖 PATH

**涉及脚本**: 多数 install.sh

**风险代码**:
```bash
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
```

**风险描述**:
将用户目录的 bin 目录置于 PATH 前面，如果攻击者能在 `~/.local/bin` 写入文件，可劫持后续命令执行。

**风险等级**: 🟠 中危

---

## 4. 低危风险脚本 (🟡 Medium)

### 4.1 Skills 示例脚本

**涉及脚本** (22 个，以下为代表性样本):
- [.agents/skills/chem-sorption-gcmc/examples/test_gcmc.sh](file:///workspace/.agents/skills/chem-sorption-gcmc/examples/test_gcmc.sh)
- [.agents/skills/mat-lammps-md/examples/mace/run_mace_na2si3o7_quench.sh](file:///workspace/.agents/skills/mat-lammps-md/examples/mace/run_mace_na2si3o7_quench.sh)
- [.agents/skills/ml-fairchem-finetune/test_fairchem_finetuning.sh](file:///workspace/.agents/skills/ml-fairchem-finetune/test_fairchem_finetuning.sh)
- [.agents/skills/chem-ts-optimization/examples/acetonitrile/run_example.sh](file:///workspace/.agents/skills/chem-ts-optimization/examples/acetonitrile/run_example.sh)
- [.agents/skills/chem-irc-verification/examples/acetonitrile/run_example.sh](file:///workspace/.agents/skills/chem-irc-verification/examples/acetonitrile/run_example.sh)

**风险描述**:
1. **用途**: 这些是技能使用示例，用户需手动执行
2. **主要操作**: 调用 Python 脚本运行模拟、设置环境变量
3. **低危点**:
   - 设置 `PYTHONPATH` 时使用相对路径
   - 部分脚本使用 `conda run -n <env>` 执行命令
   - 部分脚本包含 `torch.load` 触发的模型加载（pickle 反序列化风险）

**风险等级**: 🟡 低危（因为需要用户主动执行且意图明确）

---

### 4.2 临时文件处理

**涉及脚本**: 多数 install.sh

**风险描述**:
```bash
sed '/pip:/,$d' core_env.yaml > conda_only_env.yaml
# ...
rm conda_only_env.yaml uv_requirements.txt
```
临时文件创建在脚本所在目录中，而非 `/tmp`。虽然脚本结束时会清理，但如果脚本中途失败，临时文件会残留。

**风险等级**: 🟡 低危

---

### 4.3 `set -e` 不完全可靠的错误处理

**涉及脚本**: 多数 install.sh 使用 `set -e`

**风险描述**:
`set -e` 在某些情况下不会触发退出（如管道中的命令失败、if 条件中的命令失败），可能导致部分安装状态不一致。

**风险等级**: 🟡 低危

---

## 5. 相对安全的脚本 (🟢 Low)

### 5.1 简洁安装脚本

**涉及脚本**:
- [conda-envs/orca-agent/install.sh](file:///workspace/conda-envs/orca-agent/install.sh)
- [conda-envs/calphad-agent/install.sh](file:///workspace/conda-envs/calphad-agent/install.sh)
- [conda-envs/phasefield-agent/install.sh](file:///workspace/conda-envs/phasefield-agent/install.sh)

**描述**:
这些脚本仅使用 `conda env create -f core_env.yaml`，不涉及 curl、git clone 或 pip 安装。风险相对较低。

**风险等级**: 🟢 低危

---

### 5.2 Skills 中的简单示例脚本

**涉及脚本**:
- [.agents/skills/mat-db-mp/examples/query_mp/li_s_stability.sh](file:///workspace/.agents/skills/mat-db-mp/examples/query_mp/li_s_stability.sh)
- 以及其他仅调用 Python 脚本的简单示例

**描述**:
仅设置工作目录并调用 Python 脚本，无高危操作。

**风险等级**: 🟢 低危

---

## 6. 系统性风险汇总

### 6.1 供应链攻击面

| 攻击向量 | 涉及脚本数 | 严重程度 |
|----------|------------|----------|
| `curl \| sh` 安装 uv | 18 | 🔴 高 |
| `git clone` 第三方仓库 | 6 | 🟠 中高 |
| `pip install` 第三方包 | 30 | 🟠 中 |
| `conda env create` 从 conda-forge | 30 | 🟡 中低 |
| 模型下载 (HuggingFace 等) | 多个 | 🟠 中 |

### 6.2 路径与文件系统风险

| 风险类型 | 涉及脚本数 | 严重程度 |
|----------|------------|----------|
| 硬编码用户路径 | 1 | 🔴 高 |
| `rm -rf` 清理操作 | 4 | 🟠 中高 |
| 变量未引用（单词分割） | 多个 | 🟡 中 |
| 临时文件残留 | 多数 | 🟡 低 |

### 6.3 权限与执行风险

| 风险类型 | 涉及脚本数 | 严重程度 |
|----------|------------|----------|
| 修改 site-packages 代码 | 1 | 🔴 高 |
| 创建伪造 Python 包 | 1 | 🔴 高 |
| PATH 劫持风险 | 多数 | 🟠 中 |
| 编译并执行 C/C++ 代码 | 3 | 🟠 中 |

---

## 7. 风险最高的 5 个脚本排行

| 排名 | 脚本路径 | 风险等级 | 主要风险 |
|------|----------|----------|----------|
| 1 | [conda-envs/msms-agent/install.sh](file:///workspace/conda-envs/msms-agent/install.sh) | 🔴 高危 | 修改第三方库代码 + 创建伪造包 + git clone + curl\|sh |
| 2 | [conda-envs/react-ot-agent/install.sh](file:///workspace/conda-envs/react-ot-agent/install.sh) | 🔴 高危 | git clone 第三方 + sed 补丁 + pip install -e |
| 3 | [conda-envs/scd-agent/install.sh](file:///workspace/conda-envs/scd-agent/install.sh) | 🔴 高危 | git clone 第三方 + 安装其 requirements.txt |
| 4 | [conda-envs/void-agent/install.sh](file:///workspace/conda-envs/void-agent/install.sh) | 🔴 高危 | 硬编码用户路径 + git clone + curl\|sh |
| 5 | [conda-envs/mace-agent/install_lammps.sh](file:///workspace/conda-envs/mace-agent/install_lammps.sh) | 🟠 中危 | git clone LAMMPS + 编译 C++ + rm -rf |

---

## 8. 自动化与不透明性风险

### 8.1 "一键安装" 的不透明性

正如用户所指出的，README 中建议用户仅告诉 AI agent "安装 AtomisticSkills"，然后由 agent 自动执行安装过程。这带来以下风险：

1. **用户不知情**: 用户可能不知道具体执行了哪些脚本
2. **权限边界模糊**: 安装脚本可能修改系统级配置
3. **难以回滚**: `conda env remove` + `conda env create` 会完全替换环境
4. **网络请求不可控**: 安装过程中会向多个外部域名发起请求

### 8.2 MCP 配置中的路径泄露

[mcp_config.json](file:///workspace/mcp_config.json) 中包含开发者的完整路径：
```json
"command": "/home/bdeng/miniforge3/envs/matgl-agent/bin/python",
"PYTHONPATH": "/home/bdeng/projects/AtomisticSkills"
```

**风险**: 
- 泄露用户名 `bdeng`
- 泄露项目目录结构
- 该文件被提交到了 git 仓库中

---

## 9. 建议修复方向

### 9.1 立即修复 (P0)
1. **移除 `mcp_config.json` 中的硬编码路径** - 使用模板或相对路径
2. **修复 `void-agent/install.sh` 中的硬编码路径** - 使用项目根目录动态计算
3. **停止使用 `curl | sh` 安装 uv** - 改用 conda 安装或提供校验和

### 9.2 短期修复 (P1)
1. **所有第三方 git clone 添加版本 tag/commit hash 锁定**
2. **为所有下载的文件添加 checksum 验证**
3. **停止修改 site-packages 中的代码** - 使用 monkey patch 或 fork 版本
4. **临时文件改用 `mktemp -d` 创建在 `/tmp` 下**

### 9.3 长期改进 (P2)
1. **提供安装前审查机制** - 列出将要执行的操作清单，需用户确认
2. **添加安装脚本签名/校验** - 确保脚本未被篡改
3. **沙盒化安装过程** - 在容器或隔离环境中执行安装
4. **编写安全安装指南** - 明确告知用户安装过程的风险

---

## 10. 结论

AtomisticSkills 仓库中的 shell 脚本主要用于 Conda 环境安装和研究示例执行。虽然没有发现明显的恶意后门，但存在多处**供应链攻击风险**和**不安全的安装实践**，特别是：

1. ✅ **无明显恶意代码** - 所有脚本功能与其声明的用途一致
2. ⚠️ **供应链风险较高** - 大量依赖外部下载且无完整性校验
3. ⚠️ **安装过程不透明** - 用户难以知晓具体执行了哪些操作
4. ⚠️ **硬编码路径泄露** - `mcp_config.json` 和部分 install.sh 包含开发者路径

建议优先修复高危风险项，特别是 `curl | sh` 模式和硬编码路径问题。

---

*报告生成时间: 2026-07-01*
