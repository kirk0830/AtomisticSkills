# API Key / Token / 邮箱 获取指南

> 本指南说明如何为 AtomisticSkills 中常用的外部服务申请 API key、token 或配置邮箱。
> 建议将获取到的 key 写入 `~/.atomistic_skills.yaml`，或导出为环境变量。

---

## 快速配置模板

将以下内容保存为 `~/.atomistic_skills.yaml`（注意替换为真实值）：

```yaml
# 外部数据库/API
MP_API_KEY: "your_mp_api_key"
ELSEVIER_API_KEY: "your_elsevier_api_key"
ELSEVIER_INST_TOKEN: "your_elsevier_inst_token"  # 可选
SPRINGER_API_KEY: "your_springer_api_key"
UNPAYWALL_EMAIL: "your_email@example.com"
OPENALEX_EMAIL: "your_email@example.com"        # 推荐
RXN_API_KEY: "your_ibm_rxn_api_key"
HF_TOKEN: "your_huggingface_token"
GOOGLE_API_KEY: "your_google_api_key"
OPENAI_API_KEY: "your_openai_api_key"

# DFT 二进制/数据路径（按实际安装路径填写）
ORCA_BINARY_PATH: "/path/to/orca"
VASP_CMD: "mpirun -np 16 vasp_std"
ATOMATE2_VASP_CMD: "mpirun -np 16 vasp_std"
PMG_VASP_PSP_DIR: "/path/to/potcars"

# Atomate2 远程项目（如使用远程模式）
ATOMATE2_REMOTE_PROJECT: "remote_perlmutter"

# HPC（按需配置）
HPC_MODE: "auto"          # local / ssh / auto
HPC_PROFILE: "generic"    # nersc_perlmutter / mit_supercloud / generic
HPC_SSH_HOST: "cluster.example.com"
HPC_SSH_USER: "your_username"
HPC_SSH_KEY: "/home/your_username/.ssh/id_ed25519"
HPC_SSH_PORT: 22
HPC_MODULES_VASP: "vasp/6.4.2-cpu,intel/2023"
HPC_MODULES_ORCA: "orca/5.0.4,openmpi/4.1.5"
```

这些 key 会通过 `src/utils/config_utils.py::inject_config_into_env()` 在 MCP 服务器启动时自动注入到环境变量中。

> **注意**：`UNPAYWALL_EMAIL` 是使用 Unpaywall polite pool 的**必需**项；`OPENALEX_EMAIL` 是**可选但推荐**的，未设置时 OpenAlex 会回退到默认邮箱，且 Unpaywall 无法使用 `OPENALEX_EMAIL` 作为备选。

---

## 1. Materials Project (`MP_API_KEY`)

**用途**：查询晶体结构、能带、态密度、弹性、热力学、相图、Pourbaix 等数据。

**申请步骤**：

1. 打开 https://next-gen.materialsproject.org/api
2. 使用邮箱注册或登录 Materials Project 账号。
3. 进入 API Keys 页面，点击 **Generate API Key**。
4. 复制 key 并设置环境变量：

   ```bash
   export MP_API_KEY="your_api_key_here"
   ```

   或写入 `~/.atomistic_skills.yaml`：

   ```yaml
   MP_API_KEY: "your_api_key_here"
   ```

---

## 2. Elsevier (`ELSEVIER_API_KEY`)

**用途**：通过 ScienceDirect API 下载 Elsevier 旗下期刊论文全文或元数据。

**申请步骤**：

1. 打开 https://dev.elsevier.com/
2. 注册 Elsevier Developer Portal 账号。
3. 创建一个新的 **API Key**，选择合适的使用场景（如学术/机构）。
4. （可选）如果所在机构有 ScienceDirect 订阅，可在同一门户申请 **Institution Token**，以解锁机构订阅内容。
5. 设置环境变量：

   ```bash
   export ELSEVIER_API_KEY="your_api_key"
   export ELSEVIER_INST_TOKEN="your_inst_token"  # 可选
   ```

---

## 3. Springer Nature (`SPRINGER_API_KEY`)

**用途**：通过 Springer Meta API 或 TDM API 获取 Springer Nature 论文元数据和全文。

**申请步骤**：

1. 打开 https://dev.springernature.com/
2. 注册开发者账号。
3. 创建应用并获取 **API Key**。
4. 设置环境变量：

   ```bash
   export SPRINGER_API_KEY="your_api_key"
   ```

---

## 4. Unpaywall (`UNPAYWALL_EMAIL`)

**用途**：查找论文的开放获取（Open Access）版本，并尝试下载 PDF。

**申请步骤**：

1. 打开 https://unpaywall.org/
2. Unpaywall 不需要注册账号，只需要一个邮箱即可进入 polite pool。
3. 设置环境变量：

   ```bash
   export UNPAYWALL_EMAIL="your_email@example.com"
   ```

   `OPENALEX_EMAIL` 可作为 `UNPAYWALL_EMAIL` 的备选。

---

## 5. OpenAlex (`OPENALEX_EMAIL`)

**用途**：调用 OpenAlex polite pool，获取更稳定、更快速的学术文献搜索结果。

**申请步骤**：

1. 打开 https://openalex.org/
2. OpenAlex 完全免费开放，无需注册 key。
3. 建议在请求中附带邮箱以使用 polite pool：

   ```bash
   export OPENALEX_EMAIL="your_email@example.com"
   ```

   如果未设置，代码会回退到默认邮箱 `support@openalex.org`，但强烈推荐使用自己的邮箱。

---

## 6. IBM RXN (`RXN_API_KEY`)

**用途**：调用 IBM RXN for Chemistry 进行逆合成（retrosynthesis）预测与反应产物预测。

**申请步骤**：

1. 打开 https://rxn.res.ibm.com/
2. 注册账号并登录。
3. 进入个人设置或 API 页面，创建 **API Key**。
4. 设置环境变量：

   ```bash
   export RXN_API_KEY="your_api_key"
   ```

---

## 7. HuggingFace (`HF_TOKEN`)

**用途**：下载 gated 模型或数据集，例如 `chem-nmr-analysis` skill 中的 ReactionT5 模型。

**申请步骤**：

1. 打开 https://huggingface.co/settings/tokens
2. 登录 HuggingFace 账号。
3. 点击 **New token**，选择 `read` 权限，生成 token。
4. 若访问 gated 模型，需先在模型页面点击 **Request access** 并同意用户协议。
5. 设置环境变量：

   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
   ```

---

## 8. Google Gemini (`GOOGLE_API_KEY`)

**用途**：在 `general-plot-digitizer` skill 中调用 Gemini 视觉语言模型提取图表元数据。

**申请步骤**：

1. 打开 https://ai.google.dev/
2. 使用 Google 账号登录 Google AI Studio。
3. 进入 API keys 页面，点击 **Create API Key**。
4. 复制 key 并设置环境变量：

   ```bash
   export GOOGLE_API_KEY="your_api_key"
   ```

   也可选择设置 `GEMINI_MODEL` 覆盖默认模型：

   ```bash
   export GEMINI_MODEL="gemini-2.5-flash-lite"
   ```

---

## 9. OpenAI (`OPENAI_API_KEY`)

**用途**：在 `general-plot-digitizer` skill 中调用 OpenAI GPT-4V / GPT-4o 等模型提取图表元数据。

**申请步骤**：

1. 打开 https://platform.openai.com/
2. 注册或登录 OpenAI 账号。
3. 进入 **API keys** 页面，点击 **Create new secret key**。
4. 复制 key 并设置环境变量：

   ```bash
   export OPENAI_API_KEY="sk-xxxxxxxx"
   ```

---

## 10. ORCA 二进制路径 (`ORCA_BINARY_PATH`)

**用途**：指向本地安装的 ORCA 量子化学程序可执行文件。

**获取步骤**：

1. 从 ORCA 官方渠道（如 https://www.faccts.de/orca 或 https://orcaforum.kofo.mpg.de）下载适合您系统的 ORCA 版本。
2. 解压到固定目录，例如 `/opt/orca/5.0.4/`。
3. 设置环境变量：

   ```bash
   export ORCA_BINARY_PATH="/opt/orca/5.0.4/orca"
   ```

   确保该文件具有可执行权限。

---

## 11. VASP / POTCAR 路径 (`VASP_CMD`、`PMG_VASP_PSP_DIR`)

**用途**：运行 VASP 计算并生成 POTCAR 赝势文件。

**获取步骤**：

1. VASP 为商业软件，需从 https://www.vasp.at/ 购买许可并下载。
2. 安装 VASP 后，设置运行命令：

   ```bash
   export VASP_CMD="mpirun -np 16 vasp_std"
   export ATOMATE2_VASP_CMD="mpirun -np 16 vasp_std"
   ```

3. 从 VASP 赝势包（POTCAR）中解压出 POTCAR 目录，例如 `/opt/vasp/potcars/potpaw_PBE/`。
4. 设置环境变量：

   ```bash
   export PMG_VASP_PSP_DIR="/opt/vasp/potcars/potpaw_PBE"
   ```

   也可以在 `~/.pmgrc.yaml` 中配置：

   ```yaml
   PMG_VASP_PSP_DIR: /opt/vasp/potcars/potpaw_PBE
   MP_API_KEY: your_mp_api_key
   ```

---

## 12. 验证配置是否生效

在 `base` 环境中运行：

```bash
pixi run -e base python - <<'PY'
import os
keys = [
    "MP_API_KEY", "ELSEVIER_API_KEY", "SPRINGER_API_KEY",
    "UNPAYWALL_EMAIL", "OPENALEX_EMAIL", "RXN_API_KEY",
    "HF_TOKEN", "GOOGLE_API_KEY", "OPENAI_API_KEY",
    "ORCA_BINARY_PATH", "VASP_CMD", "PMG_VASP_PSP_DIR",
]
for k in keys:
    print(f"{k}: {'已设置' if os.getenv(k) else '未设置'}")
PY
```

---

## 参考链接

- 完整环境变量说明：`docs/environment_variables.md`
- HPC 提交配置：`docs/hpc_job_submission.md`
- 项目维护计划：`.trae/documents/api_key_inventory_draft.md`
