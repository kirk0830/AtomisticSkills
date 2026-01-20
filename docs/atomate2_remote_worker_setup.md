# NERSC Perlmutter: Atomate2 & JobFlow Remote Setup

This document summarizes the configuration and operational steps for running Atomate2 workflows remotely on NERSC Perlmutter using `jobflow-remote`.

## 1. Project Configuration
The primary configuration for this setup is located in:
`~/.jfremote/remote_perlmutter.yaml`

### Key Settings:
- **Worker Type**: `local` (Switched from `remote` to avoid SSH loopback issues on the login nodes).
- **Scheduler**: `slurm`
- **Working Directory**: `/global/cfs/cdirs/m5068/bdeng/jobflow_runs`
- **Environment Setup**: Handled via the `pre_run` script in the YAML, which:
  - Loads `vasp/6.4.2-cpu`.
  - Activates the `atomate2` conda environment.
  - Sets `export ATOMATE2_CONFIG_FILE=~/.config/atomate2/config.yaml`.

### Project Name Auto-detection
To avoid specifying `project_name` in every command, you can set the `ATOMATE2_REMOTE_PROJECT` environment variable:
```bash
export ATOMATE2_REMOTE_PROJECT=remote_perlmutter
```
When this variable is set, Atomate2 MCP tools will automatically use this project if no valid default is found in the `jobflow-remote` configuration.

## 2. Slurm Submission Policies
To comply with NERSC Perlmutter CPU node policies, the following Slurm resources are configured:
- **QoS**: `regular` (Automatically mapped to `regular_1` or `regular_0`).
- **Constraint**: `cpu` (Mandatory for Milan CPU jobs).
- **Time**: "24:00:00" (Default).
- **Note**: Explicit `partition` naming is omitted to allow the system to map based on QoS.

## 3. Database & Stores
The setup connects to an external MongoDB instance (`mongodb05.nersc.gov:27017`).

- **Queue Store**: Stores job states and metadata (`agent_db.queue`).
- **Job Store (Outputs)**: Stores calculation results (`agent_db.outputs`).
- **Additional Stores**:
  - `data`: Required for large datasets/specific Atomate2 outputs (`agent_db.data`).

## 4. Runner Operations
The JobFlow Remote Runner operates as a background daemon managed by `supervisord`.

### Common Commands:
- **Start Runner**:
  ```bash
  export PATH=/global/homes/b/bdeng/anaconda3/envs/atomate2/bin:$PATH
  jf -p remote_perlmutter runner start --log-level info
  ```
- **Stop Runner**:
  ```bash
  jf -p remote_perlmutter runner stop
  ```
- **Check Status**:
  ```bash
  jf -p remote_perlmutter runner status
  ```
- **Reset Daemon**: (Use if the daemon is stuck or reports running on another machine)
  ```bash
  jf -p remote_perlmutter runner reset
  ```

## 5. Remote Workflow: How to use it
Once this setup is complete, you can submit jobs from your **local machine** and they will be executed on **Perlmutter** without you needing to be logged in.

### Step-by-Step Execution:
1. **Submit locally**: Run your Atomate2 flow on your local machine using the `remote_perlmutter` project target. This uploads the jobs to the MongoDB queue.
2. **Daemon takes over**: The `jobflow-remote` runner daemon on Perlmutter periodically checks the MongoDB queue for `READY` jobs.
3. **Execution**: The daemon submits the jobs to Slurm, monitors them, and downloads results back to the `outputs` and `data` collections in MongoDB upon completion.

### Common Questions:
- **Do I need to start the runner every time?** No. Once started with `runner start`, it runs as a background process using `supervisord`. It will survive your logout.
- **Will jobs be submitted if I log out?** Yes. As long as the Perlmutter login node is up and the daemon hasn't been stopped, it will continue to process any new jobs you submit from your local machine.
- **How do I know if it's still alive?** Log in to Perlmutter and run `jf -p remote_perlmutter runner status`. If it says `Daemon status: running`, you are good to go.
- **What if Perlmutter reboots?** If the login node is rebooted, the daemon will stop. You will need to log in and run the `runner start` command again.

## 6. Job Management
- **List Jobs**: `jf -p remote_perlmutter job list`
- **Check Slurm Queue**: `squeue -u bdeng`
- **Runner Logs**: `~/.jfremote/remote_perlmutter/log/runner.log`

---
*Setup completed/verified on 2026-01-15.*
