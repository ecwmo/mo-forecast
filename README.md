# MO Automated WRF system

## Installation

---

### Installation Requirements

- bash environment
- slurm
- preinstalled WRF system
- python 3.9+

### 1. Clone the repo

```bash
git clone https://github.com/ecwmo/mo-forecast.git ~/forecast
```

### 2. Install conda environment

```bash
cd ~/forecast
mamba env create -f ./venv.yaml -p ./venv
```

### 3. Create a configuration file

```bash
cd ~/forecast
cp set_cron_env.sh.sample set_cron_env.sh
vi set_cron_env.sh
```

### 4. Setup required paths/files

- **input**/ - input files needed by the WRF system (eg. GFS, etc.)
- **output**/ - location of files created by post processing scripts
- **model**/ - WRF system executables and related files
  - **WPS**/ - should contain _geogrid.exe_, _ungrib.exe_, _metgrid.exe_ and related files like _namelist.wps_, _geog/_ and _VTable_. Can be a symbolic link to _\<WPS SRC DIR\>/_.
  - **WRF**/ - should contain _real.exe_, _wrf.exe_ and related files like _namelist.wrf_. Can be a symbolic link to _\<WRF SRC DIR\>/test/em_real_.

### 5. Create a cron entry

```bash
# 5-day forecast ensemble
# 00Z and 12Z
40 2,14 * * * . $HOME/forecast/set_cron_env.sh; . $HOME/forecast/scripts/run_fcst.sh
```

## Update

---

### 1. Pull from the repo

```bash
git pull
```

### 2. Update conda environment

```bash
cd ~/forecast
mamba env update -p ./venv -f ./venv.yaml --prune
```

## Contributing

---

### Prerequisites

- [pre-commit](https://pre-commit.com/)
- [tox < 4](https://tox.wiki/en/3.9.0/)

### 1. Fork the Repository

Fork the repository to your GitHub account.

### 2. Create a Feature Branch

Create a new feature branch (e.g. `git checkout -b feature-branch`).

### 3. Make Your Changes

Make the necessary changes to the codebase.

### 4. Run Pre-commit Hooks

Run pre-commit checks on all files:

```bash
pre-commit run --all-files
```

### 5. Run Tox

Execute tests defined in the `tox.ini`` file:

```bash
CONDA_EXE=mamba tox
```

### 6. Commit Your Changes

Commit your changes (e.g. `git commit -am 'Add new feature'`).

### 7. Push to Your Branch

Push your branch (e.g. `git push origin feature-branch`).

### 8. Open a Pull Request

Open a Pull Request from your feature branch to the main repository.

We appreciate your contributions and will review them as soon as possible.
