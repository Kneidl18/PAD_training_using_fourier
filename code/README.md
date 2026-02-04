**Installation:**

in folder code/: 
- if you don't already have a conda environment: `conda env create -f environment.yml`
- activate the environment: `conda activate bildverarbeitung`
- update the environment (if you already have an environment or just want to update): `conda env 
update -f environment.yml --prune`
- pip editable: `pip install -e .`

**Structure:**

in src/: the core code of the project -> this is the module defined in pyproject.toml

in experiments/: all the experiments, like testing different band sizes, 
creating plots, etc. -> here the core module is imported

**Other infos**

You need a .env file with the name of your data folder (e.g.:
`REAL_DATA_ROOT="Data_FV_Spoofing_WS2025_26"`).<br>
It is suggested to modify the .env file that exists to point to the data location of the image folders.

**Running the code:**

`core_main` -> this is the basic main function (no arguments)