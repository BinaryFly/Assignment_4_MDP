# Assignment_4_MDP

## instructions to initialize:

### If you want to bind the externally installed packages to the project and not install them on your entire system:

1. from the root directory cd into the assignment directory: `cd assignment_MDP`
2. initialize a new virtual environment to install packages in: `python -m venv ./venv` or `python3 -m venv ./venv` (depends on if you have an alias set for python3 or not)
3. activate the virtual environment: `source ./venv/bin/activate` (only works on unix based systems, linux or mac for example)
4. install the needed packages: `pip install numpy` or `pip3 install numpy` (depending on if you have an alias set for your command)

### If you don't care that you install external packages on your entire system and not locally

1. install the needed external packages globally on your system:
  - first try `pip install numpy` or `pip3 install numpy`
  - if that doesn't work try `sudo -H pip install numpy` or `sudo -H pip3 install numpy` (for unix based systems, so linux or mac)

