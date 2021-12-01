## conda is an environment and package management tool

conda --version

conda create -n python37 python=3.7  ## create an environment named with python34 with the python version 3.7

conda env list   ## list all created environments

conda activate python37   ## for windows
source activate python37 ## for mac or linux

python --version   ## version of python in this environment

conda install package_num   ## install a package

conda list     ## list all installed package

conda update package_num   ## upgard the installed package

conda remove package_num   ## remove the i

conda deactivate python37    ## for windows

source deactivate python37   ## for Mac or Linux

conda env export > environment.yaml  ## export the current environment

conda remove -n python37 --all  ## remove the created environment and all installed packages

conda env create -f environment.yaml  ## create an environment with the exported environment settings