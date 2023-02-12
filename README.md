# IntraCerebral Hemorrhage (ICH) Expansion
Code for reproduciblity and open science for ICH prediction (expansion, mRS, mortality).

## 1. Setting up the environment :deciduous_tree:
Create and activate a conda environment with Python
 ```
conda create -n ich_prediction python=3.9
conda activate ich_prediction
 ```
Install the requirements
 ```
conda install tensorflow-gpu==1.15
conda install keras==2.3.1
conda install pandas
conda install matplotlib==3.2.2
conda install scikit-learn=0.24.1
conda install scipy
