[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edwarddramirez/svi-dist-fit/HEAD) [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/license/mit) ![Python](https://img.shields.io/badge/python-3.11.4-blue.svg) ![Repo Size](https://img.shields.io/github/repo-size/edwarddramirez/svi-dist-fit) 

# svi-dist-fit
Summary notebooks showing how to fit a target distribution with a class of distributions using SVI (via NumPyro). Unlike standard SVI, our "data" is a distribution rather than a finite collection of samples. This is the first time fitting distributions with other distributions is performed using SVI-type optimization.

# Notebooks
1. `01_kl_simple.ipynb` - Fitting a parametric model with another parametric model
2. `02_kl_rate.ipynb` - Fitting a non-parametric model with a parametric model 
3. `03_kl_poisson.ipynb` - Fitting a non-parametric Poisson model with a parametric Poisson model 

# Installation
Run the `environment.yml` file by running the following command on the main repo directory:
```
conda env create
```
The installation works for `conda==4.12.0`. This will install all packages needed to run the code on a CPU with `jupyter`. 

If you want to run this code with a CUDA GPU, you will need to download the appropriate `jaxlib==0.4.13` version. For example, for my GPU running on `CUDA==12.3`, I would run:
```
pip install jaxlib==0.4.13+cuda12.cudnn89
```
The key to using this code directly would be to retain the `jax` and `jaxlib` versions. 
