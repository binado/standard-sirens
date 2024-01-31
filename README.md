## Standard sirens

This repository hosts the manuscript _Cosmology with standard sirens_, written as a project for the Cosmology I course at PPGCOSMO, UFES. The source files for the manuscript are available in the `tex` directory, and the code developed for the data analysis is available in the `src` directory.

You will also find Jupyter notebooks which complement the discussion in the main text:

- [Visualizing the galaxy catalog](./src/visualize_catalog.ipynb)
- [Inferring cosmological parameters with standard sirens and galaxy catalogs](./src/inference_with_galaxy_catalogs.ipynb)
- [Inferring cosmological and BBH merger rate parameters with standard sirens](./src/visualize_mcmc.ipynb)
- [Inference bias analysis with p-p plots](./src/pp_analysis.ipynb)

## Installation

With conda:

```bash
conda create --name <env> -f requirements.txt
```

## Running scripts

To run scripts from the `scripts` folder, use the following command at the main directory:

```bash
python -m src.scripts.foo <args>
```

You may use the flag `-h` for instructions on the commands and their accepted arguments.
