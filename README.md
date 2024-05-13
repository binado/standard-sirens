# sirenslib

This repository hosts utility methods for projects working with gravitational-wave standard sirens.

## Installation

### From source

Clone the repository to your local machine with

```bash
git clone https://github.com/binado/standard-sirens.git
```

Run

```bash
pip install .
```

You may also use flags for installing optional dependencies. Use

```bash
pip install .[gwdali]
```

to install [`gwdali`](https://gwdali.readthedocs.io/en/latest/), which is used as a benchmark in the [`fast_injections` notebook](/notebooks/fast_injections.ipynb).

The pseudo-$C_{\ell}$ computation is performed with the `naMaster` [package](https://github.com/LSSTDESC/NaMaster). You may install it with the command

```bash
pip install numpy && pip install .[pymaster]
```

See [this discussion](https://github.com/LSSTDESC/NaMaster/pull/143) for why manually installing `numpy` first is required. If this approach throws errors, a simpler alternative may be to install it via the `conda-forge` [recipe](https://anaconda.org/conda-forge/namaster) (_not sure if it is up-to-date_).

```bash
conda create -n [myenv] python=3.10 && conda activate [myenv]
python -m pip install .
conda install -c conda-forge namaster
```

## Notebooks

There are several Jupyter notebooks available in the `notebooks` directory highlighting use cases for the code.

## Running scripts

Scripts are stored in the `sirenslib/scripts` directory, and are automatically installed. Click [here](sirenslib/scripts/README.md) for a complete list and usage instractions.
