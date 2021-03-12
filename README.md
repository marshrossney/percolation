# Simple percolation model

A simple percolation model intended as a 'computer experiment' for Physics 1B undergraduate labs at the University of Edinburgh, used during the Spring semester of 2021 when courses had to be delivered remotely.

## WARNING for UoE students taking P1B labs:

**If you are on Notable and see this, it means you have cloned the wrong branch!**

Delete the whole folder and go through the steps to clone the repository again, but this time **replace master with p1b** in the 'branch' box. 

## Installation

### Local installation

Clone the repo using one of the numerous options available!
For example,
```bash
git clone https://github.com/marshrossney/percolation.git
cd percolation
```

This project has rather minimal dependencies, and should run fine with reasonably up-to-date versions of NumPy, SciPy and Matplotlib.

To use the Jupyter notebooks, you also need...Jupyter.
Alternatively, you can run everything from the command line, which will require the [ConfigArgParse](https://github.com/bw2/ConfigArgParse) tool.

The easiest option is to simply create a Conda environment using the environment file provided,
```bash
conda create -n perc -f environment.yml
conda activate perc
```
and then install the package there.
```bash
python -m pip install -e .
```

### Installation via Noteable

While it is easier to just use the 'p1b' branch, it is possible to install this as a package on Notable.

1. Open Notable and launch a Standard Notebook (Python 3)
2. Clone the repository using the '+GitRepo' tool
3. Open a terminal instance and run
```bash
cd percolation
python -m pip install -e .
```
4. You should now be able to run the Jupyter notebooks in Notable


## Usage

### In your own scripts

Basic usage will look like the following.

```python
from percolation.lattice import SquareLattice
from percolation.model import PercolationModel
from percolation.scripts.parameter_scan import parameter_scan

# Create a 100x100 lattice
lattice = SquareLattice(n_rows=100, n_cols=100)

# Instantiate a model
model = PercolationModel(lattice, inert_prob=0.4)

Have a look at how it evolves with these parameters
model.animate(n_steps=200)

# Plot an estimate for the probability that the model 'percolates' over a
# range of values of 'frozen_prob'
parameter_scan(model, start=0.1, stop=0.7, num=25, repeats=25)
```

Look at the options available by running e.g.
```python
help(parameter_scan)
```

### Command line

Installing the package will install a few scripts that can be run from the command line.
At the moment these are:
* `perc-anim` which saves an animation as a gif
* `perc-scan` which runs a 'parameter scan' (ideally over the percolation transition) and produces a nice plot.
* `perc-time` which just runs `timeit` on a couple of things and is mostly just useful to me.

Run e.g.
```bash
perc-anim --help
```
to see what arguments are required, or possible, to run the script.

You can supply arguments via the command line, but it's probably easier to store them in a configuration file, labelled `input.yml` below.
```bash
perc-scan -f input.yml
```
See the examples at `percolation/examples/` for some basic input files.

### Running the tests

Install `pytest` (included in conda environment).

In the root of the repository, run
```bash
pytest
```
There you go...

## To do

Things I'm planning to do if I get some time...

* More types of lattices and networks to use in place of the square lattice.
* More tools to analyse percolation transition.
* More options that make it more intuitive to relate the model to a pandemic.
* Jupyter notebooks for teaching and demonstrating.

## Queries and feedback

Open a Github issue, pull request or just email me < Joe.Marsh-Rossney 'at' ed.ac.uk >.

