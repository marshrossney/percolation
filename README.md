A toy model of a pandemic that is in reality a percolation model with a lazily spun narrative revolving around cake.

Written in Winter 20201 as a 'computer experiment' for Physics 1B undergraduate 'labs' at the University of Edinburgh.

# Installation

## Installation via Noteable (for UoE students)

Launch Noteable from the Learn page and choose

> Standard Notebook (Python 3)

from the dropdown menu.
Click 'Start'.

In the top right corner, click '+GitRepo' and paste *https://github.com/marshrossney/p1b-pandemic* (i.e. this url) into the box labelled 'Git Repository URL'.
The 'Branch' box should say *master*.
Click 'Clone'.

Navigate to `/p1b-pandemic/laboratory/` and click on an `.ipynb` file to start the notebook.


## Local installation

Clone the repo using one of the numerous options available!
For example,
```bash
git clone https://github.com/marshrossney/p1b-pandemic.git
cd p1b-pandemic
```

This project has rather minimal dependencies, and should run fine with reasonably up-to-date versions of NumPy, SciPy and Matplotlib.

To use the Jupyter notebooks, you also need...Jupyter.
Alternatively, you can run everything from the command line, which will require the [ConfigArgParse](https://github.com/bw2/ConfigArgParse) tool.

The easiest option is to simply create a Conda environment using the environment file provided,
```bash
conda create -n p1b -f environment.yml
conda activate p1b
```
and then install the package there.
```bash
python -m pip install -e .
```

# Usage


## Laboratory experiment (UoE students)

Students should work through the Jupyter notebooks in the `laboratory/` directory, referring to the lab manual for guidance (the one in this repository will not be the official one).


## In your own scripts

Basic usage will look like the following.

```python
from p1b_pandemic.lattice import SquareLattice
from p1b_pandemic.model import PandemicModel
from p1b_pandemic.scripts.parameter_scan import parameter_scan

# Create a 100x100 lattice
lattice = SquareLattice(dimensions=100)

# Instantiate a model
model = PandemicModel(lattice, transmission_prob=0.25, vaccine_frac=0.4)

Have a look at how it evolves with these parameters
model.animate(n_days=200)

# Plot an estimate for the probability that the model 'percolates' over a
# range of values of 'vaccine_frac'
parameter_scan(model, parameter="vaccine_frac", values=numpy.linspace(0, 0.8, 40))
```

Look at the options available by running e.g.
```python
help(SquareLattice)
help(PandemicModel)
help(parameter_scan)
```

## Command line

Installing the package will install a few scripts that can be run from the command line.
At the moment these are:
* `p1b-anim` which saves an animation as a gif
* `p1b-scan` which runs a 'parameter scan' (ideally over the percolation transition) and produces a nice plot.
* `p1b-time` which just runs `timeit` on a couple of things.

```bash
p1b-pandemic -c input.yml
```

## Running the tests

Install `pytest` (included in conda environment).

In the repository, run
```bash
pytest
```
There you go...

# To do

Things I'm planning to do...

* More types of lattices and graphs to use in place of the square lattice.
* More tools to analyse percolation transition.
* More options that make it more intuitive to relate the model to a pandemic.

# Feedback

Open a Github issue, pull request or just email me < Joe.Marsh-Rossney 'at' ed.ac.uk >.
