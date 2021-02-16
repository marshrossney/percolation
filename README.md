# P1B percolation 'laboratory'

A simple percolation model intended as a 'computer experiment' for Physics 1B undergraduate labs at the University of Edinburgh, used during the Spring semester of 2021 when courses had to be delivered remotely.

## Installation

### Installation via Noteable (for UoE students)

Launch Noteable from the Learn page and choose

> Standard Notebook (Python 3)

from the dropdown menu.
Click 'Start'.

In the top right corner, click '+GitRepo' and paste *https://github.com/marshrossney/p1b-percolation* (i.e. this url) into the box labelled 'Git Repository URL'.
The 'Branch' box should say *master*.
Click 'Clone'.

Again in the top right corner, click 'New' and then select 'Terminal' in the drop-down menu.
A window should appear which looks rather intimidating.
Don't worry!
We just need to run two commands --- you can copy and paste them from below.
```bash
cd p1b-percolation
python -m pip install -e .
```
The second line should cause the terminal to chat to you a little bit about how it's finding the installation process, and eventually it should say `Successfully installed p1b-percolation`.

You can close this window now, and close the terminal session by clicking the 'Running' tab (next to 'Files') and pressing the 'Shutdown' button.

Back in the files tab, navigate to `/p1b-percolation/laboratory/` and click on an `.ipynb` file to start the notebook.

#### I'm having problems with the terminal installation

I'm going to make some changes very shortly which mean you don't need to install anything via the terminal.

### Local installation

Clone the repo using one of the numerous options available!
For example,
```bash
git clone https://github.com/marshrossney/p1b-percolation.git
cd p1b-percolation
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

## Usage


### Laboratory experiment (UoE students)

Students should work through the Jupyter notebooks in the `laboratory/` directory, referring to the lab manual for guidance (the one in this repository will not be the official one).


### In your own scripts

Basic usage will look like the following.

```python
from p1b_percolation.lattice import SquareLattice
from p1b_percolation.model import PercolationModel
from p1b_percolation.scripts.parameter_scan import parameter_scan

# Create a 100x100 lattice
lattice = SquareLattice(n_rows=100, n_cols=100)

# Instantiate a model
model = PercolationModel(lattice, frozen_prob=0.4)

Have a look at how it evolves with these parameters
model.animate(n_steps=200)

# Plot an estimate for the probability that the model 'percolates' over a
# range of values of 'frozen_prob'
parameter_scan(model, parameter="frozen_prob", values=numpy.linspace(0, 0.8, 40))
```

Look at the options available by running e.g.
```python
help(SquareLattice)
help(PercolationModel)
help(parameter_scan)
```

### Command line

Installing the package will install a few scripts that can be run from the command line.
At the moment these are:
* `p1b-anim` which saves an animation as a gif
* `p1b-scan` which runs a 'parameter scan' (ideally over the percolation transition) and produces a nice plot.
* `p1b-time` which just runs `timeit` on a couple of things and is mostly just useful to me.

Run e.g.
```bash
p1b-anim --help
```
to see what arguments are required, or possible, to run the script.

You can supply arguments via the command line, but it's probably easier to store them in a configuration file, labelled `input.yml` below.
```bash
p1b-scan -f input.yml
```
See the examples at `p1b-percolation/examples/` for some basic input files.

### Running the tests

Install `pytest` (included in conda environment).

In the repository, run
```bash
pytest
```
There you go...

## To do

Things I'm planning to do...

* More types of lattices and networks to use in place of the square lattice.
* More tools to analyse percolation transition.
* More options that make it more intuitive to relate the model to a pandemic.

## Queries and feedback

Open a Github issue, pull request or just email me < Joe.Marsh-Rossney 'at' ed.ac.uk >.
