A toy model of a pandemic that is in reality a percolation model with a lazily spun narrative revolving around cake.

Written as a 'computer experiment' for Physics 1B undergraduate 'labs' at UoE in Spring 2021.

# Installation

If you're running the Jupyter notebook on University of Edinburgh's Notable servers, ...

# Usage

## For students: Jupyter notebook
...

## Command line

```bash
p1b-pandemic -c input.yml
```

## Python

```python
from p1b_pandemic.lattice import SquareLattice
from p1b_pandemic.model import PandemicModel

lattice = SquareLattice(dimensions=100)
model = PandemicModel(lattice, transmission_prob=0.25, vaccine_frac=0.4)

model.evolve(n_days=200)
model.plot_evolution()
```

## Running the tests

Install `pytest` (included in conda environment).
In the repository, run
```bash
pytest
```
there you go...

# To do

* Graphs...

# Feedback

Email me..
