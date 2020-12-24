"""To be replaced by a Jupyter notebook for the students..."""
import matplotlib.pyplot as plt
from time import time
from sys import argv

from model import PandemicModel
from lattice import SquareLattice


lattice = SquareLattice(length=100)
model = PandemicModel(lattice, transmission_prob=0.25, vaccine_frac=0.1, initial_infections=10)

if "ani" in argv:
    model.animate(n_days=150)

elif "help" in argv:
    print(help(model))

elif "plot" in argv:
    model.evolve(n_days=150)
    model.plot_evolution()

else:
    t1 = time()
    model.evolve(n_days=150)
    t2 = time()
    print(f"time: {t2 - t1} seconds")


