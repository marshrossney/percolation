"""To be replaced by a Jupyter notebook for the students..."""
import matplotlib.pyplot as plt
from time import time
from sys import argv

from model import PandemicModel
from lattice import SquareLattice


lattice = SquareLattice(length=100)
model = PandemicModel(lattice, transmission_prob=0.3, vaccine_frac=0.35, initial_infections=10, infection_duration=21, infected_are_immune=True)

if "ani" in argv:
    model.animate(n_days=365)

elif "help" in argv:
    print(help(model))

elif "plot" in argv:
    model.evolve(n_days=365)
    model.plot_evolution(critical_threshold=0.1)

else:
    t1 = time()
    model.evolve(n_days=365)
    t2 = time()
    print(f"time: {t2 - t1} seconds")


