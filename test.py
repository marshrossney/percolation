"""To be replaced by a Jupyter notebook for the students..."""
import matplotlib.pyplot as plt
from time import time
from sys import argv

from model import PandemicModel
from lattice import SquareLattice

L = 100
tp = 0.1
vf = 0.1
ii = 1
dur = 21
inim = True
tr = 5
days = 100


lattice = SquareLattice(length=L)
model = PandemicModel(
    lattice,
    transmission_prob=tp,
    vaccine_frac=vf,
    initial_infections=ii,
    infection_duration=dur,
    infected_are_immune=inim,
    travel_rate=tr,
)

if "time" in argv:
    t1 = time()
    model.evolve(n_days=days)
    t2 = time()
    print(f"time: {t2 - t1} seconds")

elif "help" in argv:
    print(help(model))

elif "plot" in argv:
    model.evolve(n_days=days)
    model.plot_evolution()


else:
    ani = model.animate(n_days=days)
    plt.show()
