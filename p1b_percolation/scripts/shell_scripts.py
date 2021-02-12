import numpy as np
from timeit import timeit

from p1b_percolation.lattice import SquareLattice
from p1b_percolation.model import PercolationModel
from p1b_percolation.config import parser
from p1b_percolation.scripts.parameter_scan import parameter_scan


ARGS = parser.parse_args()


def load_model():
    """Returns loaded model."""
    lattice = SquareLattice(
        dimensions=ARGS.dimensions,
        n_links=ARGS.n_links,
        periodic=ARGS.periodic,
    )
    model = PercolationModel(
        lattice,
        frozen_prob=ARGS.frozen_prob,
        transmission_prob=ARGS.transmission_prob,
        recovery_time=ARGS.recovery_time,
        recovered_are_frozen=ARGS.recovered_are_frozen,
        shuffle_prob=ARGS.shuffle_prob,
        nucleus_size=ARGS.nucleus_size,
    )
    model.init_state(reproducible=ARGS.reproducible)

    return model


MODEL = load_model()


def anim():

    MODEL.animate(n_steps=ARGS.n_steps, interval=ARGS.interval, outpath=ARGS.outpath)
    MODEL.plot_sir(outpath=ARGS.outpath)


def time():

    t_excl = timeit(
        stmt="MODEL.evolve(n_steps=ARGS.n_steps)",
        setup="MODEL.init_state()",
        number=ARGS.repeats,
        globals=globals(),
    )
    t_incl = timeit(
        stmt="MODEL.init_state; MODEL.evolve(n_steps=ARGS.n_steps)",
        number=ARGS.repeats,
        globals=globals(),
    )

    print(
        f"""
    Number of nodes:        {MODEL.network.n_nodes}
    Simulation length:      {ARGS.n_steps} steps
    Number of simulations:  {ARGS.repeats}
    Timings:
        Excluding initialisation:   {t_excl:.4g} seconds    
        Including initialisation:   {t_incl:.4g} seconds
    """
    )


def scan():

    parameter_scan(
        MODEL,
        parameter=ARGS.parameter,
        values=np.linspace(ARGS.start, ARGS.stop, ARGS.num),  # TODO improve
        repeats=ARGS.repeats,
        notebook_friendly=False,
        outpath=ARGS.outpath,
    )
