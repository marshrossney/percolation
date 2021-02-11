import numpy as np
from timeit import timeit

from p1b_pandemic.lattice import SquareLattice
from p1b_pandemic.model import PandemicModel
from p1b_pandemic.config import parser
from p1b_pandemic.scripts.parameter_scan import parameter_scan


ARGS = parser.parse_args()


def load_model():
    """Returns loaded model."""
    lattice = SquareLattice(
        dimensions=ARGS.dimensions,
        n_connections=ARGS.n_connections,
        periodic=ARGS.periodic,
    )
    model = PandemicModel(
        lattice,
        transmission_prob=ARGS.transmission_prob,
        vaccine_frac=ARGS.vaccine_frac,
        recovery_time=ARGS.recovery_time,
        recovered_are_immune=ARGS.recovered_are_immune,
        travel_prob=ARGS.travel_prob,
        nucleus_size=ARGS.nucleus_size,
    )
    model.init_state(reproducible=ARGS.reproducible)

    return model


MODEL = load_model()


def anim():

    MODEL.animate(n_days=ARGS.n_days, interval=ARGS.interval, outpath=ARGS.outpath)
    MODEL.plot_evolution(outpath=ARGS.outpath)


def time():

    t_excl = timeit(
        stmt="MODEL.evolve(n_days=ARGS.n_days)",
        setup="MODEL.init_state()",
        number=ARGS.repeats,
        globals=globals(),
    )
    t_incl = timeit(
        stmt="MODEL.init_state; MODEL.evolve(n_days=ARGS.n_days)",
        number=ARGS.repeats,
        globals=globals(),
    )

    print(
        f"""
    Number of nodes:        {MODEL.lattice.n_nodes}
    Simulation length:      {ARGS.n_days} steps
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
