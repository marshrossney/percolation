import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from p1b_pandemic.lattice import SquareLattice
from p1b_pandemic.model import PandemicModel
from p1b_pandemic.config import parser


def parameter_scan(
    model, parameter="vaccine_frac", values=np.linspace(0, 0.8, 20), repeats=20
):
    """Loops over a range of values for a given parameter of the model, evolving the
    model forwards until it has either percolated or transmission has stopped.
    This is repeated a number of times for each value of the parameter.

    Inputs
    ------
    model: PandemicModel
        The model object.
    parameter: str
        The parameter to evolve. Must be an attribute of model.
    values: iterable
        An interable containing the values of the parameter to be looped over.
    repeats: int
        The number of simulations to run for each value of the parameter.
    """

    percolation_fraction = []

    pbar = tqdm(total=(len(values) * repeats), desc=f"Simulations completed")

    for value in values:

        setattr(model, parameter, value)

        percolated = []

        for rep in range(repeats):

            model.init_state(reproducible=False)

            model.evolve_until_percolated()

            percolated.append(model.has_percolated)

            pbar.update()

        frac = sum(percolated) / len(percolated)
        percolation_fraction.append(frac)

    percolation_fraction = np.array(percolation_fraction)
    gauss_errors = np.sqrt(
        (percolation_fraction * (1 - percolation_fraction)) / repeats
    )

    # TODO: improve this. Maybe this is where students plot by hand...
    plt.errorbar(
        x=values,
        y=percolation_fraction,
        yerr=gauss_errors,
        fmt="o",
        color="green",
        capsize=2,
        elinewidth=1.5,
        ecolor="black",
        capthick=1.5,
        markeredgecolor="black",
    )
    plt.xlabel(parameter)
    plt.ylabel("fraction of simulations that percolate")
    plt.show()


def main():

    args = parser.parse_args()

    lattice = SquareLattice(
        dimensions=args.dimensions,
        n_connections=args.n_connections,
        periodic=args.periodic,
    )

    model = PandemicModel(
        lattice,
        transmission_prob=args.transmission_prob,
        vaccine_frac=args.vaccine_frac,
        recovery_time=args.recovery_time,
        recovered_are_immune=args.recovered_are_immune,
        travel_prob=args.travel_prob,
        nucleus_size=args.nucleus_size,
    )

    parameter_scan(
        model,
        parameter=args.parameter,
        values=np.linspace(args.start, args.stop, args.num),
        repeats=args.repeats,
    )


if __name__ == "__main__":

    main()
