import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from pathlib import Path

# NOTE: the following would be better but results in ExperimentalFeatureWarning
# from tqdm.autonotebook import tqdm

from p1b_pandemic.lattice import SquareLattice
from p1b_pandemic.model import PandemicModel
from p1b_pandemic.config import parser


def logistic(x, loc, steepness):
    """(One minus) the logistic function."""
    return 1 / (1 + np.exp(steepness * (x - loc)))


def parameter_scan(
    model,
    parameter="vaccine_frac",
    values=np.linspace(0, 0.15, 25),
    repeats=25,
    notebook_friendly=True,
    outpath=None,
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
    notebook_friendly: bool
        Use tqdm bar specifically tailored for Jupyter notebooks.
    outpath: str
        Path to directory in which to save plot.
    """

    if notebook_friendly:
        pbar = tqdm_notebook(
            total=(len(values) * repeats), desc=f"Simulations completed"
        )
    else:
        pbar = tqdm(total=(len(values) * repeats), desc=f"Simulations completed")

    percolation_fraction = []

    for value in values:
        setattr(model, parameter, value)
        percolation_fraction.append(model.estimate_percolation_prob(repeats))
        pbar.update(repeats)

    percolation_fraction = np.array(percolation_fraction)

    # Errors are the largest of Poisson and statistical
    std_errors = np.sqrt((percolation_fraction * (1 - percolation_fraction)) / repeats)
    poisson_errors = 1.0 / np.sqrt(repeats)
    errors = np.fmax(std_errors, poisson_errors)

    # Fit logistic curve to data
    popt, pcov = optim.curve_fit(
        logistic,
        xdata=values,
        ydata=percolation_fraction,
        sigma=errors,
        p0=(0.5, 10),
        bounds=((0, 0), (1, np.inf)),
    )

    loc, steepness = popt
    e_loc, e_steepness = np.sqrt(pcov.diagonal())
    print(f"Transition occurs at at {loc} +/- {e_loc}")
    print(f"Steepness parameter is {steepness} +/- {e_steepness}")

    # Modify error bars for plot so that they don't fall outside [0, 1]
    errors_above = errors.copy()
    errors_below = errors.copy()
    upper_cap = percolation_fraction + errors
    lower_cap = percolation_fraction - errors
    cap_above_one = upper_cap > 1
    cap_below_zero = lower_cap < 0
    errors_above[cap_above_one] -= upper_cap[cap_above_one] - 1
    errors_below[cap_below_zero] += lower_cap[cap_below_zero]
    errors_asymm = np.stack((errors_below, errors_above), axis=0)

    fig, ax = plt.subplots()
    ax.set_title("Percolation")
    ax.set_xlabel(parameter.replace("_", " "))
    ax.set_ylabel("Percolation fraction")

    ax.errorbar(
        x=values,
        y=percolation_fraction,
        yerr=errors_asymm,
        fmt="o",
        color="green",
        capsize=2,
        elinewidth=1.5,
        ecolor="black",
        capthick=1.5,
        markeredgecolor="black",
        label="data",
        zorder=0,
    )

    fit_x = np.linspace(values.min(), values.max(), 1000)
    fit_values = logistic(fit_x, loc=loc, steepness=steepness)
    ax.plot(
        fit_x,
        fit_values,
        color="r",
        linestyle="--",
        linewidth="2.0",
        label="least squares fit",
        zorder=1,
    )

    ax.legend()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exists_ok=True)
        fig.savefig(outpath / "percolation.png")
    else:
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
        notebook_friendly=False,
    )


if __name__ == "__main__":

    main()
