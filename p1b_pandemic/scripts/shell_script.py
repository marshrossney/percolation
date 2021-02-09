from p1b_pandemic.lattice import SquareLattice
from p1b_pandemic.model import PandemicModel
from p1b_pandemic.config import parser
from time import time


def main():

    args = parser.parse_args()

    lattice = SquareLattice(
        dimensions=args.dimensions, n_connections=args.n_connections, periodic=args.periodic
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

    model.init_state(reproducible=args.reproducible)

    if args.skip_animation or args.time:
        t0 = time()
        model.evolve(n_days=args.n_days)
        t1 = time()

        if args.time:
            print(
                f"{(t1 - t0):.2g} seconds taken to perform {args.n_days} updates of a model with {lattice.n_nodes} nodes"
            )

    else:
        model.animate(n_days=args.n_days, interval=args.interval, outpath=args.outpath)
    model.plot_evolution(outpath=args.outpath)


if __name__ == "__main__":
    main()
