from p1b_pandemic.lattice import SquareLattice
from p1b_pandemic.model import PandemicModel
from p1b_pandemic.config import args


def main():

    lattice = SquareLattice(length=args.length, periodic=args.periodic)

    model = PandemicModel(
        lattice,
        transmission_prob=args.transmission_prob,
        vaccine_frac=args.vaccine_frac,
        recovery_time=args.recovery_time,
        recovered_are_immune=args.recovered_are_immune,
        outpath=args.outpath,
    )

    model.seed_rng(args.seed)

    model.init_state(initial_shape=args.initial_shape, nucleus_size=args.nucleus_size)

    if args.skip_animation:
        model.evolve(n_days=args.n_days)
    else:
        model.animate(n_days=args.n_days, save=True)
    model.plot_evolution(save=True)


if __name__ == "__main__":
    main()
