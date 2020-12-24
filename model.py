import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from lattice import SquareLattice

plt.style.use("utils/p1b.mplstyle")


class PandemicModel:
    """Class containing a simple model of a pandemic.

    Inputs
    ------
    lattice: lattice.SquareLattice
        The underlying lattice upon which we perform the simulation.
    transmission_prob: float
        The probability of an infected node passing the virus onto a neighbouring
        uninfected (and non-vaccinated) node.
    vaccine_frac: float
        The fraction of nodes that are flagged as 'vaccinated' against the virus.
    initial_infections: (int, float)
        The initial number or fraction of infected nodes.

    Notes
    -----
        For now, only a square lattice is implemented, but any data structure
        that includes a set of nodes and edges (a graph) would be valid in
        principle. This may require further modifications to this class if the
        number of vertices attached to each node varies between nodes.


    """

    def __init__(
        self,
        lattice,
        transmission_prob=0.25,
        vaccine_frac=0.0,
        initial_infections=1,
    ):
        # For now, just check that the lattice is a SquareLattice
        if type(lattice) != SquareLattice:
            raise ValueError("Please provide an instance of SquareLattice.")
        self._lattice = lattice

        # Set parameters which users can modify
        self.transmission_prob = transmission_prob
        self.vaccine_frac = vaccine_frac
        self.initial_infections = initial_infections

        # Initialise the rng (uses known default seed)
        self.seed_rng()

        # Initalise the model with a single infected site
        self.init_state()

    # ----------------------------------------------------------------------------------------
    #                                                                     | Data descriptors |
    #                                                                     --------------------

    @property
    def transmission_prob(self):
        """The probability of transmitting the virus from an infected node to a neighbouring
        non-infected (and non-vaccinated) node. Can also be seen as a coupling strength
        for the (one-way) interaction between nodes on the lattice."""
        return self._transmission_prob

    @transmission_prob.setter
    def transmission_prob(self, new_value):
        """Setter for transmission_prob. Raises ValueError for inputs which are less than zero
        or greater than 1."""
        if new_value < 0 or new_value > 1:
            raise ValueError(
                f"Please enter a transmission probability between 0 and 1."
            )
        self._transmission_prob = new_value

    @property
    def vaccine_frac(self):
        """The fraction of the population who have been vaccinated against the virus. It is
        assumed that the vaccine is 100% effective. Can also be seen as the number of frozen
        nodes or defects in the lattice."""
        return self._vaccine_frac

    @vaccine_frac.setter
    def vaccine_frac(self, new_value):
        """Setter for vaccine_frac. Raises ValueError for inputs that are less than zero or
        greater than one."""
        if new_value < 0 or new_value > 1:
            raise ValueError(f"Please enter a vaccination fraction between 0 and 1.")
        self._vaccine_frac = new_value

    @property
    def initial_infections(self):
        """Initial number (if int) or fraction (if float) of infected nodes."""
        return self._initial_infections

    @initial_infections.setter
    def initial_infections(self, new_value):
        """Setter for initial_infections. Raises ValueError if input is not between 1 and
        the total number of un-vaccinated nodes, or if it is a float but not between 0 and 1."""
        if type(new_value) is float:
            if new_value < 0 or new_value > 1:
                raise ValueError(
                    "Value was given as a float but could not be intepreted as a fraction of nodes since it was outside the range [0, 1]"
                )
            new_value = int(
                new_value * self.lattice.n_nodes
            )  # redefine as number of nodes
        if new_value < 1 or new_value + self.n_vaccinated > self.lattice.n_nodes:
            raise ValueError(
                f"Value must be between 1 and {self.lattice.n_nodes-self.n_vaccinated} (#nodes - #vaccinated)."
            )
        self._initial_infections = new_value

    # ----------------------------------------------------------------------------------------
    #                                                                  | Readonly properties |
    #                                                                  -----------------------

    @property
    def lattice(self):
        """The underlying lattice whose nodes represent divisions of the population (e.g.
        individuals, households, counties...)."""
        return self._lattice

    @property
    def state(self):
        """Current state of the system, represented as a 2d boolean array where True means
        the node is 'infected'."""
        return self.lattice.lexi_to_cart(self._state)

    @property
    def n_vaccinated(self):
        """Number of nodes that have been flagged as vaccinated."""
        return int(self.vaccine_frac * self.lattice.n_nodes)

    @property
    def i_vaccinated(self):
        """1d array of indices corresponding to vaccinated nodes."""
        return self._i_vaccinated

    @property
    def n_currently_infected(self):
        """Number of nodes that are currently infected."""
        return sum(self._state)

    @property
    def r_rate(self):
        """List of reproduction rates (number of transmissions per infected node) which is
        appended to as the model is evolved forwards."""
        return self._r_rate

    @property
    def infected_frac(self):
        """List of fraction of nodes that are infected, which is appended to as the model
        is evolved forwards."""
        return self._infected_frac

    # ----------------------------------------------------------------------------------------
    #                                                                              | Methods |
    #                                                                              -----------

    def seed_rng(self, seed=1):
        """Resets the random number generator with a known seed. For reproducibility."""
        self._rng = np.random.default_rng(seed)
        return

    def init_state(self):
        """Initialises the state of the model. This will set the state to contain precisely
        `self.initial_infections` infected nodes. The locations of the vaccinated nodes will
        also be reset. The locations of the infected nodes are randomly selected from the non-
        vaccinated nodes.

        Notes
        -----
        If you want to reproduce the previous simulation, you will need to reseed the
        random number generator using the `seed_rng` method *before* resetting the state.
        """
        # Generate vaccinated nodes
        self._i_vaccinated = self._rng.choice(
            np.arange(self.lattice.n_nodes),
            size=self.n_vaccinated,  # empty array if zero!
            replace=False,
        )

        # Create mask for vaccinated nodes with same shape as state
        self._vaccinated = np.full(self.lattice.n_nodes, False)
        self._vaccinated[self.i_vaccinated] = True

        # Generate state with initial infections (avoiding vaccinated nodes)
        self._state = np.full(self.lattice.n_nodes, False)
        i_infected = self._rng.choice(
            np.arange(self.lattice.n_nodes)[~self._vaccinated],
            size=self.initial_infections,
            replace=False,
        )
        self._state[i_infected] = True

        # Reset time series'
        self._r_rate = []
        self._infected_frac = []

        return

    def _update(self):
        """Performs a single update of the model."""
        # Indices corresponding to neighbours of infected nodes
        i_contacts = self.lattice.neighbours[self._state].flatten()

        # Pull out contacts who have the potential to have the virus trasmitted to them
        i_potentials = i_contacts[
            np.logical_and(
                ~self._state,  # neither already infected...
                ~self._vaccinated,  # ...nor vaccinated
            )[i_contacts]
        ]

        # Indices corresponding to nodes to which the virus has been transmitted
        i_transmissions = np.unique(
            i_potentials[self._rng.random(i_potentials.shape) < self.transmission_prob]
        )

        # Update state with new infections
        self._state[i_transmissions] = True

        # Update properties of the system
        self._r_rate.append(
            len(i_transmissions) / (self.n_currently_infected - len(i_transmissions))
        )
        self._infected_frac.append(self.n_currently_infected / self.lattice.n_nodes)

        return

    def evolve(self, n_days):
        """Evolves the model for `n_days` iterations.

        Inputs
        ------
        n_days: int
            Number of updates.
        """

        for t in range(n_days):
            self._update()

        return

    def evolve_ensemble(self, n_days):
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------
    #                                                                        | Visualisation |
    #                                                                        -----------------
    def plot_evolution(self):
        """Plot the evolution of the model."""
        fig, ax = plt.subplots()
        ax.set_xlabel("Days")
        ax.set_ylabel("Infected fraction")
        ax.plot(self.infected_frac, color="b", label="infected fraction")
        
        ax2 = ax.twinx()
        ax2.set_ylabel("Reproduction rate")
        ax2.plot(self.r_rate, color="r", label="r rate")

        fig.legend()
        fig.tight_layout()
        plt.show()
        return


    def animate(self, n_days, interval=50):
        """Evolves the model for `n_steps` iterations and produces an animation.

        Inputs
        ------
        n_steps: int
            Number of updates.
        interval: int
            Number of millisconds delay between each update.
        """
        fig, ax = plt.subplots()
        ax.set_axis_off()

        im = ax.imshow(self.state)

        def loop(t):
            self._update()
            im.set_data(self.state)
            return (im,)

        animation = mpl.animation.FuncAnimation(
            fig, loop, frames=n_days, interval=50, repeat=False, blit=True
        )
        plt.show()
        return
