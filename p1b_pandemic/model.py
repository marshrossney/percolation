import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import maxsize
from pathlib import Path

from p1b_pandemic.lattice import SquareLattice


plt.style.use(Path(__file__).resolve().parent / "p1b.mplstyle")


class PandemicModel:
    """Class containing a simple model of a pandemic.

    Inputs
    ------
    lattice: lattice.SquareLattice
        The underlying lattice upon which we perform the simulation.
    transmission_prob: float
        Probability of an infected node transmitting the virus to a susceptible contact
        upon a single refresh of the model.
    vaccine_frac: float
        Fraction of nodes that are initially flagged as immune against the virus.
    recovery_time: int
        Number of time steps before an infected node is considered to have recovered,
        and is no longer able to spread the infection.
    recovered_are_immune: bool
        Nodes which have recovered from the infection are flagged as immune.
    travel_prob: float
        Probability for any given node to 'travel', which is to shuffle positions
        with all other travelling nodes at any given time.
    nucleus_size: int
        Linear size (side length) of the initial nucleus (square) of infected nodes.
        Alternatively, a negative number means that the initial state will have a line
        of infected nodes on one side, instead of a nucleus.

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
        transmission_prob=1.0,
        vaccine_frac=0.0,
        recovery_time=maxsize,
        recovered_are_immune=False,
        travel_prob=0.0,
        nucleus_size=1,
    ):
        # TODO: upgrade so we can use more general graphs
        # For now, just check that the lattice is a SquareLattice
        if type(lattice) != SquareLattice:
            raise ValueError("Please provide an instance of SquareLattice.")
        self._lattice = lattice

        # Set parameters which users can modify
        self.transmission_prob = transmission_prob
        self.vaccine_frac = vaccine_frac
        self.recovery_time = recovery_time
        self.recovered_are_immune = recovered_are_immune
        self.travel_prob = travel_prob
        self.nucleus_size = nucleus_size

        # Initalise the model and random number generator (with a known seed)
        self.init_state(reproducible=True)

    # ----------------------------------------------------------------------------------------
    #                                                                     | Data descriptors |
    #                                                                     --------------------

    @property
    def transmission_prob(self):
        """The probability of transmitting the virus from an infected node to a neighbouring
        non-infected (and non-vaccinated) node."""
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
        assumed that the vaccine is 100% effective."""
        return self._vaccine_frac

    @vaccine_frac.setter
    def vaccine_frac(self, new_value):
        """Setter for vaccine_frac. Raises ValueError for inputs that are less than zero or
        greater than one."""
        if new_value < 0 or new_value > 1:
            raise ValueError(f"Please enter a vaccination fraction between 0 and 1.")
        self._vaccine_frac = new_value

    @property
    def recovery_time(self):
        """Number of time steps before an infected node is considered to have recovered,
        and is no longer able to spread the infection."""
        return self._recovery_time

    @recovery_time.setter
    def recovery_time(self, new_value):
        """Setter for recovery_time. Raises ValueError if input is less than 1."""
        if new_value < 1:
            raise ValueError("Please enter a recovery time of 1 or more days.")
        self._recovery_time = new_value

    @property
    def recovered_are_immune(self):
        "Nodes which have recovered from the infection are flagged as immune."
        return self._recovered_are_immune

    @recovered_are_immune.setter
    def recovered_are_immune(self, new_flag):
        """Setter for recovered_are_immune. Raises TypeError if input is not a bool."""
        if type(new_flag) is not bool:
            raise TypeError("Please enter True/False for recovered_are_immune.")
        self._recovered_are_immune = new_flag

    @property
    def travel_prob(self):
        """Probability for any given node to 'travel', which is to shuffle positions
        with all other travelling nodes at any given time."""
        return self._travel_prob

    @travel_prob.setter
    def travel_prob(self, new_value):
        """Setter for travel_prob. Raises ValueError if input is less than 0 or greater
        than 1."""
        if new_value < 0 or new_value > 1:
            raise ValueError("Please enter a travel probability between 0 and 1.")
        self._travel_prob = new_value

    @property
    def nucleus_size(self):
        """Linear size (side length) of the initial nucleus (square) of infected nodes.
        Alternatively, a negative number means that the initial state will have a line
        of infected nodes on one side, instead of a nucleus."""
        return self._nucleus_size

    @nucleus_size.setter
    def nucleus_size(self, new_value):
        """Setter for nucleus_size. Raises TypeError if input is not an integer and
        raises ValueError if input will result in a nucleus that is too large for the
        lattice, taking the immune nodes into account."""
        if type(new_value) is not int:
            raise TypeError("Please provide an integer for the nucleus size.")
        if (new_value ** 2) / self.lattice.n_nodes + self.vaccine_frac > 1:
            raise ValueError(
                f"Not enough nodes on the lattice to support a nucleus of this size with this vaccine fraction."
            )
        self._nucleus_size = new_value

    # ----------------------------------------------------------------------------------------
    #                                                                 | Read-only properties |
    #                                                                 ------------------------

    @property
    def lattice(self):
        """The underlying lattice whose nodes represent divisions of the population (e.g.
        individuals, households, counties...)."""
        return self._lattice

    @property
    def state(self):
        """Current state of the system, represented as a 2d integer array. Nodes which have
        only just been infected take a value of `self.recovery_time`, and this count
        decreases by one for each update. Zeros are interpreted as not being infected."""
        return self.lattice.lexi_to_cart(self._state)

    @property
    def immune(self):
        """Currently immune nodes, represented as a 2d boolean array where True means the
        node is immune."""
        return self.lattice.lexi_to_cart(self._immune)

    @property
    def n_vaccinated(self):
        """Number of nodes that have been flagged as vaccinated."""
        return int(self.vaccine_frac * self.lattice.n_nodes)

    @property
    def infected_time_series(self):
        """Numpy array containing the fraction of nodes that are infected, which is appended
        to as the model is evolved forwards."""
        return np.array(self._infected_time_series) / self.lattice.n_nodes

    @property
    def susceptible_time_series(self):
        """Numpy array containing the fraction of nodes that are susceptible to infection,
        which is appended to as the model is evolved forwards."""
        return np.array(self._susceptible_time_series) / self.lattice.n_nodes

    @property
    def immune_time_series(self):
        """Numpy array containing the fraction of nodes that are immune to infection,
        either because they are vaccinated or because they have previously had the virus.
        The list is appended to as the model is evolved forwards."""
        return np.array(self._immune_time_series) / self.lattice.n_nodes

    @property
    def has_percolated(self):
        """Returns True if infections have reached the 'far boundary' defined by the
        underlying lattice object."""
        if np.any(self._state[self.lattice.far_boundary_mask]):
            return True
        else:
            return False

    # ----------------------------------------------------------------------------------------
    #                                                                    | Protected methods |
    #                                                                    ---------------------

    def _seed_rng(self, seed=None):
        """Resets the random number generator with a seed, for reproducibility. If no
        seed is provided the random number generated will be randomly re-initialised.
        """
        self._rng = np.random.default_rng(seed)

    def _update_time_series(self):
        """Helper function that appends information about the current state of the model to
        lists containing time series'."""
        n_infected = self._state.astype(bool).sum()
        n_immune = self._immune.astype(bool).sum()
        self._infected_time_series.append(n_infected)
        self._immune_time_series.append(n_immune)
        self._susceptible_time_series.append(
            self.lattice.n_nodes - n_infected - n_immune
        )

    def _shuffle_nodes(self):
        """Shuffle a subset of the nodes based on drawing uniform random numbers and
        comparing these to the travel probability."""
        i_travel = np.where(self._rng.random(self._state.size) < self.travel_prob)[0]
        if i_travel.size > 0:
            i_travel_permuted = self._rng.permutation(i_travel)
            self._state[i_travel] = self._state[i_travel_permuted]
            self._immune[i_travel] = self._immune[i_travel_permuted]

    def _update(self):
        """Performs a single update of the model.

        Returns
        -------
        n_transmissions: int
            number of transmissions for this update
        """

        # Shuffle a subset of nodes to simulate 'travel'
        if self.travel_prob > 0:
            self._shuffle_nodes()

        # If there are no infected nodes, just continue to save time
        if self._infected_time_series[-1] == 0:
            self._update_time_series()
            return 0 

        # Update array of immune nodes with those that are about to recover
        if self.recovered_are_immune:
            np.logical_or(self._immune, (self._state == 1), out=self._immune)

        # Update state by reducing the 'days' counter
        np.clip(self._state - 1, a_min=0, a_max=None, out=self._state)

        # Indices corresponding to neighbours ('contacts') of infected nodes
        i_contacts = self.lattice.neighbours[self._state.astype(bool)].flatten()

        # Indices of 'susceptible' contacts who can potentially become infected
        i_potentials = i_contacts[
            np.logical_and(
                ~self._state.astype(bool),  # neither already infected...
                ~self._immune.astype(bool),  # ...nor immune
            )[i_contacts]
        ]

        # Indices of nodes to which the virus has just been transmitted
        i_transmissions = np.unique(
            i_potentials[self._rng.random(i_potentials.size) < self.transmission_prob]
        )

        # Update state with new infections
        self._state[i_transmissions] = self.recovery_time

        # Append the latest data to the time series'
        self._update_time_series()

        return len(i_transmissions)


    # ----------------------------------------------------------------------------------------
    #                                                                       | Public methods |
    #                                                                       ------------------

    def init_state(self, reproducible=False):
        """Initialises the state of the model by first creating the initial nucleus or
        line of infected nodes, and then randomly assigning the correct number of immune
        nodes to those remaining.

        Inputs
        ------
        reproducible: bool (optional)
            If True, initialise the random number generator with a known seed, so that
            the simulation can be reproduced exactly. Otherwise, use a random seed.
        """
        # Seed random number generator
        if reproducible:
            self._seed_rng(seed=123456)
        else:
            self._seed_rng(seed=None)

        # Generate initial infections
        self._state = np.zeros(self.lattice.n_nodes)
        nucleus_mask = self.lattice.get_nucleus_mask(nucleus_size=self.nucleus_size)
        self._state[nucleus_mask] = self.recovery_time

        # Generate vaccinated nodes, avoiding the initially infected ones
        i_vaccinated = self._rng.choice(
            np.arange(self.lattice.n_nodes)[~self._state.astype(bool)],
            size=self.n_vaccinated,  # empty array if zero!
            replace=False,
        )

        # Create mask for vaccinated nodes with same shape as state
        self._immune = np.full(self.lattice.n_nodes, False)
        self._immune[i_vaccinated] = True

        # Reset time series' to empty lists then append initial conditions
        self._infected_time_series = []
        self._susceptible_time_series = []
        self._immune_time_series = []
        self._update_time_series()

    def evolve(self, n_days):
        """Evolves the model for `n_days` iterations.

        Inputs
        ------
        n_days: int
            Number of updates.
        """
        if type(n_days) is not int:
            raise TypeError("Please provide an integer for the number of days to evolve.")
        if n_days < 1:
            raise ValueError("Please enter a positive number of days.")

        for day in range(n_days):
            _ = self._update()
    
    def evolve_until_percolated(self):
        """Evolve until percolation occurs or transmission halts. Percolation is defined
        as one or more nodes on the 'far boundary' being infected. Transmission halting
        is defined as having no transmissions for 1 / self.transmission_prob days.
        """
        days_without_new_infections = 0
        days_simulated = 0

        while days_without_new_infections < (1 / self.transmission_prob):
            n_transmissions = self._update()
            days_simulated += 1

            if n_transmissions == 0:
                days_without_new_infections += 1
            else:
                days_without_new_infections = 0

            if days_simulated % 10 == 0:
                if self.has_percolated:
                    break

    # ----------------------------------------------------------------------------------------
    #                                                                        | Visualisation |
    #                                                                        -----------------

    def plot_evolution(self, outpath=None):
        """Plots the time evolution of the model.

        More specifically, plots the evolution of the fraction of nodes that are (a) susceptible
        to infection, (b) infected, and (c) immune from the virus due to either being vaccinated
        or having previously had it.

        Also plots a horizontal line representing a critical threshold of the infected fraction,
        which we would prefer to avoid acrossing.

        Inputs
        ------
        outpath: str (optional)
            If provided, specifies path to a directory in which the plot will be saved
            as 'plot.png'.
        """
        fig, ax = plt.subplots()
        ax.set_title("Time evolution of the pandemic")
        ax.set_xlabel("Days since patient 0")
        ax.set_ylabel("Fraction of nodes")
        ax.plot(
            self.susceptible_time_series,
            color="blue",
            label="susceptible",
        )
        ax.plot(
            self.infected_time_series,
            color="red",
            label="infected",
        )
        ax.plot(
            self.immune_time_series,
            color="grey",
            label="immune",
        )

        ax.legend()
        fig.tight_layout()

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath / "plot.png")
        else:
            plt.show()

    def animate(self, n_days, interval=25, outpath=None):
        """Evolves the model for `n_steps` iterations and produces an animation.

        Inputs
        ------
        n_steps: int
            Number of updates.
        interval: int (optional)
            Number of millisconds delay between each update.
        outpath: str (optional)
            If provided, specifies path to a directory in which the plot will be saved
            as 'animation.gif'.
        """
        if type(n_days) is not int:
            raise TypeError("Please provide an integer for the number of days to animate.")
        if n_days < 1:
            raise ValueError("Please enter a positive number of days for the animation.")

        fig, ax = plt.subplots()
        ax.set_axis_off()

        image = ax.imshow(
            self.state,
            norm=mpl.colors.Normalize(vmin=0, vmax=self.recovery_time),
            zorder=0,
        )
        overlay = ax.imshow(
            self.immune,
            cmap=mpl.colors.ListedColormap(
                ["#66666600", "#666666"]
            ),  # [transparent, grey]
            norm=mpl.colors.Normalize(vmin=0, vmax=1),
            zorder=1,
        )
        day_counter = ax.annotate(
            text=f"Day 0",
            xy=(0, -0.11),
            xycoords="axes fraction",
        )

        def loop(t):
            if t == 0:  # otherwise the animation starts a frame late in Jupyter...
                return (image, overlay)
            _ = self._update()
            image.set_data(self.state)
            overlay.set_data(self.immune)
            day_counter.set_text(f"Day {t}")
            return image, overlay, day_counter

        animation = mpl.animation.FuncAnimation(
            fig, loop, frames=n_days + 1, interval=interval, repeat=False, blit=True
        )

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            animation.save(outpath / "animation.gif")

        return animation
