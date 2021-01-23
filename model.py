import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sys import maxsize

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
        uninfected (and non-immune) node.
    vaccine_frac: float
        The fraction of nodes that are initially flagged as immune against the virus.
    initial_infections: (int, float)
        The initial number or fraction of infected nodes.
    infection_duration: int
        Number of days (i.e. time steps) that nodes remains infected.
    infected_are_immune: bool
        Whether or not nodes which have previously been infected are flagged as immune.
    travel_rate: (int, float)
        Number of 'travel' events (which are random swaps of nodes) per update.

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
        initial_infections=1,
        infection_duration=maxsize,
        infected_are_immune=False,
        travel_rate=0,
    ):
        # For now, just check that the lattice is a SquareLattice
        if type(lattice) != SquareLattice:
            raise ValueError("Please provide an instance of SquareLattice.")
        self._lattice = lattice

        # Set parameters which users can modify
        self.transmission_prob = transmission_prob
        self.vaccine_frac = vaccine_frac
        self.initial_infections = initial_infections
        self.infection_duration = infection_duration
        self.infected_are_immune = infected_are_immune
        self.travel_rate = travel_rate

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
        assumed that the vaccine is 100% effective. Can also be seen as the fraction of
        initially frozen nodes, or lattice defects."""
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

    @property
    def infection_duration(self):
        """Number of days (i.e. time steps) that nodes remain infected after initially
        contracting the virus."""
        return self._infection_duration

    @infection_duration.setter
    def infection_duration(self, new_value):
        """Setter for infection_duration. Raises ValueError if input is less than 1."""
        if new_value < 1:
            raise ValueError("Please enter an infection duration of 1 or more days." "")
        self._infection_duration = new_value

    @property
    def infected_are_immune(self):
        """Whether or not nodes which have previously been infected are flagged as immune."""
        return self._infected_are_immune

    @infected_are_immune.setter
    def infected_are_immune(self, new_flag):
        """Setter for infected_are_immune. Raises ValueError if input is not a bool."""
        if type(new_flag) is not bool:
            raise ValueError("Please enter True/False for infected_are_immune.")
        self._infected_are_immune = new_flag

    @property
    def travel_rate(self):
        """Number of 'travel' events (which are random swaps of nodes) per update."""
        return self._travel_rate

    @travel_rate.setter
    def travel_rate(self, new_value):
        """Setter for travel_rate. Raises ValueError if input is less than 0."""
        if new_value < 0:
            raise ValueError("Please enter an number of journeys of 0 or greater")
        self._travel_rate = new_value

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
        only just been infected take a value of `self.infection_duration`, and this count
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
        """Numpy array containing the of fraction of nodes that are immune to infection,
        either because they are vaccinated or because they have previously had the virus.
        The list is appended to as the model is evolved forwards."""
        return np.array(self._immune_time_series) / self.lattice.n_nodes

    # ----------------------------------------------------------------------------------------
    #                                                                    | Protected methods |
    #                                                                    ---------------------

    def _append_time_series(self):
        """Helper function that appends information about the current state of the model to
        lists containing time series'."""
        n_infected = sum(self._state.astype(bool))
        n_immune = sum(self._immune.astype(bool))
        self._infected_time_series.append(n_infected)
        self._immune_time_series.append(n_immune)
        self._susceptible_time_series.append(
            self.lattice.n_nodes - n_infected - n_immune
        )

    def _swap_nodes(self):
        """Swap a random pair of nodes."""
        i_travel = self._rng.choice(
            range(self.lattice.n_nodes), size=(2 * self.travel_rate), replace=False
        )
        self._state[i_travel] = self._state[np.flip(i_travel)]
        self._immune[i_travel] = self._immune[np.flip(i_travel)]

    def _update(self):
        """Performs a single update of the model."""
        # Update array of immune nodes with those that are about to lose their infection
        if self.infected_are_immune:
            np.logical_or(self._immune, (self._state == 1), out=self._immune)

        if self.travel_rate > 0:
            self._swap_nodes()

        # Update state by reducing the 'days' counter
        np.clip(self._state - 1, a_min=0, a_max=None, out=self._state)

        # Indices corresponding to neighbours of infected nodes
        i_contacts = self.lattice.neighbours[self._state.astype(bool)].flatten()

        # Pull out contacts who have the potential to have the virus trasmitted to them
        i_potentials = i_contacts[
            np.logical_and(
                ~self._state.astype(bool),  # neither already infected...
                ~self._immune.astype(bool),  # ...nor immune
            )[i_contacts]
        ]

        # Indices corresponding to nodes to which the virus has been transmitted
        i_transmissions = np.unique(
            i_potentials[self._rng.random(i_potentials.shape) < self.transmission_prob]
        )

        # Update state with new infections
        self._state[i_transmissions] = self.infection_duration

        # Append the latest data to the time series'
        self._append_time_series()

    # ----------------------------------------------------------------------------------------
    #                                                                       | Public methods |
    #                                                                       ------------------

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
        i_vaccinated = self._rng.choice(
            np.arange(self.lattice.n_nodes),
            size=self.n_vaccinated,  # empty array if zero!
            replace=False,
        )

        # Create mask for vaccinated nodes with same shape as state
        self._immune = np.full(self.lattice.n_nodes, False)
        self._immune[i_vaccinated] = True

        # Generate state with initial infections (avoiding vaccinated nodes)
        self._state = np.full(self.lattice.n_nodes, 0)
        i_infected = self._rng.choice(
            np.arange(self.lattice.n_nodes)[~self._immune],
            size=self.initial_infections,
            replace=False,
        )
        self._state[i_infected] = self.infection_duration

        # Reset time series' to empty lists then append initial conditions
        self._infected_time_series = []
        self._susceptible_time_series = []
        self._immune_time_series = []
        self._append_time_series()

    def evolve(self, n_days):
        """Evolves the model for `n_days` iterations. Displays progress bar.

        Inputs
        ------
        n_days: int
            Number of updates.
        """
        for t in tqdm(range(n_days), desc="Days"):
            self._update()

    def evolve_ensemble(self, n_days):
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------
    #                                                                        | Visualisation |
    #                                                                        -----------------

    def plot_evolution(self, critical_threshold=0.1):
        """Plots the time evolution of the model.

        More specifically, plots the evolution of the fraction of nodes that are (a) susceptible
        to infection, (b) infected, and (c) immune from the virus due to either being vaccinated
        or having previously had it.

        Also plots a horizontal line representing a critical threshold of the infected fraction,
        which we would prefer to avoid acrossing.

        Inputs
        ------
        critical_threshold: float
            The threshold infected fraction above which bad things happen...
        """
        if critical_threshold < 0 and critical_threshold > 1:
            raise ValueError("Please provide a critical threshold between 0 and 1")

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

        # Plot horizontal line at critical threshold
        ax.axhline(
            critical_threshold,
            linestyle="--",
            color="orange",
            label="critical threshold",
        )

        # Shade area between infections curve and critical threshold
        n_days = len(self.infected_time_series)
        ax.fill_between(
            x=np.arange(n_days),
            y1=self.infected_time_series,
            y2=np.full(n_days, critical_threshold),
            where=self.infected_time_series > critical_threshold,
            color="orange",
            alpha=0.2,
        )

        # Print some useful diagonstics
        days_above_thresh = sum(self.infected_time_series > critical_threshold)
        print(
            f"{days_above_thresh}/{n_days} days spent above the critical threshold of {critical_threshold}"
        )
        area_above_thresh = sum(
            self.infected_time_series[self.infected_time_series > critical_threshold]
        )
        print(f"Area above critical threshold: {area_above_thresh}")

        ax.legend()
        fig.tight_layout()
        plt.show()

    def animate(self, n_days, interval=20):
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

        image = ax.imshow(
            self.state,
            norm=mpl.colors.Normalize(vmin=0, vmax=self.infection_duration),
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

        def loop(t):
            if t == 0:  # otherwise the animation starts a frame late in Jupyter...
                return (image, overlay)
            self._update()
            image.set_data(self.state)
            overlay.set_data(self.immune)
            return image, overlay

        animation = mpl.animation.FuncAnimation(
            fig, loop, frames=n_days + 1, interval=interval, repeat=False, blit=True
        )
        return animation
