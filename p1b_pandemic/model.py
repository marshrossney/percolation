import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
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
    travel_rate: (int, float)
        Number of 'travel' events (which are random swaps of nodes) per update.
    outpath: str
        Path to directory in which to save output data, figures, animations.

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
        travel_rate=0,
        outpath="",
    ):
        # For now, just check that the lattice is a SquareLattice
        if type(lattice) != SquareLattice:
            raise ValueError("Please provide an instance of SquareLattice.")
        self._lattice = lattice

        # Set parameters which users can modify
        self.transmission_prob = transmission_prob
        self.vaccine_frac = vaccine_frac
        self.recovery_time = recovery_time
        self.recovered_are_immune = recovered_are_immune
        self.travel_rate = travel_rate
        self.outpath = outpath

        # TODO
        # Initialise the rng (uses known default seed)
        # self.seed_rng()

        # Initalise the model with a single infected site
        # self.init_state()

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
        """Setter for recovered_are_immune. Raises ValueError if input is not a bool."""
        if type(new_flag) is not bool:
            raise ValueError("Please enter True/False for recovered_are_immune.")
        self._recovered_are_immune = new_flag

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

    @property
    def outpath(self):
        """Path to directory in which to save output data, plots, animations."""
        return self._outpath

    @outpath.setter
    def outpath(self, new_path):
        """Setter for outpath. Saves as pathlib.Path object."""
        self._outpath = Path(new_path)

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
        n_infected = self._state.astype(bool).sum()
        n_immune = self._immune.astype(bool).sum()
        self._infected_time_series.append(n_infected)
        self._immune_time_series.append(n_immune)
        self._susceptible_time_series.append(
            self.lattice.n_nodes - n_infected - n_immune
        )

    def _swap_nodes(self):
        """Swap random pair of nodes. The number of pairs is given by `travel_rate`."""
        i_travel = self._rng.choice(
            self.lattice.n_nodes, size=(2 * self.travel_rate), replace=False
        )
        self._state[i_travel] = self._state[np.flip(i_travel)]
        self._immune[i_travel] = self._immune[np.flip(i_travel)]

    def _update(self):
        """Performs a single update of the model."""

        # Update array of immune nodes with those that are about to recover
        if self.recovered_are_immune:
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
        self._state[i_transmissions] = self.recovery_time

        # Append the latest data to the time series'
        self._append_time_series()

    # ----------------------------------------------------------------------------------------
    #                                                                       | Public methods |
    #                                                                       ------------------

    def seed_rng(self, seed=None):
        """Resets the random number generator with a seed, for reproducibility. If no
        seed is provided the random number generated will be randomly re-initialised.
        """
        self._rng = np.random.default_rng(seed)
        return

    def init_state(self, initial_shape="nucleus", nucleus_size=1):
        """Initialises the state of the model. This will set the state to contain precisely
        `self.initial_infections` infected nodes. The locations of the vaccinated nodes will
        also be reset. The locations of the infected nodes are randomly selected from the non-
        vaccinated nodes.

        Inputs:
        -------
        initial_shape: str
            The shape of the initially infected nodes. Either 'nucleus' for a square nucleus
            in the center of the lattice, or 'line' for a single line of infected nodes on the
            leftmost vertical edge of the lattice.
        nucleus_size: int
            Linear size of the initial nucleus of infections, i.e. side length of the square.

        Notes
        -----
        If you want to reproduce the previous simulation, you will need to reseed the
        random number generator using the `seed_rng` method *before* resetting the state.
        """
        if initial_shape not in ("nucleus", "line"):
            raise ValueError(
                "Please enter either 'nucleus' or 'line' for initial_shape"
            )
        if (nucleus_size ** 2) / self.lattice.n_nodes + self.vaccine_frac > 1:
            raise ValueError(
                f"Not enough nodes on the lattice to support a nucleus of this size with the vaccine fraction provided"
            )

        # Generate initial infections
        state_cart = np.full((self.lattice.length, self.lattice.length), 0)
        if initial_shape == "line":
            state_cart[:, 0] = self.recovery_time  # left column
        else:
            corner = self.lattice.length // 2 - nucleus_size // 2
            i_nucl = slice(corner, corner + nucleus_size)
            state_cart[i_nucl, i_nucl] = self.recovery_time
        self._state = state_cart.flatten()

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
        self._append_time_series()

    def evolve(self, n_days, display_progress_bar=True):
        """Evolves the model for `n_days` iterations.

        Inputs
        ------
        n_days: int
            Number of updates.
        display_progress_bar: bool (optional)
            Flag indicating whether or not to display a progress bar. Default: True.
        """
        if display_progress_bar:
            generator = tqdm(range(n_days), desc="Days")
        else:
            generator = range(n_days)

        for t in generator:
            self._update()

    def evolve_ensemble(self, n_days):
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------
    #                                                                        | Visualisation |
    #                                                                        -----------------

    def plot_evolution(self, critical_threshold=0.1, save=False):
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
        days_above_thresh = (self.infected_time_series > critical_threshold).sum()
        print(
            f"{days_above_thresh}/{n_days} days spent above the critical threshold of {critical_threshold}"
        )
        area_above_thresh = (
            self.infected_time_series[self.infected_time_series > critical_threshold]
        ).sum()
        print(f"Area above critical threshold: {area_above_thresh}")

        ax.legend()
        fig.tight_layout()

        if save:
            self.outpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.outpath / "plot.png")
        else:
            plt.show()

    def animate(self, n_days, interval=25, save=False):
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
            xy=(1, 0),
            xycoords="axes fraction",
        )

        def loop(t):
            if t == 0:  # otherwise the animation starts a frame late in Jupyter...
                return (image, overlay)
            self._update()
            image.set_data(self.state)
            overlay.set_data(self.immune)
            day_counter.set_text(f"Day {t}")
            return image, overlay, day_counter

        animation = mpl.animation.FuncAnimation(
            fig, loop, frames=n_days + 1, interval=interval, repeat=False, blit=True
        )

        if save:
            self.outpath.mkdir(parents=True, exist_ok=True)
            animation.save(self.outpath / "animation.gif")

        return animation
