import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, animation
from sys import maxsize
from pathlib import Path

from lattice import SquareLattice


plt.style.use(Path(__file__).resolve().parent / "p1b.mplstyle")


class PercolationModel:
    """Class containing a percolation model.

    Inputs
    ------
    network: lattice.SquareLattice
        The underlying network upon which we perform the percolation simulation.
    inert_prob: float
        Probability that any given node will be initially flagged as 'inert' i.e. not
        susceptible. In the notebooks, this is `q`.
    transmission_prob: float
        Probability of a 'live' node transmitting to a susceptible node it is connected
        to upon a single refresh of the model.
    recovery_time: int
        Number of time steps before a live node is considered to have recovered, and is
        no longer able to transmit.
    recovered_are_inert: bool
        Nodes which have recovered are flagged as inert.
    shuffle_prob: float
        Probability for any given node to shuffle positions with all other 'shuffling'
        nodes at any given time.
    nucleus_size: int
        Linear size (side length) of the initial nucleus (square) of live nodes.

    Notes
    -----
        For now, only a square lattice is implemented, but any data structure
        that includes a set of nodes and edges (a graph) would be valid in
        principle. This may require further modifications to this class if the
        number of links attached to each node varies between nodes.
    """

    def __init__(
        self,
        network,
        inert_prob=0.0,
        *,
        transmission_prob=1.0,
        recovery_time=-1,
        recovered_are_inert=True,
        shuffle_prob=0.0,
        nucleus_size=1,
    ):
        # TODO: upgrade so we can use more general networks
        # For now, just check that the network is a SquareLattice
        if type(network) != SquareLattice:
            raise ValueError("Please provide an instance of SquareLattice.")
        self._network = network

        # Set parameters which users can modify
        self.inert_prob = inert_prob
        self.transmission_prob = transmission_prob
        self.recovery_time = recovery_time
        self.recovered_are_inert = recovered_are_inert
        self.shuffle_prob = shuffle_prob
        self.nucleus_size = nucleus_size

        # Initalise the model and random number generator
        self.init_state(reproducible=False)

    # ----------------------------------------------------------------------------------------
    #                                                                     | Data descriptors |
    #                                                                     --------------------
    @property
    def inert_prob(self):
        """Probability that any given node will be initially flagged as inert i.e. not
        susceptible."""
        return self._inert_prob

    @inert_prob.setter
    def inert_prob(self, new_value):
        """Setter for inert_prob. Raises ValueError for inputs that are less than zero or
        greater than one."""
        if new_value < 0 or new_value > 1:
            raise ValueError(f"Please enter a inert probability between 0 and 1.")
        self._inert_prob = new_value

    @property
    def transmission_prob(self):
        """Probability of a 'live' node transmitting to a susceptible node it is connected
        to upon a single refresh of the model."""
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
    def recovery_time(self):
        """Number of time steps before a live node is considered to have recovered, and
        is no longer able to transmit."""
        return self._recovery_time

    @recovery_time.setter
    def recovery_time(self, new_value):
        """Setter for recovery_time. Providing a negative number sets the recovery time
        to be effectively infinite. Raises TypeError if input is not an int."""
        if type(new_value) is not int:
            raise TypeError("Please provide an integer for the recovery time")
        if new_value < 0:
            new_value = maxsize - 1
        # Add one since order of update loop is to reduce step counter first
        self._recovery_time = new_value + 1

    @property
    def recovered_are_inert(self):
        "Nodes which have recovered are flagged as inert."
        return self._recovered_are_inert

    @recovered_are_inert.setter
    def recovered_are_inert(self, new_flag):
        """Setter for recovered_are_inert. Raises TypeError if input is not a bool."""
        if type(new_flag) is not bool:
            raise TypeError("Please enter True/False for recovered_are_inert.")
        self._recovered_are_inert = new_flag

    @property
    def shuffle_prob(self):
        """Probability for any given node to shuffle positions with all other 'shuffling'
        nodes at any given time."""
        return self._shuffle_prob

    @shuffle_prob.setter
    def shuffle_prob(self, new_value):
        """Setter for shuffle_prob. Raises ValueError if input is less than 0 or greater
        than 1."""
        if new_value < 0 or new_value > 1:
            raise ValueError("Please enter a travel probability between 0 and 1.")
        self._shuffle_prob = new_value

    @property
    def nucleus_size(self):
        """Linear size (side length) of the initial nucleus (square) of live nodes."""
        return self._nucleus_size

    @nucleus_size.setter
    def nucleus_size(self, new_value):
        """Setter for nucleus_size. Raises TypeError if input is not an integer and
        raises ValueError if input will result in a nucleus that is too large for the
        network."""
        if type(new_value) is not int:
            raise TypeError("Please provide an integer for the nucleus size.")
        if new_value < 1:
            raise ValueError("Please provide a positive number for the nucleus size.")
        if (new_value ** 2) / self.network.n_nodes > 1:
            raise ValueError(
                f"Not enough nodes to support a nucleus of this size with this vaccine fraction."
            )
        self._nucleus_size = new_value

    # ----------------------------------------------------------------------------------------
    #                                                                 | Read-only properties |
    #                                                                 ------------------------

    @property
    def network(self):
        """The underlying network whose nodes represent divisions of the population (e.g.
        individuals, households, counties...)."""
        return self._network

    @property
    def state(self):
        """Current state of the system, represented as a 2d integer array. Nodes which
        have only just become live take a value of `self.recovery_time`, and this count
        decreases by one for each update. Zeros are interpreted as not being live."""
        return self.network.lexi_to_cart(self._state)

    @property
    def inert(self):
        """Currently inert nodes, represented as a 2d boolean array where True means the
        node is inert."""
        return self.network.lexi_to_cart(self._inert)

    @property
    def live_time_series(self):
        """Numpy array containing the fraction of nodes that are live, which is appended
        to as the model is evolved forwards."""
        return np.array(self._live_time_series) / self.network.n_nodes

    @property
    def susceptible_time_series(self):
        """Numpy array containing the fraction of nodes that are susceptible, i.e.
        neither inert nor live, which is appended to as the model is evolved forwards.
        """
        return np.array(self._susceptible_time_series) / self.network.n_nodes

    @property
    def inert_time_series(self):
        """Numpy array containing the fraction of nodes that are inert. The list is
        appended to as the model is evolved forwards."""
        return np.array(self._inert_time_series) / self.network.n_nodes

    @property
    def has_percolated(self):
        """Returns True if the percolating substance has reached the 'far boundary'
        defined by the underlying network object."""
        if np.any(self._state[self.network.far_boundary_mask]):
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
        n_live = self._state.astype(bool).sum()
        n_inert = self._inert.astype(bool).sum()
        self._live_time_series.append(n_live)
        self._inert_time_series.append(n_inert)
        self._susceptible_time_series.append(self.network.n_nodes - n_live - n_inert)

    def _shuffle_nodes(self):
        """Shuffle a subset of the nodes based on drawing uniform random numbers and
        comparing these to the travel probability."""
        i_shuffle = np.where(self._rng.random(self._state.size) < self.shuffle_prob)[0]
        if i_shuffle.size > 0:
            i_shuffle_permuted = self._rng.permutation(i_shuffle)
            self._state[i_shuffle] = self._state[i_shuffle_permuted]
            self._inert[i_shuffle] = self._inert[i_shuffle_permuted]

    def _update(self):
        """Performs a single update of the model.

        Returns
        -------
        n_transmissions: int
            number of transmissions for this update
        """

        # Shuffle a subset of nodes to simulate 'travel'
        if self.shuffle_prob > 0:
            self._shuffle_nodes()

        # If there are no live nodes, just continue to save time
        if self._live_time_series[-1] == 0:
            self._update_time_series()
            return 0

        # Update array of inert nodes with those that are about to recover
        if self.recovered_are_inert:
            np.logical_or(self._inert, (self._state == 1), out=self._inert)

        # Update state by reducing the 'days' counter
        np.clip(self._state - 1, a_min=0, a_max=None, out=self._state)

        # Indices corresponding to neighbours ('contacts') of live nodes
        i_contacts = self.network.neighbours[self._state.astype(bool)].flatten()

        # Indices of 'susceptible' contacts who can potentially be transmitted to
        i_potentials = i_contacts[
            np.logical_and(
                ~self._state.astype(bool),  # neither already live...
                ~self._inert.astype(bool),  # ...nor inert
            )[i_contacts]
        ]

        # Indices of nodes to which the virus has just been transmitted
        i_transmissions = np.unique(
            i_potentials[self._rng.random(i_potentials.size) < self.transmission_prob]
        )

        # Update state with new live nodes
        self._state[i_transmissions] = self.recovery_time

        # Append the latest data to the time series'
        self._update_time_series()

        return len(i_transmissions)

    # ----------------------------------------------------------------------------------------
    #                                                                       | Public methods |
    #                                                                       ------------------

    def init_state(self, reproducible=False):
        """Initialises the state of the model by first creating the initial nucleus or
        line of live nodes, and then randomly generating inert nodes with a probability
        equal to self.inert_prob.

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

        # Generate initial nucleus
        self._state = np.zeros(self.network.n_nodes)
        nucleus_mask = self.network.get_nucleus_mask(nucleus_size=self.nucleus_size)
        self._state[nucleus_mask] = self.recovery_time

        # Create mask for inert nodes with same shape as state
        self._inert = np.logical_and(
            self._rng.random(self._state.size) < self.inert_prob,  # rand < prob
            ~self._state.astype(bool),  # not part of initial nucleus
        )

        # Reset time series' to empty lists then append initial conditions
        self._live_time_series = []
        self._susceptible_time_series = []
        self._inert_time_series = []
        self._update_time_series()

    def evolve(self, n_steps):
        """Evolves the model for `n_steps` iterations.

        Inputs
        ------
        n_steps: int
            Number of updates.
        """
        if type(n_steps) is not int:
            raise TypeError(
                "Please provide an integer for the number of steps to evolve for."
            )
        if n_steps < 1:
            raise ValueError("Please enter a positive number of steps.")

        for step in range(n_steps):
            _ = self._update()

    def evolve_until_percolated(self):
        """Evolve until percolation occurs or transmission halts. Percolation is defined
        as one or more nodes on the 'far boundary' being reached. Transmission halting
        is defined as having no transmissions for 1 / self.transmission_prob days.
        """
        steps_without_transmission = 0
        steps_simulated = 0

        while steps_without_transmission < (1 / self.transmission_prob):
            n_transmissions = self._update()
            steps_simulated += 1

            if n_transmissions == 0:
                steps_without_transmission += 1
            else:
                steps_without_transmission = 0

            if steps_simulated % 10 == 0:
                if self.has_percolated:
                    break

    def estimate_percolation_prob(self, repeats=25, print_result=True):
        """Loops over evolve_until_percolated and returns the fraction of simulations
        which percolated.

        Input
        -----
        repeats: int (optional)
            Number of simulations to run
        print_result: bool (optional)
            Pretty-print the mean and standard error on the estimate of the percolation
            fraction.
        
        Returns
        -------
        frac: float
            Fraction of the `repeats` simulations that percolated.
        stderr: float
            Estimate of the standard error on the above estimate of the percolation
            probability.
        """
        num = 0
        for rep in range(repeats):
            self.init_state(reproducible=False)
            self.evolve_until_percolated()
            num += int(self.has_percolated)

        frac = num / repeats
        stderr = np.sqrt(frac * (1 - frac) / (repeats - 1))
        
        if print_result:
            print(f"{num} out of {repeats} simulations percolated: f = {frac}")
            print(f"Estimate of the standard error on f: sigma_f = {stderr:.2g}")
        else:
            return frac, stderr

    def loop_estimate_percolation_prob(self, repeats=25, loop=20):
        """Loops over estimate_percolation_prob, just to hide some confusing code from
        students."""
        print("f\tsigma_f")
        for _ in range(loop):
            f, sigma_f = self.estimate_percolation_prob(repeats, print_result=False)
            print(f"{f:.3f}\t{sigma_f:.3f}")
        print("finished!")

    # ----------------------------------------------------------------------------------------
    #                                                                        | Visualisation |
    #                                                                        -----------------

    def plot_sir(self, outpath=None):
        """Plots the time evolution of the model.

        More specifically, plots the evolution of the fraction of nodes that are (a)
        susceptible, (b) live, and (c) inert. This can be seen as a 'susceptible-
        infected-removed' plot if we interpret the simulation as an epidem model.

        Inputs
        ------
        outpath: str (optional)
            If provided, specifies path to a directory in which the plot will be saved
            as 'plot.png'.
        """
        fig, ax = plt.subplots()
        ax.set_title("Time evolution of the model")
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Fraction of nodes")
        ax.plot(
            self.susceptible_time_series,
            color="blue",
            label="susceptible",
        )
        ax.plot(
            self.live_time_series,
            color="red",
            label="infected (live)",
        )
        ax.plot(
            self.inert_time_series,
            color="grey",
            label="immune (inert)",
        )

        ax.legend()
        fig.tight_layout()

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath / "sir_plot.png")
        else:
            plt.show()

    def animate(self, n_steps=-1, interval=50, dynamic_overlay=False, outpath=None):
        """Evolves the model for `n_steps` iterations and produces an animation.

        Inputs
        ------
        n_steps: int (optional)
            Number of updates. By default, equal to the square root of the number of
            nodes, plus 1.
        interval: int (optional)
            Number of millisconds delay between each update.
        dynamic_overlay: bool (optional)
            If True, updates the overlay of inert nodes as well as the live nodes.
            This is useful if you have set recovered_are_inert and care about the
            recovered nodes taking the same colour as the initial inert ones.
        outpath: str (optional)
            If provided, specifies path to a directory in which the plot will be saved
            as 'animation.gif'.
        """
        if type(n_steps) is not int:
            raise TypeError(
                "Please provide an integer for the number of steps to animate."
            )
        if n_steps < 1:
            n_steps = int(np.sqrt(self.network.n_nodes)) + 1

        # For now, set cmap based on number of links
        if self.network.n_links == 1:
            cmap = "viridis"
        elif self.network.n_links == 3:
            cmap = "seismic_r"
        else:
            cmap = "YlOrRd"

        fig, ax = plt.subplots()
        ax.set_axis_off()

        image = ax.imshow(
            self.state,
            norm=colors.Normalize(vmin=0, vmax=self.recovery_time),
            zorder=0,
            cmap=cmap,
        )
        overlay = ax.imshow(
            self.inert,
            cmap=colors.ListedColormap(["#66666600", "#666666"]),  # [transparent, grey]
            norm=colors.Normalize(vmin=0, vmax=1),
            zorder=1,
        )
        step_counter = ax.annotate(
            f"Step 0",
            xy=(0, -0.11),
            xycoords="axes fraction",
        )

        def loop_without_overlay(t):
            if t == 0:  # otherwise the animation starts a frame late in Jupyter...
                return image, step_counter
            _ = self._update()
            image.set_data(self.state)
            step_counter.set_text(f"Step {t}")
            return image, step_counter

        def loop_with_overlay(t):
            if t == 0:
                return image, overlay, step_counter
            _ = self._update()
            image.set_data(self.state)
            overlay.set_data(self.inert)
            step_counter.set_text(f"Step {t}")
            return image, overlay, step_counter

        if dynamic_overlay:
            loop = loop_with_overlay
        else:
            loop = loop_without_overlay

        ani = animation.FuncAnimation(
            fig, loop, frames=n_steps + 1, interval=interval, repeat=False, blit=True
        )

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            ani.save(outpath / "animation.gif")

        return ani
