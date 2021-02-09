import numpy as np

MIN_LENGTH = 2
MAX_LENGTH = 1000


class SquareLattice:
    """Class containing attributes and methods related to a Square lattice in which nodes
    are coupled to their nearest neighbours along the Cartesian axes (i.e. left/right/up/
    down, but not along the diagonals)."""

    def __init__(
        self, dimensions: tuple, directed: str = "isotropic", periodic: bool = False
    ):
        self.directed = directed
        self.periodic = periodic
        self.dimensions = dimensions  # last since it involves cache_neighbours

    # ----------------------------------------------------------------------------------------
    #                                                                     | Data descriptors |
    #                                                                     --------------------

    @property
    def dimensions(self):
        """Number of nodes along each axis of the lattice."""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, new_value):
        """Setter for dimensions. If new_value is an int, set dimensions to be a 2-tuple
        with both elements equal. Raises TypeError for inputs that are not (int, tuple,
        float, list), and ValueError for those that are the wrong length, or are outside
        the range [MIN_LENGTH, MAX_LENGTH]."""
        if type(new_value) in (int, float):
            new_value = (int(new_value), int(new_value))
        elif type(new_value) in (tuple, list):
            if len(new_value) == 1:
                new_value = tuple([int(new_value[0]), int(new_value[0])])
            elif len(new_value) == 2:
                new_value = tuple([int(i) for i in new_value])
            else:
                raise ValueError(
                    "Please provide lattice dimensions as a single integer or a 2-tuple of integers"
                )
        else:
            raise TypeError(
                "Please provide lattice dimensions as a single integer or a 2-tuple of integers"
            )
        for new_dim in new_value:
            if new_dim < MIN_LENGTH or new_dim > MAX_LENGTH:
                raise ValueError(
                    f"Please enter lattice dimensions beween {MIN_LENGTH} and {MAX_LENGTH}"
                )
        self._dimensions = new_value
        self._cache_neighbours()  # must update neighbour array if dimensions modified!

    @property
    def directed(self):
        """Tuple with two elements indicating whether the connections are directed along
        the corresponding axes (True) or whether they are bi-directional (False). In the
        case of directed connections, nodes will have fewer than 4 neighbours. This us
        Used to generate anisotropy in percolation simulations."""
        return self._directed

    @directed.setter
    def directed(self, new_string):
        """Setter for directed. Raises TypeError if input is not a string and ValueError
        if it is not one of 'isotropic', 'right', 'down', or 'both'."""
        opts = ("isotropic", "right", "down", "both")
        msg = f"Please choose from: {', '.join([opt for opt in opts])}."
        if type(new_string) is not str:
            raise TypeError(msg)
        if new_string not in opts:
            raise ValueError(msg)
        self._directed = new_string

    @property
    def periodic(self):
        """Flag indicating whether the lattice has boundaries that wrap around in a
        toroidal topology."""
        return self._periodic

    @periodic.setter
    def periodic(self, new_flag):
        """Setter for periodic. Raises TypeError if input is not a bool."""
        if type(new_flag) is not bool:
            raise TypeError("Please enter True/False for periodic.")
        self._periodic = new_flag

    # ----------------------------------------------------------------------------------------
    #                                                                 | Read-only properties |
    #                                                                 ------------------------

    @property
    def n_nodes(self):
        """Number of nodes on the lattice."""
        return self.dimensions[0] * self.dimensions[1]

    @property
    def n_neighbours(self):
        """Number of nearest neighbours that each node is in 'contact' with, meaning the
        virus can be transmitted."""
        return len(self.connections)

    @property
    def neighbours(self):
        """Array containing the coordinates of the nearest neighbours for each node,
        in the lexicographic representation. The first dimension corresponds to lattice
        nodes in lexicographic representation. The second dimension of the array represents
        the direction (right, left, up, down).
        """
        return self._neighbours

    @property
    def connections(self):
        """Convenience property, expressing the connections for any given node as a
        tuple of tuples ((shift_1, axis_1), ...).

        Note that the conventions defined by numpy.roll are as follows:

            shift   axis    result                  returns neighbours
            ----------------------------------------------------------
            1       0       rows shifted down       above
            -1      0       rows shifted up         below
            1       1       cols shifted right      to left
            -1      1       cols shifted left       to right

        """
        if self.directed == "isotropic":
            return ((1, 0), (-1, 0), (1, 1), (-1, 1))
        elif self.directed == "right":
            return ((1, 0), (-1, 0), (-1, 1))
        elif self.directed == "down":
            return ((-1, 0), (1, 1), (-1, 1))
        elif self.directed == "both":
            return ((-1, 0), (-1, 1))

    # ----------------------------------------------------------------------------------------
    #                                                                    | Protected methods |
    #                                                                    ---------------------

    def _boundary_nodes_mask(self):
        """Convenience method that returns a tuple of masks which can be used to access
        the strips of nodes at each of the four boundaries."""
        lexi_like_cart = np.arange(self.n_nodes).reshape(*self.dimensions)
        mask_dict = {
            (1, 0): lexi_like_cart // self.dimensions[0] == 0,  # top row
            (-1, 0): lexi_like_cart // self.dimensions[0]
            == self.dimensions[0] - 1,  # bottom row
            (1, 1): lexi_like_cart % self.dimensions[1] == 0,  # left column
            (-1, 1): lexi_like_cart % self.dimensions[1]
            == self.dimensions[1] - 1,  # right column
        }
        masks = [mask_dict[conn] for conn in self.connections]
        return tuple(masks)

    def _cache_neighbours(self):
        """Cache the array of neighbours to avoid repeated calculations.

        Notes:
        ------
        In the case of non-periodic boundaries, the neighbours that would otherwise wrap
        around the lattice are set to be the indices of the nodes themselves (i.e. they act
        as their own neighbour). This doesn't lead to strange behaviour since, after taking
        the list of neighbours of infected nodes, we then discard all those that are already
        infected.
        """
        lexi_like_cart = np.arange(self.n_nodes).reshape(*self.dimensions)
        neighbours = np.zeros((self.n_neighbours, self.n_nodes), dtype=int)

        for i, (shift, axis) in enumerate(self.connections):
            # Roll the cartesian array and flatten for 1d array of neighbours
            neighbours[i] = np.roll(lexi_like_cart, shift, axis=axis).flatten()

        if not self.periodic:
            for i, mask in enumerate(self._boundary_nodes_mask()):
                np.putmask(
                    neighbours[i],
                    mask=mask.flatten(),  # pull out one row/col
                    values=np.arange(self.n_nodes),  # equivalent to zero shift
                )

        self._neighbours = neighbours.transpose()  # shape (n_nodes, n_neighbours)

    # ----------------------------------------------------------------------------------------
    #                                                                       | Public methods |
    #                                                                       ------------------

    def lexi_to_cart(self, state_lexi):
        """Convert a state in 1d lexicographic representation to 2d Cartesian representation.

        Inputs
        ------
        state_lexi: numpy.ndarray
            One dimensional array containing the state in lexicographic representation.
        """
        return state_lexi.reshape(*self.dimensions)
