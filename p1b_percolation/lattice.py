import numpy as np

MIN_LENGTH = 2
MAX_LENGTH = 1000


class SquareLattice:
    """Class containing attributes and methods related to a Square lattice in which nodes
    are coupled to their nearest neighbours along the Cartesian axes (i.e. left/right/up/
    down, but not along the diagonals)."""

    def __init__(
        self, dimensions: tuple = (25, 25), n_links: int = 1, periodic: bool = False
    ):
        self.dimensions = dimensions
        self.n_links = n_links
        self.periodic = periodic

        self._cache_properties()

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

        if hasattr(self, "_neighbours"):
            self._cache_properties()  # must update neighbour lists and boundary masks

    @property
    def n_links(self):
        """Number of links per node. The directions of the links are not randomised,
        so for n_links < 4 the lattice becomes anisotropic. Visually, this looks like
        the following:

            n_links             4           3           2           1

                                ^           ^
            directions      < - | - >       | - >       | - >       - >
                                v           V           V
        """
        return self._n_links

    @n_links.setter
    def n_links(self, new_value):
        """Setter for n_links. Raises TypeError if not int and ValueError if not between
        1 and 4."""
        if type(new_value) is not int:
            raise TypeError("Please provide an integer for the number of links.")
        if new_value < 1 or new_value > 4:
            raise ValueError("Please provide a number of links between 1 and 4")
        self._n_links = new_value

        if hasattr(self, "_neighbours"):
            self._cache_properties()

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
    def neighbours(self):
        """Array containing the coordinates of the nearest neighbours for each node,
        in the lexicographic representation. The first dimension corresponds to lattice
        nodes in lexicographic representation. The second dimension of the array represents
        the direction (right, left, up, down).
        """
        return self._neighbours

    @property
    def links(self):
        """Convenience property, expressing the links for any given node as a
        tuple of tuples ((shift_1, axis_1), ...).

        Note that the conventions defined by numpy.roll are as follows:

            shift   axis    result                  returns neighbours
            ----------------------------------------------------------
            1       0       rows shifted down       above
            -1      0       rows shifted up         below
            1       1       cols shifted right      to left
            -1      1       cols shifted left       to right

        """
        if self.n_links == 4:
            return ((1, 0), (-1, 0), (1, 1), (-1, 1))
        elif self.n_links == 3:  # exclude left links
            return ((1, 0), (-1, 0), (-1, 1))
        elif self.n_links == 2:  # exclude left and up links
            return ((-1, 0), (-1, 1))
        else:  # exclude all but right links
            return ((-1, 1),)

    @property
    def far_boundary_mask(self):
        """A boolean mask which selects the nodes at the 'far' boundary."""
        if self.n_links == 4:
            return self.get_boundary_mask(key="all")
        else:
            return self.get_boundary_mask(key="right")

    # ----------------------------------------------------------------------------------------
    #                                                                    | Protected methods |
    #                                                                    ---------------------

    def _cache_properties(self):
        """Cache boundary masks and neighbours in correct order."""
        self._cache_boundary_masks()
        self._cache_neighbours()

    def _cache_boundary_masks(self):
        """Cache the masks that are used to select boundary nodes. The four boundaries
        are saved as a dict which may be accessed using either human-readible strings
        top/bottom/left/right or the (shift, dim) tuples returned by self.links.
        Also caches the 'far' boundary mask.
        """
        lexi_like_cart = np.arange(self.n_nodes).reshape(*self.dimensions)

        top_row = (lexi_like_cart // self.dimensions[0] == 0).flatten()
        bottom_row = (
            lexi_like_cart // self.dimensions[0] == self.dimensions[0] - 1
        ).flatten()
        left_col = (lexi_like_cart % self.dimensions[1] == 0).flatten()
        right_col = (
            lexi_like_cart % self.dimensions[1] == self.dimensions[1] - 1
        ).flatten()
        all_boundaries = np.logical_or.reduce(
            (top_row, bottom_row, left_col, right_col)
        )

        self._boundary_masks = {
            # Access using string
            "top": top_row,
            "bottom": bottom_row,
            "left": left_col,
            "right": right_col,
            "all": all_boundaries,
            # Access using (shift, dim)
            (1, 0): top_row,
            (-1, 0): bottom_row,
            (1, 1): left_col,
            (-1, 1): right_col,
        }

    def _cache_neighbours(self):
        """Caches the array of neighbours to avoid repeated calculations.

        Notes:
        ------
        In the case of non-periodic boundaries, the neighbours that would otherwise wrap
        around the lattice are set to be the indices of the nodes themselves (i.e. they act
        as their own neighbour). This doesn't lead to strange behaviour since, after taking
        the list of neighbours of infected nodes, we then discard all those that are already
        infected.
        """
        lexi_like_cart = np.arange(self.n_nodes).reshape(*self.dimensions)
        neighbours = np.zeros((self.n_links, self.n_nodes), dtype=int)

        for i, (shift, axis) in enumerate(self.links):
            # Roll the cartesian array and flatten for 1d array of neighbours
            neighbours[i] = np.roll(lexi_like_cart, shift, axis=axis).flatten()

            if not self.periodic:
                np.putmask(
                    neighbours[i],
                    mask=self.get_boundary_mask(key=(shift, axis)),
                    values=np.arange(self.n_nodes),  # equivalent to zero shift
                )

        self._neighbours = neighbours.transpose()  # shape (n_nodes, n_links)

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

    def get_boundary_mask(self, key="all"):
        """Convenience method that returns a mask that selects the nodes at one or
        all of the boundaries.

        Inputs
        ------
        key: str (optional)
            Key which selects which boundary's mask to return. Options are
            'left', 'right', 'top', 'bottom', 'all'.
        """
        return self._boundary_masks[key]

    def get_nucleus_mask(self, nucleus_size=1):
        """Returns a 1d boolean mask which selects nodes representing a nucleus of
        infections at day 0. This can be either

            * isotropic case -> a nucleus_size^2 area in the center of the lattice
            * anisotropic case -> a line at the left boundary

        Inputs
        ------
        nucleus_size: int (optional)
            Side length of the nucleus, in nodes.
        """
        if self.n_links == 4:
            left_edge = self.dimensions[0] // 2 - nucleus_size // 2
            top_edge = self.dimensions[1] // 2 - nucleus_size // 2
            mask = np.full(self.dimensions, False)
            mask[
                slice(left_edge, left_edge + nucleus_size),
                slice(top_edge, top_edge + nucleus_size),
            ] = True
            return mask.flatten()
        else:
            return self.get_boundary_mask(key="left")
