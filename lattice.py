import numpy as np

MIN_LATTICE_LENGTH = 2
MAX_LATTICE_LENGTH = 100


class SquareLattice:
    """"""

    def __init__(self, length: int):
        self.length = length
        self._cache_neighbours()

    @property
    def length(self):
        """Number of nodes along one dimension of the lattice."""
        return self._length

    @length.setter
    def length(self, new_value):
        """Setter for length. Raises ValueError for inputs that are outside the range
        [MIN_LATTICE_LENGTH, MAX_LATTICE_LENGTH]. Will convert float inputs to integers."""
        new_value = int(new_value)
        if new_value < MIN_LATTICE_LENGTH or new_value > MAX_LATTICE_LENGTH:
            raise ValueError(
                f"Please enter an integer lattice length beween {MIN_LATTICE_LENGTH} and {MAX_LATTICE_LENGTH}"
            )
        self._length = new_value

    @property
    def n_nodes(self):
        """Number of nodes on the lattice."""
        return self.length ** 2

    @property
    def n_neighbours(self):
        """Number of nearest neighbours that each node is in 'contact' with, meaning the
        virus can be transmitted."""
        return 4

    @property
    def neighbours(self):
        """Array containing the coordinates of the nearest neighbours for each node,
        in the lexicographic representation. The first dimension corresponds to lattice
        nodes in lexicographic representation. The second dimension of the array represents
        the direction (right, left, up, down). 
        """
        return self._neighbours

    def _cache_neighbours(self):
        """Cache the array of neighbours to avoid repeated calculations."""
        lexi_like_cart = np.arange(self.n_nodes).reshape(self.length, self.length)
        neighbours = np.zeros((self.n_neighbours, self.n_nodes), dtype=int)

        for i, shift, axis in zip(
            range(self.n_neighbours),  # number of neighbour arrays
            (1, -1, 1, -1),  # positive or negative shifts
            (0, 0, 1, 1),  # axes for shifts
        ):
            # Roll the cartesian array and flatten for 1d array of neighbours
            neighbours[i] = np.roll(lexi_like_cart, shift, axis=axis).flatten()

        self._neighbours = neighbours.transpose()  # shape (n_nodes, 4)
        return

    def lexi_to_cart(self, state_lexi):
        """Convert a state in 1d lexicographic representation to 2d Cartesian representation."""
        return state_lexi.reshape(self.length, self.length)

