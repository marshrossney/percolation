import numpy as np
from percolation.lattice import SquareLattice


def err_msg(desc, expected, got):
    return f"{desc}\nExpected {expected}, but got {got}."


class TestNeighboursPeriodic:
    def test_square(self):
        self._test_neighbours(6, 6)

    def test_rectangle_6x5(self):
        self._test_neighbours(6, 5)

    def test_rectangle_5x6(self):
        self._test_neighbours(5, 6)

    @staticmethod
    def lex_to_cart(lex, n_rows, n_cols):
        return np.unravel_index(lex, (n_rows, n_cols))

    @staticmethod
    def cart_to_lex(row, col, n_rows, n_cols):
        return np.ravel_multi_index((row,col), (n_rows, n_cols))


    def list_neighbours_of_node(self, lex, n_rows, n_cols, shifts, axes):
        neighbours = list()
        row, col = self.lex_to_cart(lex, n_rows, n_cols)
        for shift, axis in zip(shifts, axes):
            if axis == 1:
                row_new = (row - shift) % n_rows
                col_new = col
            else:
                row_new = row
                col_new = (col - shift) % n_cols
            neighbour = self.cart_to_lex(row_new, col_new, n_rows, n_cols)
            neighbours.append(neighbour)

        return neighbours


    def _test_neighbours(self, n_rows, n_cols):
        """ Test neighbours of SquareLattice of size (n_rows, n_cols) with 4 posible link types """

        link_definitions = [
            {'n_links': 1, 'shift': [-1],           'axis': [1],            'name':'down'},
            {'n_links': 2, 'shift': [-1, -1],       'axis': [0, 1],         'name':'down/right'},
            {'n_links': 3, 'shift': [1, -1, -1],    'axis': [0, 0, 1],      'name':'left/right/down'},
            {'n_links': 4, 'shift': [1, -1, -1, 1], 'axis': [0, 0, 1, 1],   'name':'isotropic'},
        ]

        for link_def in link_definitions:
            shifts = link_def['shift']
            axes = link_def['axis']
            n_links = link_def['n_links']
            name = link_def['name']
            n_nodes = n_rows * n_cols

            lattice = SquareLattice(n_rows, n_cols, n_links=n_links, periodic=True)

            desc = (
                f"Neighbours array does not match that expected from numpy.roll ({name}, {n_rows}x{n_cols})."
            )

            expected = np.zeros((n_nodes, n_nodes), dtype=np.bool8)
            for i in range(n_nodes):
                for j in self.list_neighbours_of_node(i, n_rows, n_cols, shifts, axes):
                    expected[i, j] = True

            got = lattice.matrix.toarray()

            np.testing.assert_array_equal(
                got, expected, err_msg=err_msg(desc, expected, got)
            )


class TestNeighboursBounded:
    def test_square(self):
        self._test_neighbours(6, 6)

    def test_rectangle_6x5(self):
        self._test_neighbours(6, 5)

    def test_rectangle_5x6(self):
        self._test_neighbours(5, 6)

    @staticmethod
    def lex_to_cart(lex, n_rows, n_cols):
        return np.unravel_index(lex, (n_rows, n_cols))

    @staticmethod
    def cart_to_lex(row, col, n_rows, n_cols):
        return np.ravel_multi_index((row,col), (n_rows, n_cols))


    def list_neighbours_of_node(self, lex, n_rows, n_cols, shifts, axes):
        neighbours = list()
        row, col = self.lex_to_cart(lex, n_rows, n_cols)
        for shift, axis in zip(shifts, axes):
            if axis == 1:
                row_new = row - shift
                col_new = col
                if not 0 <= row_new < n_rows:
                    continue
            else:
                row_new = row
                col_new = col - shift
                if not 0 <= col_new < n_cols:
                    continue
            neighbour = self.cart_to_lex(row_new, col_new, n_rows, n_cols)
            neighbours.append(neighbour)

        return neighbours


    def _test_neighbours(self, n_rows, n_cols):
        """ Test neighbours of SquareLattice of size (n_rows, n_cols) with 4 posible link types """

        link_definitions = [
            {'n_links': 1, 'shift': [-1],           'axis': [1],            'name':'down'},
            {'n_links': 2, 'shift': [-1, -1],       'axis': [0, 1],         'name':'down/right'},
            {'n_links': 3, 'shift': [1, -1, -1],    'axis': [0, 0, 1],      'name':'left/right/down'},
            {'n_links': 4, 'shift': [1, -1, -1, 1], 'axis': [0, 0, 1, 1],   'name':'isotropic'},
        ]

        for link_def in link_definitions:
            shifts = link_def['shift']
            axes = link_def['axis']
            n_links = link_def['n_links']
            name = link_def['name']
            n_nodes = n_rows * n_cols

            lattice = SquareLattice(n_rows, n_cols, n_links=n_links, periodic=False)

            desc = (
                f"Neighbours array does not match that expected from numpy.roll ({name}, {n_rows}x{n_cols})."
            )

            expected = np.zeros((n_nodes, n_nodes), dtype=np.bool8)
            for i in range(n_nodes):
                for j in self.list_neighbours_of_node(i, n_rows, n_cols, shifts, axes):
                    expected[i, j] = True

            got = lattice.matrix.toarray()

            np.testing.assert_array_equal(
                got, expected, err_msg=err_msg(desc, expected, got)
            )