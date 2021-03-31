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

    def _test_neighbours(self, n_rows, n_cols):
        """ Test neighbours of SquareLattice of size (n_rows, n_cols) in 4 directions. """
        reference_lattice = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        

        link_definitions = [
            {'n_links': 1, 'shift': [-1],           'axis': [1],            'name':'right'},
            {'n_links': 2, 'shift': [-1, -1],       'axis': [0, 1],         'name':'down/right'},
            {'n_links': 3, 'shift': [1, -1, -1],    'axis': [0, 0, 1],      'name':'up/down/right'},
            {'n_links': 4, 'shift': [1, -1, -1, 1], 'axis': [0, 0, 1, 1],   'name':'isotropic'},
        ]

        for link_def in link_definitions:
            shifts = link_def['shift']
            axes = link_def['axis']
            n_links = link_def['n_links']
            name = link_def['name']

            lattice = SquareLattice(n_rows, n_cols, n_links=n_links, periodic=True)

            desc = (
                f"Neighbours array does not match that expected from numpy.roll ({name}, {n_rows}x{n_cols})."
            )
            expected = np.sort(
                np.concatenate(
                    [
                        np.roll(reference_lattice, shift=shift, axis=axis) 
                        for shift, axis in zip(shifts, axes)
                    ]
                ).flatten()
            )
            got = np.sort(lattice.neighbours.flatten())
            np.testing.assert_array_equal(
                got, expected, err_msg=err_msg(desc, expected, got)
            )




class TestNeighboursFixed:
    L=6
    lattice = SquareLattice(L, n_links=4, periodic=False)
    reference_lattice=np.arange(L**2).reshape(L, L)

    def test_top_left_corner(self):
        desc = "Wrong neighbours at top left corner (fixed boundaries)."
        expected = [0, 0, 1, self.L]
        got = sorted(list(self.lattice.neighbours[0]))
        assert got == expected, err_msg(desc, expected, got)

    def test_top_right_corner(self):
        desc = "Wrong neighbours at top right corner (fixed boundaries)."
        expected = [self.L - 2, self.L - 1, self.L - 1, 2 * self.L - 1]
        got = sorted(list(self.lattice.neighbours[self.L - 1]))
        assert got == expected, err_msg(desc, expected, got)

    def test_bottom_left_corner(self):
        desc = "Wrong neighbours at bottom left corner (fixed boundaries)."
        expected = [(self.L - 2) * self.L, (self.L - 1) * self.L, (self.L - 1) * self.L, (self.L - 1) * self.L + 1]
        got = sorted(list(self.lattice.neighbours[(self.L - 1) * self.L]))
        assert got == expected, err_msg(desc, expected, got)

    def test_bottom_right_corner(self):
        desc = "Wrong neighbours at bottom right corner (fixed boundaries)."
        expected = [(self.L - 1) * self.L - 1, self.L * self.L - 2, self.L * self.L - 1, self.L * self.L - 1]
        got = sorted(list(self.lattice.neighbours[self.L * self.L - 1]))
        assert got == expected, err_msg(desc, expected, got)

    def test_left_edge(self):
        desc = "Wrong neighbours at left edge (fixed boundaries)."
        expected = np.sort(
            np.concatenate(
                [
                    self.reference_lattice[1:-1, 0],  # left -> identity
                    self.reference_lattice[1:-1, 1],  # right
                    self.reference_lattice[2:, 0],  # down
                    self.reference_lattice[:-2, 0],  # up
                ]
            )
        )
        got = np.sort(self.lattice.neighbours[self.reference_lattice[1:-1, 0]].flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_right_edge(self):
        desc = "Wrong neighbours at right edge (fixed boundaries)."
        expected = np.sort(
            np.concatenate(
                [
                    self.reference_lattice[1:-1, -2],  # left
                    self.reference_lattice[1:-1, -1],  # right -> identity
                    self.reference_lattice[2:, -1],  # down
                    self.reference_lattice[:-2, -1],  # up
                ]
            )
        )
        got = np.sort(self.lattice.neighbours[self.reference_lattice[1:-1, -1]].flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_bottom_edge(self):
        desc = "Wrong neighbours at bottom edge (fixed boundaries)."
        expected = np.sort(
            np.concatenate(
                [
                    self.reference_lattice[-1, :-2],  # left
                    self.reference_lattice[-1, 2:],  # right
                    self.reference_lattice[-1, 1:-1],  # down -> identity
                    self.reference_lattice[-2, 1:-1],  # up
                ]
            )
        )
        got = np.sort(self.lattice.neighbours[self.reference_lattice[-1, 1:-1]].flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_top_edge(self):
        desc = "Wrong neighbours at top edge (fixed boundaries)."
        expected = np.sort(
            np.concatenate(
                [
                    self.reference_lattice[0, :-2],  # left
                    self.reference_lattice[0, 2:],  # right
                    self.reference_lattice[1, 1:-1],  # down
                    self.reference_lattice[0, 1:-1],  # up -> identity
                ]
            )
        )
        got = np.sort(self.lattice.neighbours[self.reference_lattice[0, 1:-1]].flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )
