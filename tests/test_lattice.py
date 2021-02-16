import numpy as np
from p1b_percolation.lattice import SquareLattice

L = 6
REFERENCE_LATTICE = np.arange(L ** 2).reshape(L, L)


def err_msg(desc, expected, got):
    return f"{desc}\nExpected {expected}, but got {got}."


class TestNeighbours:
    def test_four_links(self):
        lattice = SquareLattice(L, n_links=4, periodic=True)
        desc = (
            "Neighbours array does not match that expected from numpy.roll (isotropic)."
        )
        expected = np.sort(
            np.concatenate(
                [
                    np.roll(REFERENCE_LATTICE, shift=1, axis=0),  # up
                    np.roll(REFERENCE_LATTICE, shift=-1, axis=0),  # down
                    np.roll(REFERENCE_LATTICE, shift=1, axis=1),  # left
                    np.roll(REFERENCE_LATTICE, shift=-1, axis=1),  # right
                ]
            ).flatten()
        )
        got = np.sort(lattice.neighbours.flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_three_links(self):
        lattice = SquareLattice(L, n_links=3, periodic=True)
        desc = "Neighbours array does not match that expected from numpy.roll (down/up/right-directed)."
        expected = np.sort(
            np.concatenate(
                [
                    np.roll(REFERENCE_LATTICE, shift=1, axis=0),  # up
                    np.roll(REFERENCE_LATTICE, shift=-1, axis=0),  # down
                    np.roll(REFERENCE_LATTICE, shift=-1, axis=1),  # right
                ]
            ).flatten()
        )
        got = np.sort(lattice.neighbours.flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_two_links(self):
        lattice = SquareLattice(L, n_links=2, periodic=True)
        desc = "Neighbours array does not match that expected from numpy.roll (down/right-directed)."
        expected = np.sort(
            np.concatenate(
                [
                    np.roll(REFERENCE_LATTICE, shift=-1, axis=0),  # down
                    np.roll(REFERENCE_LATTICE, shift=-1, axis=1),  # right
                ]
            ).flatten()
        )
        got = np.sort(lattice.neighbours.flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_one_connection(self):
        lattice = SquareLattice(L, n_links=1, periodic=True)
        desc = "Neighbours array does not match that expected from numpy.roll (right-directed)."
        expected = np.sort(
            np.roll(REFERENCE_LATTICE, shift=-1, axis=1).flatten()
        )  # right
        got = np.sort(lattice.neighbours.flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )


class TestFixedBoundaries:
    lattice = SquareLattice(L, n_links=4, periodic=False)

    def test_top_left_corner(self):
        desc = "Wrong neighbours at top left corner (fixed boundaries)."
        expected = [0, 0, 1, L]
        got = sorted(list(self.lattice.neighbours[0]))
        assert got == expected, msg(desc, expected, got)

    def test_top_right_corner(self):
        desc = "Wrong neighbours at top right corner (fixed boundaries)."
        expected = [L - 2, L - 1, L - 1, 2 * L - 1]
        got = sorted(list(self.lattice.neighbours[L - 1]))
        assert got == expected, msg(desc, expected, got)

    def test_bottom_left_corner(self):
        desc = "Wrong neighbours at bottom left corner (fixed boundaries)."
        expected = [(L - 2) * L, (L - 1) * L, (L - 1) * L, (L - 1) * L + 1]
        got = sorted(list(self.lattice.neighbours[(L - 1) * L]))
        assert got == expected, msg(desc, expected, got)

    def test_bottom_right_corner(self):
        desc = "Wrong neighbours at bottom right corner (fixed boundaries)."
        expected = [(L - 1) * L - 1, L * L - 2, L * L - 1, L * L - 1]
        got = sorted(list(self.lattice.neighbours[L * L - 1]))
        assert got == expected, msg(desc, expected, got)

    def test_left_edge(self):
        desc = "Wrong neighbours at left edge (fixed boundaries)."
        expected = np.sort(
            np.concatenate(
                [
                    REFERENCE_LATTICE[1:-1, 0],  # left -> identity
                    REFERENCE_LATTICE[1:-1, 1],  # right
                    REFERENCE_LATTICE[2:, 0],  # down
                    REFERENCE_LATTICE[:-2, 0],  # up
                ]
            )
        )
        got = np.sort(self.lattice.neighbours[REFERENCE_LATTICE[1:-1, 0]].flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_right_edge(self):
        desc = "Wrong neighbours at right edge (fixed boundaries)."
        expected = np.sort(
            np.concatenate(
                [
                    REFERENCE_LATTICE[1:-1, -2],  # left
                    REFERENCE_LATTICE[1:-1, -1],  # right -> identity
                    REFERENCE_LATTICE[2:, -1],  # down
                    REFERENCE_LATTICE[:-2, -1],  # up
                ]
            )
        )
        got = np.sort(self.lattice.neighbours[REFERENCE_LATTICE[1:-1, -1]].flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_bottom_edge(self):
        desc = "Wrong neighbours at bottom edge (fixed boundaries)."
        expected = np.sort(
            np.concatenate(
                [
                    REFERENCE_LATTICE[-1, :-2],  # left
                    REFERENCE_LATTICE[-1, 2:],  # right
                    REFERENCE_LATTICE[-1, 1:-1],  # down -> identity
                    REFERENCE_LATTICE[-2, 1:-1],  # up
                ]
            )
        )
        got = np.sort(self.lattice.neighbours[REFERENCE_LATTICE[-1, 1:-1]].flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )

    def test_top_edge(self):
        desc = "Wrong neighbours at top edge (fixed boundaries)."
        expected = np.sort(
            np.concatenate(
                [
                    REFERENCE_LATTICE[0, :-2],  # left
                    REFERENCE_LATTICE[0, 2:],  # right
                    REFERENCE_LATTICE[1, 1:-1],  # down
                    REFERENCE_LATTICE[0, 1:-1],  # up -> identity
                ]
            )
        )
        got = np.sort(self.lattice.neighbours[REFERENCE_LATTICE[0, 1:-1]].flatten())
        np.testing.assert_array_equal(
            got, expected, err_msg=err_msg(desc, expected, got)
        )
