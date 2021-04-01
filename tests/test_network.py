from percolation.networks import BooleanNetwork, BooleanEdge, WeightedNetwork, WeightedEdge
from typing import List
import numpy as np

class TestBooleanNetwork:
    def test_creation_single_element(self):
        edges: List[BooleanEdge] = [(0,0)]
        network = BooleanNetwork(2, edges)
        assert network.matrix[0, 0] == True
        assert network.matrix[0, 1] == False

    def test_symmetry(self):
        edges: List[BooleanEdge] = [(0,1)]
        network = BooleanNetwork(2, edges, directed=False)
        assert network.matrix[0, 1] == True
        assert network.matrix[1, 0] == True
        assert network.matrix[0, 0] == False
        assert network.matrix[1, 1] == False

    def test_directed(self):
        edges: List[BooleanEdge] = [(0,1)]
        network = BooleanNetwork(2, edges, directed=True)
        assert network.matrix[0, 1] == True
        assert network.matrix[1, 0] == False
        assert network.matrix[0, 0] == False
        assert network.matrix[1, 1] == False

    def test_dot_product_propagation(self):
        edges: List[BooleanEdge] = [(0,1), (0,2), (2,3)]
        network = BooleanNetwork(4, edges, directed=False)
        state = np.array([True, False, False, False], dtype=np.bool8)
        propagation = state * network.matrix
        expected = np.array([False, True, True, False], dtype=np.bool8)
        np.testing.assert_array_equal(propagation, expected)