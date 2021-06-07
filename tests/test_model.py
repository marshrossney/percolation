from percolation.lattice import SquareLattice
from percolation.model import PercolationModel
from typing import List
import numpy as np

class TestModel:
    def test_simple_percolation(self):
        network = SquareLattice(5)
        perc = PercolationModel(network, 0.2)
        perc.evolve(5)