import numpy as np
from scipy.sparse import csc_matrix
from typing import List, Tuple

BooleanEdge = Tuple[int, int]
WeightedEdge = Tuple[float, Tuple[int, int]]


class BooleanNetwork:
    """ 
        Class describing a network with any shape, which can be represented as a graph.
        Connections betwen nodes are stored in an sparse adjecency matrix.
    """
    def __init__(self, size: int, edges: List[BooleanEdge], directed=True):
        self.create_adjecency_matrix(size, edges, directed)
    
    def create_adjecency_matrix(self, size: int, edges: List[BooleanEdge], directed=True):
        self.shape = (size, )

        data = len(edges) * [True]
        rows, columns = zip(*edges)
        
        self.directed = directed
        if not self.directed:
            data *= 2
            columns_copy = columns[:]
            columns += rows
            rows += columns_copy

        self._matrix = csc_matrix((data, (rows, columns)), shape=2*self.shape, dtype=np.bool8)

    @property
    def matrix(self):
        return self._matrix