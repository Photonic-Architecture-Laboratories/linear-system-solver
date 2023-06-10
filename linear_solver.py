"""Wrappers for linear solver packages to handle factorizing and solving linear systems."""

import time
from typing import Hashable, Dict

import numpy as np
import scipy as sp
import scipy.sparse
from pyMKL import pardisoSolver


# This is an arbitrary limit on the maximum number of factorizations we will store before clearing memory.
_MAX_FACTORIZATIONS: int = 20


class MultipleSystemPardisoSolver(pardisoSolver):
    """A specialized subclass of the pardisoSolver from pyMKL for supporting factorizations of multiple systems.
    Note that if the matrix sparsity pattern or size changes, then `clear()` MUST be called before trying to solve with
    the new matrix.
    Args:
        verbose: Whether or not to show verbose output.  Defaults to True.
        maxfct: The maximum number of factorizations to store before resetting and clearing all memory.
    """

    def __init__(self, verbose=True, maxfct: int = _MAX_FACTORIZATIONS):
        self.maxfct = maxfct
        self._cache: Dict[Hashable, int] = {}
        self._next_mnum = 1
        self._verbose = verbose
        self._must_initialize_super = True

    @staticmethod
    def _matrix_to_key(matrix: sp.sparse.csr_matrix) -> Hashable:
        # We implicitly assume here that all matrices with the same nonzero entries have the same sparsity structure!
        return matrix.data.tobytes()

    def clear(self):
        """Clear the memory for all matrices and reset the cache."""
        super().clear()
        self._next_mnum = 1
        self._cache = {}
        self._must_initialize_super = True

    def _initialize_if_needed(self, matrix: sp.sparse.csr_matrix):
        if not self._must_initialize_super:
            return
        old = self.maxfct
        super().__init__(matrix, mtype=13)  # Set to 13 (complex unsymmetric), which is correct for SC-PML.
        self.maxfct = old

        self._cache[self._matrix_to_key(matrix)] = self._next_mnum

        if self._verbose:
            print("Performing a brand-new symbolic factorization...")
            start = time.time()
        self.run_pardiso(phase=11)
        if self._verbose:
            print("(took %3.3f seconds)" % (time.time() - start))

        self._must_initialize_super = False

    def set_matrix(self, matrix: sp.sparse.csr_matrix):
        """Set the matrix to `matrix`, which will perform a new factorization if `matrix` has not been seen before."""
        key = self._matrix_to_key(matrix)
        if key in self._cache:
            self.mnum = self._cache[key]
            return

        # Otherwise, this is a brand new matrix.
        if self._next_mnum > self.maxfct:
            if self._verbose:
                print("Clearing all factorizations (reached max limit)")
            self.clear()

        self._initialize_if_needed(matrix)
        self.mnum = self._next_mnum
        self._cache[key] = self.mnum
        self._next_mnum += 1
        self._set_pardiso_matrix_data(matrix)

    def solve(self, matrix: sp.sparse.csr_matrix, rhs: np.ndarray, transpose: bool = False) -> np.ndarray:
        """Return `matrix` inverse times `rhs`."""
        self.set_matrix(matrix)
        if transpose:
            self.iparm[11] = 2
        else:
            self.iparm[11] = 0
        if self._verbose:
            print("Performing a solve...", transpose)
            start = time.time()
        out = super().solve(rhs).reshape(rhs.shape)
        if self._verbose:
            print("(took %3.3f seconds)" % (time.time() - start))
        return out

    def _set_pardiso_matrix_data(self, matrix: sp.sparse.csr_matrix):
        A = matrix
        # If A is symmetric, store only the upper triangular portion
        if self.mtype in [2, -2, 4, -4, 6]:
            A = sp.sparse.triu(A, format="csr")
        elif self.mtype in [11, 13]:
            A = A.tocsr()

        if not A.has_sorted_indices:
            A.sort_indices()

        self.a = A.data
        self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
        if self._verbose:
            print("Performing a brand-new numerical factorization...")
            start = time.time()
        self.run_pardiso(phase=22)
        if self._verbose:
            print("(took %3.3f seconds)" % (time.time() - start))

multiple_system_solver = MultipleSystemPardisoSolver(verbose=False)
