import unittest
from moore_penrose import *

class MoorePenroseTestCase(unittest.TestCase):
    """Tests for `moore_penrose.py`."""

    def driver(self, n, m):
        """Driver function for testing Moore-Penrose pseudoinverse"""
        M = produceMatrix(n, m)
        A_ = moorePenrose(M)
        A_np = np.linalg.pinv(M)
        self.assertTrue(checkMatrix(A_, A_np))

    def test_moore_penrose_square(self):
        """Is the Moore-Penrose pseudoinverse of a matrix equal to the numpy pseudoinverse?"""
        n = 25
        m = 25
        self.driver(n, m)

    def test_moore_penrose_row(self):
        n = 10
        m = 5
        self.driver(n, m)

    def test_moore_penrose_column(self):
        n = 5
        m = 10
        self.driver(n, m)

if __name__ == '__main__':
    unittest.main()