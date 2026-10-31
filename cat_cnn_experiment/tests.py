"""Numerical, orientation, support and cold-start guards."""

import unittest

import numpy as np
from scipy import sparse

from model import penalty
from src.development import identity_matrix
from src.estimator import objective_gradient


class CatCnnTests(unittest.TestCase):
    def test_cat_difference_cancels_within_cat(self):
        x = identity_matrix([{"a_cat": "a", "b_cat": "a"}], ["a"], "cat", side=False)
        np.testing.assert_array_equal(x.toarray(), [[0.]])

    def test_unseen_cat_is_explicitly_unsupported(self):
        with self.assertRaises(ValueError):
            identity_matrix([{"a_cat": "new", "b_cat": "a"}], ["a"], "cat", side=False)

    def test_display_reversal_preserves_intercept(self):
        cats = identity_matrix([{"a_cat": "a", "b_cat": "b"}, {"a_cat": "b", "b_cat": "a"}],
                               ["a", "b"], "cat", side=False)
        visual = sparse.csr_matrix([[.3, -.2], [-.3, .2]])
        x = sparse.hstack([np.ones((2,1)), cats, visual], format="csr")
        z = np.asarray(x @ np.asarray([.1, .2, -.4, .5, -.8])).ravel()
        self.assertAlmostEqual(z[0]+z[1], .2)

    def test_penalty_layout_and_gradient(self):
        p = penalty(2, 2, .01, .1)
        np.testing.assert_array_equal(p, [1e-6, .01, .01, .1, .1])
        x = np.asarray([[1.,1.,-1.,.2,.3],[1.,-1.,1.,-.1,.4]])
        y = np.asarray([1.,0.]); theta = np.asarray([.1,.2,-.1,.3,.2])
        _, gradient = objective_gradient(theta,x,y,np.ones(2),p)
        numerical=[]
        for i in range(len(theta)):
            shift=np.zeros_like(theta); shift[i]=1e-6
            numerical.append((objective_gradient(theta+shift,x,y,np.ones(2),p)[0]-
                              objective_gradient(theta-shift,x,y,np.ones(2),p)[0])/2e-6)
        np.testing.assert_allclose(gradient,numerical,rtol=0,atol=1e-8)


if __name__ == "__main__":
    unittest.main()
