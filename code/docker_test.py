"""
"""

from algorithms.pfedme import test_pfedme  # noqa: F401
from algorithms.fedopt import test_fedopt  # noqa: F401

from algorithms._experiments.fedprox import test_fedprox  # noqa: F401
from algorithms._experiments.fedpd import test_fedpd  # noqa: F401

# from algorithms._experiments.feddr import test_feddr  # noqa: F401
# from algorithms._experiments.feddr_new import test_feddr_new  # noqa: F401


if __name__ == "__main__":
    test_pfedme()
    # test_fedprox()
    # test_feddr()
