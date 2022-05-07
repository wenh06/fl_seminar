# Optimizers that solve **inner (local)** optimization problems

Most (inner) optimizers are based on the [ProxSGD](base.py) optimizer, including
- [FedDROptimizer](feddr.py)
- [FedProxOptimizer](fedprox.py)
- [pFedMeOptimizer](pfedme.py)

Other (inner) optimizers include
- [FedPD Optimizers](fedpd.py) (Not checked yet)
