# <a name="top"></a> The Code for Studying Problems of Federated Learning

This folder includes code for studying problems of federated learning, ~~under construction....~~ migrated to [this repo](https://github.com/wenh06/fl-sim).

The main part of this code repository is a standalone simulation framework of federated training.

**~~This folder temporarily suspends updating. Code are currently developed in a private repository and would be released public after some time.~~**

**This folder permanently suspends updating. Codes are migrated to [this repo](https://github.com/wenh06/fl-sim), and are actively updating.**

<!-- toc -->

- [Optimizers](#optimizers)
- [Regularizers](#regularizers)
- [Compression](#compression)
- [Data Processing](#data-processing)
- [Models](#models)
- [Algorithms Implemented](#algorithms-implemented)

<!-- tocstop -->

## [Optimizers](optimizers/)

The module (folder) [optimizers](optimizers/) contains optimizers for solving inner (local) optimization problems. Despite optimizers from `torch` and `torch_optimizers`, this module implements

1. `ProxSGD`
2. `FedPD_SGD`
3. `FedPD_VR`
4. `PSGD`
5. `PSVRG`
6. `pFedMe`
7. `FedProx`
8. `FedDR`

Most of the optimizers are derived from `ProxSGD`.

:point_right: [Back to TOC](#top)

## [Regularizers](regularizers/)

The module (folder) [regularizers](regularizers/) contains code for regularizers for model parameters (weights).

1. `L1Norm`
2. `L2Norm`
3. `L2NormSquared`
4. `NullRegularizer`

These regularizers are subclasses of a base class `Regularizer`, and can be obtained by passing the name of the regularizer to the function `get_regularizer`. The regularizers share common methods `eval` and `prox_eval`.

:point_right: [Back to TOC](#top)

## [Compression](compressors/)

The module (folder) [compressors](compressors/) contains code for constructing compressors.

:point_right: [Back to TOC](#top)

## [Data Processing](data_processing/)

The module (folder) [data_processing](data_processing/) contains code for data preprocessing, io, etc. The following datasets are included in this module:

1. `FedCIFAR`
2. `FedCIFAR100`
3. `FedEMNIST`
4. `FedShakespeare`
5. `FedSynthetic`
6. `FedProxFEMNIST`
7. `FedProxMNIST`

:point_right: [Back to TOC](#top)

## [Models](models/)

The module (folder) [models](models/) contains pre-defined (neural network) models, most of which are very simple:

1. `MLP`
2. `FedPDMLP`
3. `CNNMnist`
4. `CNNFEMnist`
5. `CNNFEMnist_Tiny`
6. `CNNCifar`
7. `RNN_OriginalFedAvg`
8. `RNN_StackOverFlow`
9. `ResNet18`
10. `ResNet10`

One can call the `module_size` or `module_size_` properties to check the size (in terms of number of parameters and memory consumption respectively) of the model.

:point_right: [Back to TOC](#top)

## [Algorithms Implemented](algorithms/)

to write

:point_right: [Back to TOC](#top)

## [Benchmarks](/benchmarks)

to write

:point_right: [Back to TOC](#top)

## [Boyd-admm](algorithms/boyd-admm/)

the folder [boyd-admm](algorithms/boyd-admm/) contains matlab code from the website of S. Boyd for his ADMM long paper.

:point_right: [Back to TOC](#top)
