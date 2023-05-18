# fl_seminar

![docker-ci](https://github.com/wenh06/fl_seminar/actions/workflows/docker-image.yml/badge.svg)
![format_check](https://github.com/wenh06/fl_seminar/actions/workflows/check-formatting.yml/badge.svg)

federated seminar held at BUAA

<!-- toc -->

- [fl_seminar](#fl_seminar)
  - [Time](#time)
  - [Venue](#venue)
  - [Programme (planned)](#programme-planned)
  - [Compilation](#compilation)
  - [Code](#code)
- [More resources](#more-resources)

<!-- tocstop -->

## Time

Each Thursday 20:00, excluding public holidays

## Venue

Usually in E402

## Programme (planned)

to turn into a table

1. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers ([PDF](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)) chapter 7
    - time: 2021-04-29 Thursday 20:00
    - venue: E402
    - speaker: WEN Hao
    - [notes](notes/talk1-boyd-chap7.tex)
2. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers ([PDF](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)) chapter 8
    - time: 2021-05-12 Thursday 20:00
    - venue: E502
    - speaker: WEN Hao
    - [notes](notes/talk2-boyd-chap8.tex)
3. Personalization problems in federated learning ([main resource](https://arxiv.org/pdf/1703.03400))
    - time: 2021-05-24 Thursday 20:00
    - venue: E402
    - speaker: WEN Hao
    - [slides](slides/talk3-personalization.tex)
4. Personalization problems in federated learning continued ([main resource1](https://arxiv.org/pdf/2002.05516), [main resource 2](https://arxiv.org/pdf/2006.08848))
    - time: 2021-06-10 Thursday 20:00
    - venue: E402
    - speaker: WEN Hao
    - [slides](slides/talk4-personalization-2.tex)
5. GADMM and CQ-GADMM ([main resource1](https://arxiv.org/abs/1909.00047), [main resource2](https://arxiv.org/abs/2009.06459), [main resource3](FLOW/CQ-GGADMM%20-%20FLOW.pdf))
    - time: 2021-06-24 Thursday 20:00
    - venue: E402
    - speaker: WEN Hao
    - [slides](slides/talk5-gadmm.tex)
6. Compression (resources inside the [slides](slides/talk6-compression.tex))
    - time: 2021-07-15 Thursday 20:00
    - venue: E402
    - speaker: WEN Hao
    - [slides](slides/talk6-compression.tex)
7. Gradient Tracking in Decentralized Optimization
    - time: 2021-09-09 Thursday 19:00
    - venue: E402
    - speaker: JIN Zhengfen
    - [raw notes](notes/talk7-decentralized.tex) (updating)
8. pFedMac ([main resource](https://arxiv.org/pdf/2107.05330))
    - time: 2021-09-16 Thursday 20:00
    - venue: E402
    - speaker: WEN Hao
    - [slides](slides/talk8-pfedmac.tex) (updating)
9. Operator Splitting and FL (main resources: [FedSplit](https://arxiv.org/abs/2005.05238) and [FedDR](https://arxiv.org/abs/2103.03452))
    - time (planned): 2021-10-28 Thursday 20:00
    - venue: E402
    - speaker: WEN Hao
    - notes/slides (To update....)
<!--7. FedAvg ([PDF](https://arxiv.org/abs/1602.05629))-->
<!--8. Local SGD Converges Fast and Communicates Little ([PDF](https://arxiv.org/abs/1805.09767))-->
<!--9. On the Convergence of FedAvg on Non-IID Data ([PDF](https://arxiv.org/abs/1907.02189))-->
<!--10. Adaptive Federated Optimization ([PDF](https://arxiv.org/abs/2003.00295))-->
<!--11. FedProx ([PDF](https://arxiv.org/abs/1812.06127))-->
<!--12. Federated Learning of a Mixture of Global and Local Models ([PDF](https://arxiv.org/abs/2002.05516))-->
<!--13. more....-->

## Compilation

The best way for compilation is to import this project into [Overleaf](https://www.overleaf.com/).
For local compilation,

```bash
python compile.py
```

with `texlive-full` and `latexmk` etc. installed.

## Code

~~[Code](code) folder contains codes for research purpose, as well as codes that re-implement some published FL algorithms.~~

[Code](code) folder is deprecated. Please refer to [this repo](https://github.com/wenh06/fl-sim) for the latest codes.

## More resources

1. A comprehensive survey article: Advances and open problems in federated learning ([PDF](https://arxiv.org/abs/1912.04977))
2. [Federated Learning One World Seminar (FLOW)](https://sites.google.com/view/one-world-seminar-series-flow/home)
3. [Awesome Federated Learning on GitHub](https://github.com/chaoyanghe/Awesome-Federated-Learning)
4. [Another Introductory Repository on GitHub](https://github.com/ZeroWangZY/federated-learning)
5. [Yet Another Awesome Federated Learning Repository on GitHub](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning)
6. [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench)
7. [References netdisk folder](https://mega.nz/folder/tNoiCbQR#_HgtoFiy4PYc4Uf8-9tYTQ)
8. [AMiner: List of Must-Read](https://www.aminer.org/topic/600e890992c7f9be21d74695)
9. [Boyd ADMM book website](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
