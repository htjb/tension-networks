Tensionnet
----------

Code for Bevins, Handley, Gessey-Jones, [Calibrating Bayesian Tension Statistics using Neural Ratio Estimation](https://arxiv.org/abs/2407.15478), 2024.

When the inference from two expeiments disgrees this is known as tension.
Understanding and properly quantifying tension helps us to better understand
any systematics in our analysis and identify whether new physics is needed
to explain the observed phemonena. A number of tensions have kept researchers
busy in recent years including the Hubble tension and the $\sigma_8$ tension.

A Bayesian way to quantify tension is with the ratio 

$R = \frac{P(D_A, D_B)}{P(D_A)P(D_B)} = \frac{\mathcal{Z}_{AB}}{\mathcal{Z}_{A}\mathcal{Z}_{B}}$

where $D_A$ is the data from experiment $A$, $D_B$ is the data from experiment $B$,
$\mathcal{Z}_{AB}$ is the Bayesian evidence for the joint data set and $\mathcal{Z}_{A}$
and $\mathcal{Z}_{B}$ are the Bayesian evidences for the individual data sets. $R$ is
often very costly to calcualte since it involves evaluating three different evidences
and it is also hard to interpret because it has a non-trivial prior dependence.

We demonstrate that Neural Ratio Estimation can be used to calcualte the bayesian
tension statistics $R = \frac{P(D_A, D_B)}{P(D_A)P(D_B)}$ if properly
trained on simulations of two experiments observables. We then show that NREs can be used to predict the plausible distribution of $R$ given some prior and model choice for two data sets with common parameters. This inconcordance $\log R$ distribution can then
be used for calibrating the observed $R$ value for the real data into a prior independent
$N \sigma$ estimate of tension where here $\sigma$ refers to the standard deviation of the
standard normal distribution.


Citation
--------

If you use the tensionnet work in any of your papers please cite the paper
[Calibrating Bayesian Tension Statistics using Neural Ratio Estimation](https://arxiv.org/abs/2407.15478).

```bibtex
@ARTICLE{2024arXiv240715478B,
       author = {{Bevins}, Harry T.~J. and {Handley}, William J. and {Gessey-Jones}, Thomas},
        title = "{Calibrating Bayesian Tension Statistics using Neural Ratio Estimation}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = jul,
          eid = {arXiv:2407.15478},
        pages = {arXiv:2407.15478},
archivePrefix = {arXiv},
       eprint = {2407.15478},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240715478B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```