Tensionnet
----------

REPO IS UNDER CONSTRUCTION

Idea is to use NREs to predict the plausible distribution of the tension statistic $R = \frac{P(D_A, D_B)}{P(D_A)P(D_B)}$ over some prior and given a model choice for two data sets with common parameters.

We need a set of simulations for data set A and data set B with the same and mismatched common parameters. We then train the NRE to predict the ratio $R$ for pairs of realisations of each data set. 

Once trained we input a series of matched realisations of data set A and data set B. This gives us the plausible values of $R$ for the two data sets given the model over a prior. We can then use nested sampling to get $Z_A = P(D_A| M)$, $Z_B = P(B_A| M)$ and $Z_{A,B} = P(D_A, D_B |M)$ to get

$R_{obs} = \frac{Z_{A, B}}{Z_A Z_B} = \frac{P(D_A, D_B |M)}{P(D_A| M)P(D_B| M)}$.

If $R_{obs}$ falls well within the distribution then we can conclude that the datasets are likely not in tension (and probably caluclate to what degree?) else if $R_{obs}$ falls outside the distribution then the data sets are in tension.

Generated a mock example that is just two noisy gaussian absorption signals. See below.

Network just uses a binary cross entropy + sigmoid on last layer. We can drop the last layer to get $\log(R)$ but we need it for training to turn $\log(R)$ into a probability.


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