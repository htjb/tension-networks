Tensionnet
----------

Idea is to use NREs to predict the plausible distribution of the tension statistic $R = \frac{P(D_A, D_B)}{P(D_A)P(D_B)}$ over some prior and given a model choice for two data sets with common parameters.

We need a set of simulations for data set A and data set B with the same and mismatched common parameters. We then train the NRE to predict the ratio $R$ for pairs of realisations of each data set. 

Once trained we input a series of matched realisations of data set A and data set B. This gives us the plausible values of $R$ for the two data sets given the model over a prior. We can then use nested sampling to get $Z_A = P(D_A| M)$, $Z_B = P(B_A| M)$ and $Z_{A,B} = P(D_A, D_B |M)$ to get

$R_{obs} = \frac{Z_{A, B}}{Z_A Z_B} = \frac{P(D_A, D_B |M)}{P(D_A| M)P(D_B| M)}$.

If $R_{obs}$ falls well within the distribution then we can conclude that the datasets are likely not in tension (and probably caluclate to what degree?) else if $R_{obs}$ falls outside the distribution then the data sets are in tension.

Generated a mock example that is just two noisy gaussian absorption signals. See below.

Network just uses a binary cross entropy + sigmoid on last layer. We can drop the last layer to get $\log(R)$ but we need it for training to turn $\log(R)$ into a probability.

Toy Example
-----------

Two experiments observing a gaussian absorption feature in different bands with
the same level of noise.

Training on 200,000 pairs of data sets here (half in tension).

This was initially not working. I edited the structure of the last few layers a bit and changed the normalisation on the simulated data sets which seems to have helped.

When I train the classifier I get the following confusion matrix for
1000 test samples which looks good. Here confused is any value of $R$ between 0.25 and 0.75. Otherwise prediction
is correct or wrong.

![confusion matrix](https://github.com/htjb/tension-networks/blob/main/test_confusion_matrix.png)

And if I try and predict R for a set of 2000 matched realisations (no tension, 
same parameters) of the two experiments I get the following distribution

![R distribution](https://github.com/htjb/tension-networks/blob/main/test_r_hist.png)

where the dashed line is $R_{obs}$ for two experiments that observed the same
signal and the dotten line is for two experiments in tension. Of the 2000 models histogrammed 0.15% were mis-classified as being in tension. As a kde this looks like

![R distribution](https://github.com/htjb/tension-networks/blob/main/test_r_kde.png)

The data corresponding to the two $R_{obs}$ values is shown below.

![R distribution](https://github.com/htjb/tension-networks/blob/main/test_case_data.png)

BAO + Planck Example
--------------------

Need some toy likelihoods to be able to do this. Built a BAO likelihood using the SDSS DR12 and DR16 measurements/covariances of angular diameter/hubble distance divided by the sound horizon and CAMB.

Had some issues with the prior/nlive. Found that with a wide prior the run was terminating early with the warning `Warning, unable to proceed after    151: failed spawn events`. Might be able to solve it with increased nlive but currently using the narrow prior from Wills qunatifying tensions paper. Get the foloowing results

![bao fit](https://github.com/htjb/tension-networks/blob/main/bao_fit_narrow_prior.png)


Currently setting up Planck likelihood.

Reporting evidences below...

| | Evidence |
|----|------|
|BAO | $-4.238 \pm 0.092$ |
| Planck | $- 53.95 \pm 0.19$|
| Planck + BAO |