Tensionnet
----------

Idea is to use NREs to predict the plausible distribution of the tension statistic $R = \frac{P(A, B)}{P(A)P(B)}$ for two data sets with common parameters.

In order to do this we need a set of simulator for data set A and data set B with the same and mismatched common parameters. We then train the NRE to predict the ratio $R$ for pairs of realisations of each data set. 

Once trained we input a series of matched realisations of data set A and data set B. This gives us the plausible values of $R$ for the two data sets with which we can compare the true value for the real data calucalted with Nested Sampling runs. If $R_{obs}$ falls well within the distribution then we can conclude that the datasets are likely not in tension else if $R_{obs}$ falls in the tails then the data sets are in tension.

Need a simple test case to versify against. Maybe two in tension mock observations 21-cm signal + noise with no foregrounds.

Stolen some of Thomas code for reading in the SARAS3 and EDGES data.

Network just uses a binary cross entropy + sigmoid on last layer which means it outputs

$\sigma (\log r) = \frac{1}{1 + \exp(-\log r)}$

which we can recover r from by doing

$r = \frac{\sigma}{1 - \sigma}$

Toy Example
-----------

Two experiments observing a gaussian absorption feature in different bands with
the same level of noise.

When I train the classifier I get the following confusion matrix for
1000 test samples. Seems the network isn't very good at recognising when the
data sets are not in tension which is where we need it to be performing well
otherwise we cannot trust the predicted distribution for R.
Here confused is any value of $R$ between 0.25 and 0.75. Otherwise prediction
is correct or wrong.

![confusion matrix](https://github.com/htjb/tension-networks/blob/main/test_confusion_matrix.png)

And if I try and predict R for a set of 2000 matched realisations (no tension, 
same parameters) of the two experiments I get the following distribution

![R distribution](https://github.com/htjb/tension-networks/blob/main/test_r_hist.png)

where the dashed line is $R_{obs}$ for two experiments that observed the same
signal and the dotten line is for two experiments in tension.

The data corresponding to the two $R_{obs}$ values is shown below.

![R distribution](https://github.com/htjb/tension-networks/blob/main/test_case_data.png)