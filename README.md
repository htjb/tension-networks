Tensionnet
----------

Idea is to use NREs to predict the plausible distribution of the tension statistic $R = \frac{P(A, B)}{P(A)P(B)}$ for two data sets with common parameters.

In order to do this we need a set of simulator for data set A and data set B with the same and mismatched common parameters. We then train the NRE to predict the ratio $R$ for pairs of realisations of each data set. 

Once trained we input a series of matched realisations of data set A and data set B. This gives us the plausible values of $R$ for the two data sets with which we can compare the true value for the real data calucalted with Nested Sampling runs. If $R_{obs}$ falls well within the distribution then we can conclude that the datasets are likely not in tension else if $R_{obs}$ falls in the tails then the data sets are in tension.

The code seems to be doing the right thing but the classifier is not learning whether the two data sets go together.

Need a simple test case to versify against. Maybe two in tension mock observations 21-cm signal + noise with no foregrounds.

Stolen some of Thomas code for reading in the SARAS3 and EDGES data.

Network just uses a binary cross entropy + sigmoid on last layer which means it outputs

$\sigma (\log r) = \frac{1}{1 + \exp(-\log r)}$

which we can recover r from by doing

$r = \exp(-\log(1/\sigma - 1))$

Hmm the above is causing some issues with divide by zeros. Lets try this again


$\sigma (\log r) = \frac{1}{1 + \exp(-\log r)}$

$\sigma (\log r) = \frac{1}{1 + \exp(\log 1/r)}$

$\sigma (\log r) = \frac{1}{1 + 1/r}$

$ \sigma = \frac{r}{1 + r}$

$\sigma + r \sigma = r$

$r = \frac{\sigma}{1 - \sigma}$

## To Dos

- [ ] Need to generate some 21cm signal + noise data sets that are in and out of tension.
- [ ] Need to write load and save functions!