# Notes

Just so i dont forget

cosmopower_joint_likelihood.keras shuffles planck i.e. the sets are
$s = {D_w(\theta), \theta, D_P(\theta)}$ and $s^\prime = {D_w(\theta), \theta, D_P(\phi)}$ 

whereas cosmopower_joint_likelihoodwmap.keras shuffles wmap i.e. the sets are 
$s = {D_w(\theta), \theta, D_P(\theta)}$ and $s^\prime = {D_w(\phi), \theta, D_P(\theta)}$ 

unless otherwise stated with `_planck_centric' in file string everything is done with cosmopower_joint_likelihood.keras.