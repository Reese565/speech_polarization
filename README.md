# Politics as a Full-Contact Sport
## A Neural Topic Modeling Approach to Gauging Polarization in Congressional Speeches

## Description

We take a deep learning approach to measuing political polarization in the United States using almost 4 million congressional speech documents ranging from 1873 to 2011. Our approach reveals reveals a polarization trend that increases steadily from the post-war period to 2011. Morever, our neural model also facilitates the inference of the granular underlying topics driving polarization. Find technical details and insights in the [paper](https://github.com/Reese565/speech_polarization/blob/master/SpeechPolarization_Cassius%26Williams.pdf).


## Architecture

We implement a neural architecture similar to the Relationship Modeling Network (RMN) put forward bby Iyyer et al. [(2016)](https://www.aclweb.org/anthology/N16-1180/). The code used to implement our unique variant of the RMN is [here](https://github.com/Reese565/speech_polarization/blob/master/scripts/modeling/rmn.py).

