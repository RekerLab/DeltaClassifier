
![DeltaClassifier](https://github.com/RekerLab/DeltaClassifier/assets/127516906/bb265317-43a7-462d-aed1-7371dac2bb84)

## Overview

Molecular machine learning algorithms are becoming increasingly powerful with increasing data availability. However, much of the currently available biological data is inexact and inaccessible to traditional regression algorithms. Pairing regression datapoints and comparing potency values can leverage this inexact data for classification. We present a machine learning approach (DeltaClassifier) to directly train upon and learn molecular changes to classify molecular improvements. Across 230 small ChEMBL datasets of IC50 values, both tree-based and neural network-based DeltaClassifiers show significant improvement over traditional regression approaches when collapsed to classifications for the prediction of molecular improvements.   

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Comparison Models
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [ChemProp v1.5.2](https://github.com/chemprop/chemprop)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use ChemProp with GPUs, you will need:
* cuda >= 8.0
* cuDNN

<br />


## Descriptions of Folders

### Code

Python code for evaluating DeltaClassifer and traditional models based on their ability to classify potency differences between two molecules.

### Datasets

230 curated benchmarking training sets for potency prediction from [ChEMBL](https://www.ebi.ac.uk/chembl/).

### Results

Results from 1x10-fold cross-validation.

<br />

## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
