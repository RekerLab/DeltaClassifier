
![DeltaClassifier](https://github.com/RekerLab/DeltaClassifier/assets/127516906/bb265317-43a7-462d-aed1-7371dac2bb84)

## Overview

Much of the currently available biological data is bounded ("less than and greater than gators") and inaccessible to traditional regression algorithms, but pairing regression datapoints and comparing potency values can leverage this bounded data for classification. We present DeltaClassifier, a machine learning method that directly trains upon and learns molecular changes to classify potency improvements leveraging these bounded datapoints. Across 230 small IC50 datasets, both tree-based and neural network-based DeltaClassifiers show significant improvement over traditional regression approaches for the prediction of potency improvements.   

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Comparison Models
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [ChemProp v1.5.2](https://github.com/chemprop/chemprop)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use ChemProp with GPUs, you will need:
* cuda >= 8.0
* cuDNN

<br />


## Descriptions of Folders

### Code

Python code for evaluating DeltaClassifers and traditional models based on their ability to classify potency differences between two molecules.

### Datasets

230 curated training sets for potency prediction from [ChEMBL](https://www.ebi.ac.uk/chembl/).

### Results

Results from 3x10-fold cross-validation.

<br />

## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
