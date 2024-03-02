
![DeltaClassifierSimple](https://github.com/RekerLab/DeltaClassifier/assets/127516906/d0e68a7c-5ea0-418a-b41c-e1eae837958c)


## Overview

A substantial amount of inhibition data is bounded and inaccessible to traditional regression algorithms, but pairing regression datapoints and comparing potency values can leverage this bounded data for classification. We present DeltaClassifier, a novel molecular pairing approach to process this data. This creates a new classification task of predicting which one of two paired molecules is more potent. This novel classification task can be accurately solved by various, established molecular machine learning algorithms, including XGBoost and Chemprop. Across 230 ChEMBL IC50 datasets, both tree-based and neural network-based “DeltaClassifiers” show improvements over traditional regression approaches in correctly classifying molecular potency improvements. 

The associated publication is currently under review. 

If you use the MPNN-based implementation, please also kindly cite: Vermeire, F. H., & Green, W. H. (2021). Transfer learning for solvation free energies: From quantum chemistry to experiments. Chemical Engineering Journal, 418, 129307.

If you use the tree-based implementation, please also kindly cite: Mitchell, R., & Frank, E. (2017). Accelerating the XGBoost algorithm using GPU computing. PeerJ Computer Science, 3, e127.

We would like to thank the Chemprop and the Scikit-learn developers for making their machine learning algorithms publicly available. 

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
