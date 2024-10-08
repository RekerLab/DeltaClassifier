
![DeltaClassifierSimple](https://github.com/RekerLab/DeltaClassifier/assets/127516906/d0e68a7c-5ea0-418a-b41c-e1eae837958c)


## Overview

A substantial amount of inhibition data is bounded and inaccessible to traditional regression algorithms, but pairing regression datapoints and comparing potency values can leverage this bounded data for classification. We present DeltaClassifier, a novel molecular pairing approach to process this data. This creates a new classification task of predicting which one of two paired molecules is more potent. This novel classification task can be accurately solved by various, established molecular machine learning algorithms, including XGBoost and Chemprop. Across 230 ChEMBL IC50 datasets, both tree-based and neural network-based “DeltaClassifiers” show improvements over traditional regression approaches in correctly classifying molecular potency improvements. 

For more information, please refer to the [associated publication](https://pubs.rsc.org/en/content/articlelanding/2024/md/d4md00325j#cit11)

If you use this data or code, please kindly cite: Fralish, Z., Skaluba, P., & Reker, D. (2024). Leveraging bounded datapoints to classify molecular potency improvements. RSC Med Chem. 15, 2474-2482.

We would like to thank the Chemprop, XGBoost, and the Scikit-learn developers for making their machine learning algorithms publicly available. 

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Machine Learning Models
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [Chemprop v1.5.2](https://github.com/chemprop/chemprop)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use Chemprop with GPUs, you will need:
* cuda >= 8.0
* cuDNN

<br />


## Descriptions of Folders

### Code

Python code for evaluating DeltaClassifers and traditional models based on their ability to classify potency differences between two molecules.

### Command_Line_Implementation

Provides shell scripts to run directly from the command line.

### Datasets

230 curated training sets for potency prediction from [ChEMBL](https://www.ebi.ac.uk/chembl/).

### Results

Results from 3x10-fold cross-validation and 80-20 scaffold split.

<br />

## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
