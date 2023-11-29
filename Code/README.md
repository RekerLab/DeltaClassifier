## Code for Model Evaluation

#### cross_validations.py
* Test model performance using 1x10-fold cross-validation on 230 benchmarking datasets.

#### evaluate_additional_test_sets.py
* Evaluate model performance on test sets with demilitarization, removal of same molecule pairs, or removal of inexact values in test sets.

#### models.py
* Functions for [DeepDeltaClassifier](https://github.com/RekerLab/DeltaClassifer), [DeltaClassifierLite](https://github.com/RekerLab/DeltaClassifer),
[ChemProp](https://github.com/chemprop/chemprop), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), and
[XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html) machine learning models to classify molecular improvements.
