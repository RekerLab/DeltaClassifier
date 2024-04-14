## Code for Model Evaluation

#### adversarial_attacks.py
* Perform Y-shuffling to ablate model performance on 230 benchmarking datasets.

#### cross_validations.py
* Test model performance using 3x10-fold cross-validation on 230 benchmarking datasets.

#### evaluate_additional_test_sets.py
* Evaluate model performance on test sets with demilitarization, removal of same molecule pairs, or removal of inexact values in test sets.

#### models.py
* Functions for the [DeepDeltaClassifier](https://github.com/RekerLab/DeltaClassifer) implementation of [ChemProp](https://github.com/chemprop/chemprop), the[DeltaClassifierLite](https://github.com/RekerLab/DeltaClassifer) implementation of [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html), and the standard regression implementations of [ChemProp](https://github.com/chemprop/chemprop), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), and [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html).

#### scaffold_split.py
* Test model performance using 80-20 scaffold split on 230 benchmarking datasets.
