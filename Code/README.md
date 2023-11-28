## Code for Model Evaluation

#### cross_validations.py
* Test model performance using 5x10-fold cross-validation on 56 benchmarking datasets.

#### evaluate_additional_test_sets.py
* Evaluate model performance on test sets with only exact values or demilitarized test sets.

#### models.py
* Functions for [DeepDeltaClassifier](https://github.com/RekerLab/DeltaClassifer), [DeltaClassifierLite](https://github.com/RekerLab/DeltaClassifer),
[ChemProp](https://github.com/chemprop/chemprop), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), and
[XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html) machine learning models.

#### plot_performance_correlation_with_inexact_data.py
* Determine and plot correlation DeepDeltaClassifier improvement over other models with percentage of inexact datapoints in training across 230 datasets. 


