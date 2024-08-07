# Imports
import os
import abc
import math
import shutil
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import sklearn
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as rocauc

from scipy import stats as stats
from sklearn.model_selection import KFold
import chemprop
from sklearn.ensemble import RandomForestRegressor as RF
from xgboost import XGBRegressor
from models import *


# Adjustable Parameters
model = DeepDeltaClassifier()
training_data = 'CHEMBL202.csv'
testing_data = 'CHEMBL202_ext.csv'

# Fit model on entire training dataset
df = pd.read_csv(training_data)
x = df[df.columns[0]]
relation = df[df.columns[1]]
y = df[df.columns[2]]
model.fit(x,relation, y, 0.1, False) 

# Predict on external test set
df_pred = pd.read_csv(testing_data)
df_pred['Pred_Y'] = model.predict(df) # Make predictions      
df_pred.to_csv("{}_{}_ExtTestSet.csv".format(prop, model), index=False) # Save results

