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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", dest = "inFile", type=str)
args = parser.parse_args()

inFile = args.inFile

def cross_validation(x, relation, y, prop, model, k=10, seed=1, demilitarization=-1, only_equals=False): # provide option to cross validate with x and y instead of file
  kf = KFold(n_splits=k, random_state=seed, shuffle=True)
  preds = []
  vals  = []

  for train, test in kf.split(x):
      model.fit(x[train], relation[train], y[train], demilitarization, only_equals) # Fit on training data
      if model.name(demilitarization = 0.1, only_equals = False) == 'XGBoost' or model.name(demilitarization = 0.1, only_equals = False) == 'RandomForest' or model.name(demilitarization = 0.1, only_equals = False) == 'ChemProp50': 
        preds = np.append(preds, model.classify_improvement(x[test], relation[test], y[test], model.predict(x[test]))) # Predict on testing data for traditional models
      else: # Using a DeltaClassifier model
        preds = np.append(preds, model.predict(model.classify_improvement(x[test], relation[test], y[test]))) # Predict on testing data

      # Get true potency differences of the molecular pairs 
      data = pd.DataFrame(np.transpose(np.vstack([x[test],relation[test],y[test]])),columns=["SMILES",'Relation',"Value"])
      dataset = classify_pair_improvement(data, demilitarization=-1, only_equals=False) # apply for all possible pairs
      vals = np.append(vals, dataset['Y']) # Add predictions to the values

  return [vals,preds] # Return true delta values and predicted delta values


def cross_validation_file(data_path, prop, model, k=10, seed=1, demilitarization=-1, only_equals=False): # Cross-validate from a file
  df = pd.read_csv(data_path)
  x = df[df.columns[0]]
  relation = df[df.columns[1]]
  y = df[df.columns[2]]
  return cross_validation(x,relation,y,prop,model,k,seed,demilitarization,only_equals)

def evaluate(pred_vals,true_vals,pred_prob): # Calculate accuracy, f1 score, and rocauc scores
	return [accuracy(true_vals,pred_vals),
	f1_score(true_vals,pred_vals),
	rocauc(true_vals,pred_prob)]


properties = [inFile]


############################################################
### Training Optimization for DeltaClassifiers - 1x10 CV ###
############################################################

# training_approaches contains the demilitarization value and the whether or not we are training on only equals
# A demilitarization value of -1 ensures that there is no demilitarization
training_approaches = [(0.1, False), (-1, False), (-1, True)] # Set for training with our standard demilitarization, all data, and only equals training

models = [DeltaClassifierLite(), DeepDeltaClassifier()]

for prop in properties:
  for training_approach in training_approaches:
    for model in models:
      dataset = '{}-Curated.csv'.format(prop)
      results = cross_validation_file(data_path=dataset, prop = prop, model=model, k=10, seed = 1, demilitarization = training_approach[0], only_equals = training_approach[1])

      pd.DataFrame(results).to_csv("{}_{}_{}.csv".format(prop, model.name(training_approach[0], training_approach[1]), 1), index=False)
      # If you .T the dataframe, then the first column is ground truth, the second is predictions

      df = pd.read_csv("{}_{}_{}.csv".format(prop, model.name(training_approach[0], training_approach[1]), 1)).T
      df.columns =['True', 'Pred']
      trues = df['True'].tolist()
      preds = df['Pred'].tolist()

      # Get Additional Metrics for the Models
      results = pd.DataFrame(columns=['model', 'accuracy', 'f1','rocauc'])
      preds2 = df["Pred"] > 0.5 # Get the binary predictions instead of predicted probability
      trues2 = df["True"] > 0.5 # Get the binary values for the classification problem
      results = evaluate(preds2, trues2, df["Pred"]) # Calculate accuracy, f1 score, and rocauc scores 

      results = pd.DataFrame({'model': [model.name(training_approach[0], training_approach[1])], 'accuracy': [results[0]],
                              'f1': [results[1]], 'rocauc': [results[2]]})

      results.to_csv("{}-{}-Metrics-{}.csv".format(prop, model.name(training_approach[0], training_approach[1]), 1), index = False)
        
        
        
        
###################################################
### Compare to Traditional Approaches - 3x10 CV ###
###################################################

models = [Trad_RandomForest(), Trad_ChemProp(), Trad_XGBoost(), DeltaClassifierLite(), DeepDeltaClassifier()]

for prop in properties:
  for model in models:
    for i in range(1,4): # 3 repeats of cross-validation
        dataset = '{}-Curated.csv'.format(prop)
        results = cross_validation_file(data_path=dataset, prop = prop, model=model, k=10, seed = i, demilitarization = 0.1, only_equals = False)

        pd.DataFrame(results).to_csv("{}_{}_{}.csv".format(prop, model.name(demilitarization = 0.1, only_equals = False), i), index=False)
        # If you .T the dataframe, then the first column is ground truth, the second is predictions

        df = pd.read_csv("{}_{}_{}.csv".format(prop, model.name(demilitarization = 0.1, only_equals = False), i)).T
        df.columns =['True', 'Pred']
        trues = df['True'].tolist()
        preds = df['Pred'].tolist()

        # Get Additional Metrics for the Models
        results = pd.DataFrame(columns=['model', 'accuracy', 'f1','rocauc'])
        preds2 = df["Pred"] > 0.5 # Get the binary predictions instead of predicted probability
        trues2 = df["True"] > 0.5 # Get the binary values for the classification problem
        results = evaluate(preds2, trues2, df["Pred"]) # Calculate accuracy, f1 score, and rocauc scores 

        results = pd.DataFrame({'model': [model.name(demilitarization = 0.1, only_equals = False)], 'accuracy': [results[0]],
                                'f1': [results[1]], 'rocauc': [results[2]]})

        results.to_csv("{}-{}-Metrics-{}.csv".format(prop, model.name(demilitarization = 0.1, only_equals = False), i), index = False)
