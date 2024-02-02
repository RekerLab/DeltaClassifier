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
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score as bac
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import roc_auc_score as rocauc
from imblearn.metrics import sensitivity_specificity_support

from scipy import stats as stats
from sklearn.model_selection import KFold
import chemprop
from sklearn.ensemble import RandomForestRegressor as RF
from xgboost import XGBRegressor
from models import *



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
	tn, fp, fn, tp = confusion_matrix(true_vals, pred_vals).ravel()
	sensitivity, specificity, _ = sensitivity_specificity_support(true_vals, pred_vals,labels=[1],average='binary')
	return [accuracy(true_vals,pred_vals),
	f1_score(true_vals,pred_vals),
	rocauc(true_vals,pred_prob)]


properties = ["CHEMBL4561",
"CHEMBL202",
"CHEMBL217",
"CHEMBL228",
"CHEMBL1785",
"CHEMBL1792",
"CHEMBL1801",
"CHEMBL1806",
"CHEMBL1966",
"CHEMBL2716",
"CHEMBL2850",
"CHEMBL3024",
"CHEMBL3234",
"CHEMBL3369",
"CHEMBL3568",
"CHEMBL3582",
"CHEMBL3807",
"CHEMBL4204",
"CHEMBL4372",
"CHEMBL4588",
"CHEMBL5080",
"CHEMBL5285",
"CHEMBL5331",
"CHEMBL5469",
"CHEMBL5697",
"CHEMBL1250348",
"CHEMBL1293311",
"CHEMBL1795186",
"CHEMBL2189110",
"CHEMBL2216739",
"CHEMBL3988599",
"CHEMBL205",
"CHEMBL238",
"CHEMBL239",
"CHEMBL288",
"CHEMBL1781",
"CHEMBL1917",
"CHEMBL2007",
"CHEMBL2563",
"CHEMBL3286",
"CHEMBL3587",
"CHEMBL3721",
"CHEMBL3746",
"CHEMBL4015",
"CHEMBL4072",
"CHEMBL4161",
"CHEMBL4439",
"CHEMBL4462",
"CHEMBL4599",
"CHEMBL4766",
"CHEMBL4835",
"CHEMBL5491",
"CHEMBL5543",
"CHEMBL5620",
"CHEMBL6164",
"CHEMBL1163101",
"CHEMBL1293289",
"CHEMBL2169736",
"CHEMBL3638344",
"CHEMBL4296013",
"CHEMBL249",
"CHEMBL261",
"CHEMBL274",
"CHEMBL338",
"CHEMBL1871",
"CHEMBL1900",
"CHEMBL1929",
"CHEMBL2028",
"CHEMBL2207",
"CHEMBL2439",
"CHEMBL2808",
"CHEMBL2820",
"CHEMBL2993",
"CHEMBL3045",
"CHEMBL3332",
"CHEMBL3468",
"CHEMBL3535",
"CHEMBL3864",
"CHEMBL3910",
"CHEMBL4073",
"CHEMBL4074",
"CHEMBL4128",
"CHEMBL4662",
"CHEMBL5147",
"CHEMBL5192",
"CHEMBL5408",
"CHEMBL6115",
"CHEMBL298",
"CHEMBL299",
"CHEMBL339",
"CHEMBL1856",
"CHEMBL2002",
"CHEMBL2996",
"CHEMBL2998",
"CHEMBL3066",
"CHEMBL3081",
"CHEMBL3476",
"CHEMBL3524",
"CHEMBL3691",
"CHEMBL3891",
"CHEMBL4102",
"CHEMBL4117",
"CHEMBL4303",
"CHEMBL4481",
"CHEMBL4506",
"CHEMBL4696",
"CHEMBL4895",
"CHEMBL4937",
"CHEMBL5747",
"CHEMBL6136",
"CHEMBL1075138",
"CHEMBL1921666",
"CHEMBL2146302",
"CHEMBL2331053",
"CHEMBL2366517",
"CHEMBL4523582",
"CHEMBL241",
"CHEMBL254",
"CHEMBL1855",
"CHEMBL1868",
"CHEMBL2179",
"CHEMBL2345",
"CHEMBL2425",
"CHEMBL2431",
"CHEMBL2593",
"CHEMBL2622",
"CHEMBL2635",
"CHEMBL2828",
"CHEMBL2959",
"CHEMBL3474",
"CHEMBL3475",
"CHEMBL3514",
"CHEMBL3529",
"CHEMBL3855",
"CHEMBL3869",
"CHEMBL3920",
"CHEMBL4068",
"CHEMBL4093",
"CHEMBL4508",
"CHEMBL4681",
"CHEMBL4683",
"CHEMBL4860",
"CHEMBL5366",
"CHEMBL5979",
"CHEMBL6135",
"CHEMBL208",
"CHEMBL209",
"CHEMBL252",
"CHEMBL1808",
"CHEMBL1809",
"CHEMBL1853",
"CHEMBL1926",
"CHEMBL1955",
"CHEMBL2051",
"CHEMBL2252",
"CHEMBL2285",
"CHEMBL2508",
"CHEMBL2534",
"CHEMBL3116",
"CHEMBL3710",
"CHEMBL3776",
"CHEMBL3922",
"CHEMBL3979",
"CHEMBL4321",
"CHEMBL4718",
"CHEMBL4780",
"CHEMBL4804",
"CHEMBL5407",
"CHEMBL6140",
"CHEMBL6154",
"CHEMBL1293244",
"CHEMBL1741186",
"CHEMBL1795116",
"CHEMBL222",
"CHEMBL236",
"CHEMBL1947",
"CHEMBL1980",
"CHEMBL1981",
"CHEMBL1994",
"CHEMBL2127",
"CHEMBL2292",
"CHEMBL2318",
"CHEMBL2335",
"CHEMBL2527",
"CHEMBL2871",
"CHEMBL3229",
"CHEMBL3238",
"CHEMBL3563",
"CHEMBL3577",
"CHEMBL3589",
"CHEMBL3629",
"CHEMBL4026",
"CHEMBL4036",
"CHEMBL4191",
"CHEMBL4393",
"CHEMBL4581",
"CHEMBL4618",
"CHEMBL4803",
"CHEMBL5077",
"CHEMBL5314",
"CHEMBL5462",
"CHEMBL1075145",
"CHEMBL1795177",
"CHEMBL3988596",
"CHEMBL224",
"CHEMBL227",
"CHEMBL233",
"CHEMBL1825",
"CHEMBL1944",
"CHEMBL1952",
"CHEMBL1978",
"CHEMBL2041",
"CHEMBL2108",
"CHEMBL2243",
"CHEMBL2334",
"CHEMBL2487",
"CHEMBL2525",
"CHEMBL2652",
"CHEMBL2730",
"CHEMBL2903",
"CHEMBL3202",
"CHEMBL4550",
"CHEMBL4578",
"CHEMBL4625",
"CHEMBL4801",
"CHEMBL5103",
"CHEMBL6032",
"CHEMBL1293257",
"CHEMBL1293269",
"CHEMBL2189121"]


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
      dataset = '../Datasets/{}-Curated.csv'.format(prop)
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
        dataset = '../Datasets/{}-Curated.csv'.format(prop)
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
