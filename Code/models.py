# Imports
import os
import abc
import math
import shutil
import tempfile
import numpy as np
import pandas as pd
import chemprop
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RF
from xgboost import XGBRegressor


# Define abstract class to define interface of models
class AbstractModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass


class Trad_RandomForest(abstractDeltaModel):
    model = None

    def __init__(self):
        self.model = RF()

    def fit(self, x, relation, y, metric='r2'):
        data = pd.DataFrame({'X': x, 'Relation': relation,'Y': y})
        data = data[data['Relation'] == '='] # Remove any values that are not exact
        data.index = range(len(data)) # Reindex
        data = data[['X', 'Y']]# Remove the relation column

        mols = [Chem.MolFromSmiles(s) for s in data['X']]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        self.model.fit(fps,data['Y']) # Fit using traditional methods

    def predict(self, x, relation, y):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        cleanedx = [i for i in x if str(i) != 'nan']
        cleanedrelation = [i for i in relation if str(i) != 'nan']
        cleanedy = [i for i in y if str(i) != 'nan']
        predictions = pd.DataFrame(self.model.predict(fps)) # Predict using traditional methods
        data = pd.DataFrame({'SMILES': cleanedx, 'Relation': cleanedrelation,'Value': cleanedy,'Predictions': predictions[0]})
        paired_predictions = classify_improvement_CTRL_normalized(data)
        return paired_predictions

    def __str__(self):
        return "RandomForest"
        
        
class Trad_ChemProp(abstractDeltaModel):
    epochs = None
    dirpath = None
    dirpath_single = None

    def __init__(self, epochs=50, dirpath = None, dirpath_single = None):
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit(self, x, relation, y, metric='r2'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name # use temporary file to store model

        data = pd.DataFrame(np.transpose(np.vstack([x,relation,y])),columns=["X",'Relation',"Y"])
        data = data[data['Relation'] == '='] # Remove any values that are not exact
        data = data[['X', 'Y']]# Remove the relation column

        train = pd.DataFrame(data)

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        # store default arguments for ChemProp model
        arguments = [
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name,
            '--dataset_type', 'regression',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '1',
            '--metric', metric,
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x, relation, y):

        dataset = pd.DataFrame(x)

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name,
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '1'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Make prediction

        predictions = pd.read_csv(temp_predfile.name)['Y']

        data = pd.DataFrame(np.transpose(np.vstack([x,relation,y,predictions])),columns=["SMILES",'Relation',"Value","Predictions"])
        paired_predictions = classify_improvement_CTRL_normalized(data)

        temp_datafile.close()
        temp_predfile.close()

        return paired_predictions

    def __str__(self):
        return "ChemProp" + str(self.epochs)
        
        
class Trad_XGBoost(abstractDeltaModel):
    model = None

    def __init__(self):
        self.model = XGBRegressor(tree_method='gpu_hist')

    def fit(self, x, relation, y, metric='r2'):
        data = pd.DataFrame({'X': x, 'Relation': relation,'Y': y})
        data = data[data['Relation'] == '='] # Remove any values that are not exact
        data.index = range(len(data)) # Reindex
        data = data[['X', 'Y']]# Remove the relation column

        mols = [Chem.MolFromSmiles(s) for s in data['X']]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        self.model.fit(fps,data['Y']) # Fit using traditional methods

    def predict(self, x, relation, y):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        cleanedx = [i for i in x if str(i) != 'nan']
        cleanedrelation = [i for i in relation if str(i) != 'nan']
        cleanedy = [i for i in y if str(i) != 'nan']
        predictions = pd.DataFrame(self.model.predict(fps)) # Predict using traditional methods
        data = pd.DataFrame({'SMILES': cleanedx, 'Relation': cleanedrelation,'Value': cleanedy,'Predictions': predictions[0]})
        paired_predictions = classify_improvement_CTRL_normalized(data)
        return paired_predictions

    def __str__(self):
        return "XGBoost"
        

class DeltaClassifierLiteOnlyEquals(AbstractModel):
    model = None

    def __init__(self):
        self.model = XGBRegressor(tree_method='gpu_hist')

    def fit(self, x, relation, y):
        data = pd.DataFrame({'SMILES': x, 'Relation': relation,'Value': y})
        train = classify_improvement_XGBoost_only_equals(data)
        self.model.fit(np.vstack(train.fps.to_numpy()), train.Y) # Fit using traditional methods

    def predict(self, x, relation, y):
        data = pd.DataFrame({'SMILES': x, 'Relation': relation,'Value': y})
        data2 = classify_improvement_XGBoost(data)
        predictions = pd.DataFrame(self.model.predict(np.vstack(data2.fps.to_numpy()))) # Predict using traditional methods
        return predictions

    def __str__(self):
        return "DeltaClassifierLiteOnlyEquals"
        
        

class DeltaClassifierLite(AbstractModel):
    model = None

    def __init__(self):
        self.model = XGBRegressor(tree_method='gpu_hist')

    def fit(self, x, relation, y):
        data = pd.DataFrame({'SMILES': x, 'Relation': relation,'Value': y})
        train = classify_improvement_XGBoost(data)
        self.model.fit(np.vstack(train.fps.to_numpy()), train.Y) # Fit using traditional methods

    def predict(self, x, relation, y):
        data = pd.DataFrame({'SMILES': x, 'Relation': relation,'Value': y})
        data2 = classify_improvement_XGBoost(data)
        predictions = pd.DataFrame(self.model.predict(np.vstack(data2.fps.to_numpy()))) # Predict using traditional methods
        return predictions

    def __str__(self):
        return "DeltaClassifierLite"


class XGBoost_DeltaClassifier_DM(AbstractModel):
    model = None

    def __init__(self):
        self.model = XGBRegressor(tree_method='gpu_hist')

    def fit(self, x, relation, y):
        data = pd.DataFrame({'SMILES': x, 'Relation': relation,'Value': y})
        train = classify_improvement_XGBoost_DM(data)
        self.model.fit(np.vstack(train.fps.to_numpy()), train.Y) # Fit using traditional methods

    def predict(self, x, relation, y):
        data = pd.DataFrame({'SMILES': x, 'Relation': relation,'Value': y})
        data2 = classify_improvement_XGBoost(data)
        predictions = pd.DataFrame(self.model.predict(np.vstack(data2.fps.to_numpy()))) # Predict using traditional methods
        return predictions

    def __str__(self):
        return "XGBoost_DM01"


class DeepDeltaClassifierOnlyEquals(AbstractModel):
    epochs = None
    dirpath = None
    dirpath_single = None

    def __init__(self, epochs=5, dirpath = None, dirpath_single = None): # ADJUST EPOCHS ACCORDINGLY!!!!!!!!
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit(self, x, relation, y, metric='auc'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name # use temporary file to store model

        # create pairs of training data
        data = pd.DataFrame(np.transpose(np.vstack([x,relation,y])),columns=["SMILES",'Relation',"Value"])
        train = classify_improvement_only_equals(data)

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        # store default arguments for ChemProp model
        arguments = [
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name,
            '--dataset_type', 'classification',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '2',
            '--metric', metric,
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x, relation, y):

        # create pairs of training data
        data = pd.DataFrame(np.transpose(np.vstack([x,relation,y])),columns=["SMILES",'Relation',"Value"])
        dataset = classify_improvement(data)

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name,
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Make prediction

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions

    def __str__(self):
        return "DeepDeltaClassifierOnlyEquals" + str(self.epochs)




class DeepDeltaClassifierAllData(AbstractModel):
    epochs = None
    dirpath = None
    dirpath_single = None

    def __init__(self, epochs=5, dirpath = None, dirpath_single = None): 
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit(self, x, relation, y, metric='auc'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name # use temporary file to store model

        # create pairs of training data
        data = pd.DataFrame(np.transpose(np.vstack([x,relation,y])),columns=["SMILES",'Relation',"Value"])
        train = classify_improvement(data)

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        # store default arguments for ChemProp model
        arguments = [
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name,
            '--dataset_type', 'classification',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '2',
            '--metric', metric,
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x, relation, y):

        # create pairs of training data
        data = pd.DataFrame(np.transpose(np.vstack([x,relation,y])),columns=["SMILES",'Relation',"Value"])
        dataset = classify_improvement(data)

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name,
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Make prediction

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions

    def __str__(self):
        return "DeepDeltaClassifierAllData" + str(self.epochs)




class DeepDeltaClassifier(AbstractModel):
    epochs = None
    dirpath = None
    dirpath_single = None

    def __init__(self, epochs=1, dirpath = None, dirpath_single = None): 
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit(self, x, relation, y, metric='auc'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name # use temporary file to store model

        # create pairs of training data
        data = pd.DataFrame(np.transpose(np.vstack([x,relation,y])),columns=["SMILES",'Relation',"Value"])
        train = classify_improvement_demilitarized(data) # Function to cross-merge and collapse regression into classification with Demilitarized Data

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        # store default arguments for ChemProp model
        arguments = [
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name,
            '--dataset_type', 'classification',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '2',
            '--metric', metric,
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x, relation, y):

        # create pairs of training data
        data = pd.DataFrame(np.transpose(np.vstack([x,relation,y])),columns=["SMILES",'Relation',"Value"])
        dataset = classify_improvement(data)

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name,
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Make prediction

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions

    def __str__(self):
        return "DeepDeltaClassifier" + str(self.epochs)


# Function to make pairs, determine if the pair improves or not, and remove values where this is unknown
def classify_improvement(data): # Specific version for MPNN-based models
  data2 = pd.merge(data, data, how='cross') # Make Pairs
  data3 = pd.DataFrame(columns=['SMILES_x', 'SMILES_y', 'Y'], index=range(len(data2))) # For final results

  for i in range(len(data2)):
    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse

    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '>':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '<':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '>':
      continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '<':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '>':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '<':
      continue # Unsure which datapoint is better - don't keep pair


  data3 = data3[data3['Y'].notna()] # Remove anything without values
  data3.index = range(len(data3)) # Reindex
  data3['Y']=data3['Y'].astype('int') # Ensure values are interpreted as integers
  return data3


# Function to make pairs, determine if the pair improves or not, and remove values where this is unknown
def classify_improvement_demilitarized(data): # Specific version for also implemented a buffer room for close pairs for MPNN-based models
  data2 = pd.merge(data, data, how='cross') # Make Pairs
  data3 = pd.DataFrame(columns=['SMILES_x', 'SMILES_y', 'Y'], index=range(len(data2))) # For final results

  for i in range(len(data2)):
    if abs(data2['Value_x'][i] - data2['Value_y'][i]) > 0.1: # Demilitarize
      if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '=':
        if data2['Value_x'][i] < data2['Value_y'][i]:
          data3["SMILES_x"][i] =  data2['SMILES_x'][i]
          data3["SMILES_y"][i] =  data2['SMILES_y'][i]
          data3['Y'][i] = 1 # Datapoint 2 is better
        else:
          data3["SMILES_x"][i] =  data2['SMILES_x'][i]
          data3["SMILES_y"][i] =  data2['SMILES_y'][i]
          data3['Y'][i] = 0 # Datapoint 2 is worse

      if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '>':
        if data2['Value_x'][i] < data2['Value_y'][i]:
          data3["SMILES_x"][i] =  data2['SMILES_x'][i]
          data3["SMILES_y"][i] =  data2['SMILES_y'][i]
          data3['Y'][i] = 1 # Datapoint 2 is better
        else:
          continue # Unsure which datapoint is better - don't keep pair

      if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '<':
        if data2['Value_x'][i] > data2['Value_y'][i]:
          data3["SMILES_x"][i] =  data2['SMILES_x'][i]
          data3["SMILES_y"][i] =  data2['SMILES_y'][i]
          data3['Y'][i] = 0 # Datapoint 2 is worse
        else:
          continue # Unsure which datapoint is better - don't keep pair

      if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '=':
        if data2['Value_x'][i] > data2['Value_y'][i]:
          data3["SMILES_x"][i] =  data2['SMILES_x'][i]
          data3["SMILES_y"][i] =  data2['SMILES_y'][i]
          data3['Y'][i] = 0 # Datapoint 2 is worse
        else:
          continue # Unsure which datapoint is better - don't keep pair

      if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '>':
        continue # Unsure which datapoint is better - don't keep pair

      if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '<':
        if data2['Value_x'][i] > data2['Value_y'][i]:
          data3["SMILES_x"][i] =  data2['SMILES_x'][i]
          data3["SMILES_y"][i] =  data2['SMILES_y'][i]
          data3['Y'][i] = 0 # Datapoint 2 is worse
        else:
          continue # Unsure which datapoint is better - don't keep pair

      if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '=':
        if data2['Value_x'][i] < data2['Value_y'][i]:
          data3["SMILES_x"][i] =  data2['SMILES_x'][i]
          data3["SMILES_y"][i] =  data2['SMILES_y'][i]
          data3['Y'][i] = 1 # Datapoint 2 is better
        else:
          continue # Unsure which datapoint is better - don't keep pair

      if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '>':
        if data2['Value_x'][i] < data2['Value_y'][i]:
          data3["SMILES_x"][i] =  data2['SMILES_x'][i]
          data3["SMILES_y"][i] =  data2['SMILES_y'][i]
          data3['Y'][i] = 1 # Datapoint 2 is better
        else:
          continue # Unsure which datapoint is better - don't keep pair

      if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '<':
        continue # Unsure which datapoint is better - don't keep pair


  data3 = data3[data3['Y'].notna()] # Remove anything without values
  data3.index = range(len(data3)) # Reindex
  data3['Y'] = data3['Y'].astype('int') # Ensure values are interpreted as integers
  return data3
  
  
# Function to make pairs, determine if the pair improves or not, and remove values where this is unknown
def classify_improvement_only_equals(data): # Specific version to only train on the absolute values for MPNN models
  data2 = pd.merge(data, data, how='cross') # Make Pairs
  data3 = pd.DataFrame(columns=['SMILES_x', 'SMILES_y', 'Y'], index=range(len(data2))) # For final results

  for i in range(len(data2)):
    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse
    else:
      continue # Unsure which datapoint is better - don't keep pair

  data3 = data3[data3['Y'].notna()] # Remove anything without values
  data3.index = range(len(data3)) # Reindex
  data3['Y']=data3['Y'].astype('int') # Ensure values are interpreted as integers
  return data3
  
  
  


# Function to make pairs, determine if the pair improves or not, and remove values where this is unknown
def classify_improvement_XGBoost(data): # Specific version for XGBoost-based models
  data2 = pd.merge(data, data, how='cross') # Make Pairs
  data3 = pd.DataFrame(columns=['fps', 'Y'], index=range(len(data2))) # For final results

  for i in range(len(data2)):
    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
        fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
        data3['fps'][i] =  np.append(fps_x, fps_y)
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
        fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
        data3['fps'][i] =  np.append(fps_x, fps_y)
        data3['Y'][i] = 0 # Datapoint 2 is worse

    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '>':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
        fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
        data3['fps'][i] =  np.append(fps_x, fps_y)
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '<':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
        fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
        data3['fps'][i] =  np.append(fps_x, fps_y)
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
        fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
        data3['fps'][i] =  np.append(fps_x, fps_y)
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '>':
      continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '<':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
        fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
        data3['fps'][i] =  np.append(fps_x, fps_y)
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
        fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
        data3['fps'][i] =  np.append(fps_x, fps_y)
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '>':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
        fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
        data3['fps'][i] =  np.append(fps_x, fps_y)
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '<':
      continue # Unsure which datapoint is better - don't keep pair


  data3 = data3[data3['Y'].notna()] # Remove anything without values
  data3.index = range(len(data3)) # Reindex
  data3['Y']=data3['Y'].astype('int') # Ensure values are interpreted as integers
  return data3


# Function to make pairs, determine if the pair improves or not, and remove values where this is unknown
def classify_improvement_XGBoost_only_equals(data): # Specific version to only train on the absolute values for XGBoost-based models
  data2 = pd.merge(data, data, how='cross') # Make Pairs
  data3 = pd.DataFrame(columns=['SMILES_x', 'SMILES_y', 'Y'], index=range(len(data2))) # For final results

  for i in range(len(data2)):
    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse
    else:
      continue # Unsure which datapoint is better - don't keep pair

  data3 = data3[data3['Y'].notna()] # Remove anything without values
  data3.index = range(len(data3)) # Reindex
  data3['Y']=data3['Y'].astype('int') # Ensure values are interpreted as integers
  return data3
  
  
# Function to make pairs, determine if the pair improves or not, and remove values where this is unknown
def classify_improvement_XGBoost_DM(data): # Specific version for XGBoost-based models with demilitarized data
  data2 = pd.merge(data, data, how='cross') # Make Pairs
  data3 = pd.DataFrame(columns=['fps', 'Y'], index=range(len(data2))) # For final results

  for i in range(len(data2)):
    if abs(data2['Value_x'][i] - data2['Value_y'][i]) > 0.1: # Demilitarize
        if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '=':
            if data2['Value_x'][i] < data2['Value_y'][i]:
                fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
                fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
                data3['fps'][i] =  np.append(fps_x, fps_y)
                data3['Y'][i] = 1 # Datapoint 2 is better
            else:
                fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
                fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
                data3['fps'][i] =  np.append(fps_x, fps_y)
                data3['Y'][i] = 0 # Datapoint 2 is worse

        if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '>':
            if data2['Value_x'][i] < data2['Value_y'][i]:
                fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
                fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
                data3['fps'][i] =  np.append(fps_x, fps_y)
                data3['Y'][i] = 1 # Datapoint 2 is better
            else:
                continue # Unsure which datapoint is better - don't keep pair

        if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '<':
            if data2['Value_x'][i] > data2['Value_y'][i]:
                fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
                fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
                data3['fps'][i] =  np.append(fps_x, fps_y)
                data3['Y'][i] = 0 # Datapoint 2 is worse
            else:
                continue # Unsure which datapoint is better - don't keep pair

        if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '=':
            if data2['Value_x'][i] > data2['Value_y'][i]:
                fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
                fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
                data3['fps'][i] =  np.append(fps_x, fps_y)
                data3['Y'][i] = 0 # Datapoint 2 is worse
            else:
                continue # Unsure which datapoint is better - don't keep pair

        if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '>':
            continue # Unsure which datapoint is better - don't keep pair

        if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '<':
            if data2['Value_x'][i] > data2['Value_y'][i]:
                fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
                fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
                data3['fps'][i] =  np.append(fps_x, fps_y)
                data3['Y'][i] = 0 # Datapoint 2 is worse
            else:
                continue # Unsure which datapoint is better - don't keep pair

        if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '=':
            if data2['Value_x'][i] < data2['Value_y'][i]:
                fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
                fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
                data3['fps'][i] =  np.append(fps_x, fps_y)
                data3['Y'][i] = 1 # Datapoint 2 is better
            else:
                continue # Unsure which datapoint is better - don't keep pair

        if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '>':
            if data2['Value_x'][i] < data2['Value_y'][i]:
                fps_x =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_x'][i]), 2)
                fps_y =  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data2['SMILES_y'][i]), 2)
                data3['fps'][i] =  np.append(fps_x, fps_y)
                data3['Y'][i] = 1 # Datapoint 2 is better
            else:
                continue # Unsure which datapoint is better - don't keep pair

        if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '<':
            continue # Unsure which datapoint is better - don't keep pair


  data3 = data3[data3['Y'].notna()] # Remove anything without values
  data3.index = range(len(data3)) # Reindex
  data3['Y']=data3['Y'].astype('int') # Ensure values are interpreted as integers
  return data3
  
  
# Function to make pairs, determine if the pair improves or not, and remove values where this is unknown
# This version specifically gives predictive confidence based on how large the differences are
def classify_improvement_CTRL_normalized(data): 
  data2 = pd.merge(data, data, how='cross') # Make Pairs
  data3 = pd.DataFrame(columns=['Predictions'], index=range(len(data2))) # For final results

  for i in range(len(data2)):
    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '=':
      data3["Predictions"][i] = data2['Predictions_y'][i] - data2['Predictions_x'][i]

    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '>':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["Predictions"][i] = data2['Predictions_y'][i] - data2['Predictions_x'][i]
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '<':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["Predictions"][i] = data2['Predictions_y'][i] - data2['Predictions_x'][i]
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["Predictions"][i] = data2['Predictions_y'][i] - data2['Predictions_x'][i]
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '>':
      continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '<':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["Predictions"][i] = data2['Predictions_y'][i] - data2['Predictions_x'][i]
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["Predictions"][i] = data2['Predictions_y'][i] - data2['Predictions_x'][i]
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '>':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["Predictions"][i] = data2['Predictions_y'][i] - data2['Predictions_x'][i]
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '<':
      continue # Unsure which datapoint is better - don't keep pair


  data3 = data3[data3['Predictions'].notna()] # Remove anything without values
  data3.index = range(len(data3)) # Reindex
  data3['Predictions'] = (data3['Predictions'] - data3['Predictions'].min()) / (data3['Predictions'].max() - data3['Predictions'].min())    # Normalize 
  return data3
