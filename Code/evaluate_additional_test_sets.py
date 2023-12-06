# Imports
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import math
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold

# machine learning metrics
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


# Function to make pairs, determine if the pair improves or not, and remove values where this is unknown
def classify_improvement_scaffolds_keep_relation_keep_values(data): # Specific version that also keeps relations and true values
  data2 = pd.merge(data, data, how='cross') # Make Pairs
  data3 = pd.DataFrame(columns=['SMILES_x', 'SMILES_y', 'Scaffold_x', 'Scaffold_y', 'Value_x', 'Value_y', 'Relation_x', 'Relation_y', 'Y'], index=range(len(data2))) # For final results

  for i in range(len(data2)):
    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3["Scaffold_x"][i] =  data2['Scaffold_x'][i]
        data3["Scaffold_y"][i] =  data2['Scaffold_y'][i]
        data3["Value_x"][i] =  data2['Value_x'][i]
        data3["Value_y"][i] =  data2['Value_y'][i]
        data3["Relation_x"][i] =  data2['Relation_x'][i]
        data3["Relation_y"][i] =  data2['Relation_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3["Scaffold_x"][i] =  data2['Scaffold_x'][i]
        data3["Scaffold_y"][i] =  data2['Scaffold_y'][i]
        data3["Value_x"][i] =  data2['Value_x'][i]
        data3["Value_y"][i] =  data2['Value_y'][i]
        data3["Relation_x"][i] =  data2['Relation_x'][i]
        data3["Relation_y"][i] =  data2['Relation_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse

    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '>':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3["Scaffold_x"][i] =  data2['Scaffold_x'][i]
        data3["Scaffold_y"][i] =  data2['Scaffold_y'][i]
        data3["Value_x"][i] =  data2['Value_x'][i]
        data3["Value_y"][i] =  data2['Value_y'][i]
        data3["Relation_x"][i] =  data2['Relation_x'][i]
        data3["Relation_y"][i] =  data2['Relation_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '=' and data2['Relation_y'][i] == '<':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3["Scaffold_x"][i] =  data2['Scaffold_x'][i]
        data3["Scaffold_y"][i] =  data2['Scaffold_y'][i]
        data3["Value_x"][i] =  data2['Value_x'][i]
        data3["Value_y"][i] =  data2['Value_y'][i]
        data3["Relation_x"][i] =  data2['Relation_x'][i]
        data3["Relation_y"][i] =  data2['Relation_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3["Scaffold_x"][i] =  data2['Scaffold_x'][i]
        data3["Scaffold_y"][i] =  data2['Scaffold_y'][i]
        data3["Value_x"][i] =  data2['Value_x'][i]
        data3["Value_y"][i] =  data2['Value_y'][i]
        data3["Relation_x"][i] =  data2['Relation_x'][i]
        data3["Relation_y"][i] =  data2['Relation_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '>':
      continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '>' and data2['Relation_y'][i] == '<':
      if data2['Value_x'][i] > data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3["Scaffold_x"][i] =  data2['Scaffold_x'][i]
        data3["Scaffold_y"][i] =  data2['Scaffold_y'][i]
        data3["Value_x"][i] =  data2['Value_x'][i]
        data3["Value_y"][i] =  data2['Value_y'][i]
        data3["Relation_x"][i] =  data2['Relation_x'][i]
        data3["Relation_y"][i] =  data2['Relation_y'][i]
        data3['Y'][i] = 0 # Datapoint 2 is worse
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '=':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3["Scaffold_x"][i] =  data2['Scaffold_x'][i]
        data3["Scaffold_y"][i] =  data2['Scaffold_y'][i]
        data3["Value_x"][i] =  data2['Value_x'][i]
        data3["Value_y"][i] =  data2['Value_y'][i]
        data3["Relation_x"][i] =  data2['Relation_x'][i]
        data3["Relation_y"][i] =  data2['Relation_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '>':
      if data2['Value_x'][i] < data2['Value_y'][i]:
        data3["SMILES_x"][i] =  data2['SMILES_x'][i]
        data3["SMILES_y"][i] =  data2['SMILES_y'][i]
        data3["Scaffold_x"][i] =  data2['Scaffold_x'][i]
        data3["Scaffold_y"][i] =  data2['Scaffold_y'][i]
        data3["Value_x"][i] =  data2['Value_x'][i]
        data3["Value_y"][i] =  data2['Value_y'][i]
        data3["Relation_x"][i] =  data2['Relation_x'][i]
        data3["Relation_y"][i] =  data2['Relation_y'][i]
        data3['Y'][i] = 1 # Datapoint 2 is better
      else:
        continue # Unsure which datapoint is better - don't keep pair

    if data2['Relation_x'][i] == '<' and data2['Relation_y'][i] == '<':
      continue # Unsure which datapoint is better - don't keep pair


  data3 = data3[data3['Y'].notna()] # Remove anything without values
  data3.index = range(len(data3)) # Reindex
  data3['Y']=data3['Y'].astype('int') # Ensure values are interpreted as integers
  return data3

#######################
### SMILES Analysis ###
#######################

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

models = ['RandomForest', 'ChemProp50', 'XGBoost', 
          'DeltaClassifierLiteOnlyEquals', 'DeltaClassifierLiteAllData', 'DeltaClassifierLite', 
          'DeepDeltaClassifierOnlyEquals5', 'DeepDeltaClassifierAllData5', 'DeepDeltaClassifier5']

for model in models:

    all_scores_demilitarized = pd.DataFrame(columns=['Dataset', 'Accuracy', 'F1', 'AUC'])
    all_scores_no_same_molecule_pairs = pd.DataFrame(columns=['Dataset', 'Accuracy', 'F1', 'AUC'])    
    all_scores_Gatorless = pd.DataFrame(columns=['Dataset', 'Accuracy', 'F1', 'AUC'])    
    nonmatching_scaffolds_demilitarized = pd.DataFrame(columns=['Dataset', 'Accuracy', 'F1', 'AUC'])
    matching_scaffolds_demilitarized = pd.DataFrame(columns=['Dataset', 'Accuracy', 'F1', 'AUC'])
    all_scores_Gatorless_demilitarized = pd.DataFrame(columns=['Dataset', 'Accuracy', 'F1', 'AUC']) 


    # Evaluate scoring without same molecular pairs for all Datasets
    for name in properties:
        dataframe = pd.read_csv("../Datasets/{}-Curated.csv".format(name))
        predictions = pd.read_csv('../Results/{}/{}_{}_1.csv'.format(model, name, model)).T 
        predictions.columns =['True', 'Delta']

        mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
        scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]
        data = pd.DataFrame(data={'SMILES': mols, 'Scaffold': scaffolds, 'Relation': dataframe['Relation'], 'Value': dataframe['Value']})

        # Emulate previous train-test splits (Only need the test split so the train will be ignored)
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        datapoint_x = []
        datapoint_y = []
        scaffold_x = []
        scaffold_y = []
        Relation_x = []
        Relation_y = []
        Value_x = []
        Value_y = []

        for train_index, test_index in cv.split(data):
            test_df = data[data.index.isin(test_index)]
            pair_subset_test = classify_improvement_scaffolds_keep_relation_keep_values(test_df)
            datapoint_x += [pair_subset_test.SMILES_x]
            datapoint_y += [pair_subset_test.SMILES_y]
            scaffold_x += [pair_subset_test.Scaffold_x]
            scaffold_y += [pair_subset_test.Scaffold_y]
            Value_x += [pair_subset_test.Value_x]
            Value_y += [pair_subset_test.Value_y]
            Relation_x += [pair_subset_test.Relation_x]
            Relation_y += [pair_subset_test.Relation_y]


        datapoints = pd.DataFrame(data={'SMILES_X':  np.concatenate(datapoint_x), 'SMILES_Y':  np.concatenate(datapoint_y),
                                        'Scaffold_x':  np.concatenate(scaffold_x), 'Scaffold_y':  np.concatenate(scaffold_y),
                                        'Value_x':  np.concatenate(Value_x), 'Value_y':  np.concatenate(Value_y),
                                        'Relation_x':  np.concatenate(Relation_x), 'Relation_y':  np.concatenate(Relation_y)})

        # Add the actual deltas and predicted deltas
        trues = predictions["True"] > 0.5
        trues = [float(i) for i in trues]
        datapoints['True'] = trues

        Deltas = predictions['Delta']
        Deltas = [float(i) for i in Deltas]
        datapoints['Delta'] = Deltas
        DeltaClass = predictions['Delta'] > 0.5
        DeltaClass = [float(i) for i in DeltaClass]
        datapoints['DeltaClass'] = DeltaClass

        # Demilitarize all the datapoints
        demilitarized = datapoints.drop(datapoints[abs(datapoints['Value_x'] - datapoints['Value_y']) < 0.1].index)

        # Run Stats
        Accuracy = accuracy(demilitarized["True"], (demilitarized['DeltaClass']))
        F1 = f1_score(demilitarized["True"], (demilitarized['DeltaClass']))
        AUC = rocauc(demilitarized["True"], (demilitarized['Delta']))

        scoring = pd.DataFrame({'Dataset': [name],
                                'Accuracy': [round(Accuracy, 4)], 'F1': [round(F1, 4)], 'AUC': [round(AUC, 4)]})

        all_scores_demilitarized = pd.concat([all_scores_demilitarized, scoring])

        # Exclude all same molecule pairs
        No_SMP = datapoints[datapoints['SMILES_X'] != datapoints['SMILES_Y']]

        # Run Stats
        Accuracy = accuracy(No_SMP["True"], (No_SMP['DeltaClass']))
        F1 = f1_score(No_SMP["True"], (No_SMP['DeltaClass']))
        AUC = rocauc(No_SMP["True"], (No_SMP['Delta']))

        scoring = pd.DataFrame({'Dataset': [name],
                                'Accuracy': [round(Accuracy, 4)], 'F1': [round(F1, 4)], 'AUC': [round(AUC, 4)]})

        all_scores_no_same_molecule_pairs = pd.concat([all_scores_no_same_molecule_pairs, scoring])

        # Only keep datapoints that are exact ('=' relations, i.e., 'gatorless') and no same molecular pairs
        only_exact = No_SMP[(No_SMP['Relation_x'] == '=') & (No_SMP['Relation_y'] == '=')]

        # Run Stats
        Accuracy = accuracy(only_exact["True"], (only_exact['DeltaClass']))
        F1 = f1_score(only_exact["True"], (only_exact['DeltaClass']))
        AUC = rocauc(only_exact["True"], (only_exact['Delta']))

        scoring = pd.DataFrame({'Dataset': [name],
                                'Accuracy': [round(Accuracy, 4)], 'F1': [round(F1, 4)], 'AUC': [round(AUC, 4)]})

        all_scores_Gatorless = pd.concat([all_scores_Gatorless, scoring])


        # Only keep demilitarized nonmatching scaffolds
        nonmatching_demilitarized = demilitarized[demilitarized['Scaffold_x'] != demilitarized['Scaffold_y']]

        # Run Stats
        Accuracy = accuracy(nonmatching_demilitarized["True"], (nonmatching_demilitarized['DeltaClass']))
        F1 = f1_score(nonmatching_demilitarized["True"], (nonmatching_demilitarized['DeltaClass']))
        AUC = rocauc(nonmatching_demilitarized["True"], (nonmatching_demilitarized['Delta']))

        scoring = pd.DataFrame({'Dataset': [name],
                                'Accuracy': [round(Accuracy, 4)], 'F1': [round(F1, 4)], 'AUC': [round(AUC, 4)]})

        nonmatching_scaffolds_demilitarized = pd.concat([nonmatching_scaffolds_demilitarized, scoring])


        # Only keep demilitarized matching scaffolds
        matching_demilitarized = demilitarized[demilitarized['Scaffold_x'] == demilitarized['Scaffold_y']]

        # Run Stats
        Accuracy = accuracy(matching_demilitarized["True"], (matching_demilitarized['DeltaClass']))
        F1 = f1_score(matching_demilitarized["True"], (matching_demilitarized['DeltaClass']))
        AUC = rocauc(matching_demilitarized["True"], (matching_demilitarized['Delta']))

        scoring = pd.DataFrame({'Dataset': [name],
                                'Accuracy': [round(Accuracy, 4)], 'F1': [round(F1, 4)], 'AUC': [round(AUC, 4)]})

        matching_scaffolds_demilitarized = pd.concat([matching_scaffolds_demilitarized, scoring])


        # Only keep datapoints that are exact ('=' relations, i.e., 'gatorless') and demilitarized
        only_exact_demilitarized = demilitarized[(demilitarized['Relation_x'] == '=') & (demilitarized['Relation_y'] == '=')]

        # Run Stats
        Accuracy = accuracy(only_exact_demilitarized["True"], (only_exact_demilitarized['DeltaClass']))
        F1 = f1_score(only_exact_demilitarized["True"], (only_exact_demilitarized['DeltaClass']))
        AUC = rocauc(only_exact_demilitarized["True"], (only_exact_demilitarized['Delta']))

        scoring = pd.DataFrame({'Dataset': [name],
                                'Accuracy': [round(Accuracy, 4)], 'F1': [round(F1, 4)], 'AUC': [round(AUC, 4)]})

        all_scores_Gatorless_demilitarized = pd.concat([all_scores_Gatorless_demilitarized, scoring])


    all_scores_demilitarized.to_csv("{}_Demilitarized.csv".format(model), index = False) # Save general demilitarized results
    all_scores_no_same_molecule_pairs.to_csv("{}_No_SMP.csv".format(model), index = False) # Save results with no same molecule pairs
    all_scores_Gatorless.to_csv("{}_No_SMP_Gatorless.csv".format(model), index = False) # Save gatorless (only exact) results with no same molecule pairs
    nonmatching_scaffolds_demilitarized.to_csv("{}_nonmatching_scaffolds_demilitarized.csv".format(model), index = False) # Save Non-matching Scaffolds following demilitarization
    matching_scaffolds_demilitarized.to_csv("{}_matching_scaffolds_demilitarized.csv".format(model), index = False) # Save matching Scaffolds following demilitarization
    all_scores_Gatorless_demilitarized.to_csv("{}_demilitarized_Gatorless.csv".format(model), index = False) # Save gatorless (only exact) results with demilitarization