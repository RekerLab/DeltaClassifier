## Scripts for Running Directly from the Command Line

#### cross_validations_standard.sh
* To call this from the command line in Linux, you would want to go to the directory containing the python file, shell script, and datasets of interests and then type ‘bash cross_validations_standard.sh’. To call different datasets instead of ours in the python script, you would want to replace the names of the datasets called in the python file. 

#### cross_validations_argparse.sh
* In cross_validations_argparse.py, we coded it such that the python file knows this argument is the input file by using either “-i” or “--input_file”. To call this from the command line in linux, you would type ‘bash [shellscript.sh] -i [dataset.csv]’  for example ‘bash cross_validations_ argparse.sh -i CHEMBL1075138-Curated.csv’.

#### cross_validations_array.sh
* In this one, we indicate the directory where the files are, move to that directory, and then call the python file using all datasets that end in ‘-Curated.csv’. To call this from the command line in Linux, you would type ‘bash [shellscript.sh] [pythonscript.py] [directory]’ for example ‘bash cross_validations_array.sh cross_validations_array.py home/user/chemprop’. 

#### external_test.sh
* This script fits the Chemprop model on the entire training dataset using the DeltaClassifier approach and then it predicts on an external test set (that must be pairs of molecules) and adds predictions that compares the two molecules. Within the script, you can specific the training and testing data names.

