This repository contains code produced by Damir Korenčić and Ivan Grubišić
during participation of the IRB-NLP Team on the Semeval 2022 CODWOE challenge: https://codwoe.atilf.fr/
The paper describing the system can be found here: https://arxiv.org/abs/2205.06840

Parts of the code are based on the challenge organizer's code:
https://github.com/TimotheeMickus/codwoe


**** Repository structure

- mlalgobuild package
code of the models (models_core package), training and test scripts,
and utility code for machine learning process
defmod_combine contains code for cobining the results of two defmod models (main and fallback model)

- lang_resources package
code for building sentencepiece tokenizers and glove vectors

- data_analysis package
utilities and code for data analysis and plots,
as well as for transformations of the datasets (glosses within the dataset)


**** Setting up

Setup the environment using either codwoe-conda-setup.sh script, or the docker file in the organizers' repo.
Then create file named setting.py from settings-template.py
and point the variables defining the resource folders to existing folder locations.

- dataset_folder
raw data in .json format resides under settings.dataset_folder folder
Subfolders of this folder contain different dataset versions - see --vocab_subdir cmdline param.
These subfolders' names define the semantics of the parameters that define the vocab,
which is the base for all the models - therefore the names must be synchronized across all the deployments.

- built resources
Glove and sentencepiece models are constructed automatically from a set of parameters,
and cached in cache folders (defined in settings.py) for subsequent loading.
If the construction code is changed, then cache folders should be cleaned,
plus old trained torch models will not necessarily work the same.
-- Glove
Glove relies on an external tool that is accessed using cmdline.
First run setup_glove.sh, from the lang_resources folder - it will download
and make the tool, and the executable will be stored under settings.glove_folder
!Important - script build_glove_vectors.sh must be given executable permissions (chmod +x)


**** Running the experiments

The code is located in the mlalgobuild package.

The main entry point that controls all the model-related actions (train, predict) is main.py
All the parameters are described at start, in the code defining the argparse.ArgumentParser

The defmod_train.sh, defmod_predict.sh, revdict_train.sh, and revdict_predict.sh
execute the main.py script for the specific tasks, and contain examples of using the main.py.

- Defmod experiments
-- dataset transformation
Use the orginizers' original datasets (https://codwoe.atilf.fr/) and transform
the using the data_analysis.transformations.createDsetV1 method.
The new dataset will be saved in a dataset subfolder named 'dset_v1',
you can change this name but then you must change it in execution scripts - see for example defmod_batch.sh.
-- language resource
Sentencepiece and glove models will be built automatically if the
environment and glove executable are setup as described earlier.
-- execution
defmod_master.sh and defmod_batch.sh are used for batch execution of training tasks via defmod_train.sh.
defmod_master.sh is the entry point with a definition of a batch run,
while defmod_batch.sh defines load/save folders and adapts parameters for main.sh
These scripts were used to build the models for the paper.
You can use the defmod_train.sh and defmod_predict.sh directly, but consult
defmod_master.sh and especially defmod_batch.sh to get the parameters right.

- Revdict experiments
-- lowercasing the dataset
Use the data_analysis.transformations.createDsetOrigLcase method to transform the original datasets.
If you change the name of the lowercased dataset form 'orig_lc', also change it in the revdict execution scripts.