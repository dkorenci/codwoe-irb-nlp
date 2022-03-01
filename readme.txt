This repository contains code produced  by Damir Korenčić and Ivan Grubišić 
during participation of the IRB-NLP Team on the Semeval 2022 CODWOE challenge: 
https://codwoe.atilf.fr/

Parts of the code are based on the challenge organizer's code:
https://github.com/TimotheeMickus/codwoe

**** Setting up resources

- !Dataset Folder structure
raw data in .json format resides under settings.dataset_folder folder
Subfolders of this folder contain different dataset versions - see --vocab_subdir cmdline param.
These subfolders' names define the semantics of the parameters that define the vocab,
which is the base for all the models - therefore the names must be synchronized for all deployments.

Glove and sentencepiece models are constructed automatically from a set of parameters,
and cached in a save folder (defined in settings.py) for subsequent loading.
If the construction code is changed beyond the parameter-defined changes,
then save folders should be cleaned, plus old trained torch models will not necessarily work the same.

- Glove
Glove relies on an external tool that is accessed using cmdline.
First run setup_glove.sh, from the lang_resources folder - it will download
and make the tool, and the executable will be stored under settings.glove_folder
!Important - script build_glove_vectors.sh must be given executable permissions (chmod +x)

**** Modelbuilding command line arguments

All the arguments are documented at start of mlalgobuild/main.py
Examples of use are in *_train.sh and *_predict.sh scripts
Important, several parameters defining the dataset modifications
are tied to the model trained but not saved(1) as part of the serialized model,
they must be passed to predict, these are --maxlen and --vocab_* params.
These params are saved separately in hparams.json file in the same folder as the serialized model.
predict() should be modified to load these params.

