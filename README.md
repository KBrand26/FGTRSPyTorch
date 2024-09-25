# FGTRS PyTorch
The experiments in this repoitory consisted of training various CNNs for the morphological classification of radio sources. These experiments investigated the impact of rotational variations on model performance, as well as whether guiding models to extract informative features during classification has any benefts.

# Data
Prior to conducting experiments the dataset has to be downloaded from:
```
https://zenodo.org/records/7645530
```
and extracted into the **data/** directory.

Once the dataset has been downloaded, the **prepare_data.py** script in the **src/** directory can be executed to prepare the data for model ingestion and to create a holdout test set.

# Project layout
  - **data/**: Contains the FRGMRC data, as well as a PyTorch Lightning dataset and datamodules which will be responsible for preparing and feeding the data into the various models during the experiments.
  - **models/**: Contains the model architectures that will be used in the experiments.
  - **src/**: Contains the scripts necessary for conducting the experiments.

For more information, the reader is referred to the READMEs in each of the subdirectories. For instructions regarding suggested usage, refer to the README in the src/ directory.