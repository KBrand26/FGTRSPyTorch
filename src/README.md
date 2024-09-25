# File descriptions
  - **ExperimentDriver.py**: This is the main script for conducting experiments. It defines all of the behaviour necessary for the various experiments that investigate rotational variations and feature guidance. It can simply be executed to conduct the experiments, as they have already been defined.
  - **FeatureExtractor.py**: Defines the behaviour for a feature extractor object that should be used to extract the auxiliary feature labels for the various radio sources.
  - **Filter.py**: An abstract class that defines what the expected behaviour for a noise filter is.
  - **MAANFFilter.py**: A Filter class that defines the behaviour for the MAANF filter.
  - **MASCFilter.py**: A Filter class that defines the behaviour for the MASC filter.
  - **model_constructor.py**: This file defines how the various CNN architectures should be constructed, as well as how they should be loaded from model checkpoints.
  - **prepare_data.py**: This file contains most of the data preparation and processing logic. It should be used to process the download FRGMRC data and to create a holdout test set prior to experimentation.

# Suggested usage:
1. Execute the **prepare_data.py** script to process FRGMRC and create a holdout test set.
2. Execute the **ExperimentDriver.py** script to conduct the rotational variation and feature guidance experiments.