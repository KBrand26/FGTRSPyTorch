import torch
import pytorch_lightning as pl
import numpy as np
import sys
# tell interpreter where to look
sys.path.insert(0,"data/datamodules")
from StandardDataModule import StandardDataModule
from model_constructor import construct_scnn, load_scnn, construct_mhcnn, load_mhcnn, construct_mcnn, load_mcnn
import os
import pandas as pd
import wandb
from matplotlib import pyplot as plt
import seaborn as sns
from prepare_data import probe_dir

class ExperimentDriver():
    MODEL_CONSTRUCTORS = {
        "SCNN": construct_scnn,
        "MhCNN": construct_mhcnn,
        "MCNN": construct_mcnn
    }
    MODEL_LOADERS = {
        "SCNN": load_scnn,
        "MhCNN": load_mhcnn,
        "MCNN": load_mcnn
    }
    ROTATION_LABEL = {
        "standardise": "standardised_",
        "augment": "augmented_",
        "none": "",
    }
    FILTER_LABEL = {
        "MASC": "MASC_filter_",
        "MAANF": "MAANF_filter_"
    }
    
    def __init__(self, runs, weight="even", model="SCNN", rotation_method="augment", filter_method="MAANF", epochs=100):
        """Initializes the experiment driver, which will be responsible for conducting each step of the various experiments.

        Args:
            runs (int): The number of runs to conduct.
            weight (str, optional): The weight setting to use when working with guided architecturs. Options include even, main and aux.
                Defaults to "even".
            model (str, optional): Indicates which model architecture to use. Options include SCNN, MhCNN and MCNN. Defaults to "SCNN".
            rotation_method (str, optional): Indicates which method to use to address rotations.
                Options include "none", "standardise" and "augment". Defaults to "augment".
            filter_method (str, optional): Indicates which noise filtering approach to employ during certain preprocessing steps.
                Options include MASC and MAANF. Defaults to "MAANF".
            epochs (int, optional): Indicates the maximum amount of epochs for which to train models. Defaults to 100.
        """
        self.runs = runs
        self.weight = weight
        self.model = model
        self.rotation = rotation_method
        self.filter = filter_method
        self.epochs = epochs
    
    def setup_parameters(self):
        """
        Sets the necessary seeds and prepares the datamodule for the experiments.
        """
        pl.seed_everything(42)
        # Ensures that the output directories exists
        probe_dir("outputs/confusion/")
        probe_dir("outputs/plots/")
        probe_dir("outputs/dataframes/")
        # Create datamodule
        self.dm = StandardDataModule(batch_size=64, seed=42, rotation_method=self.rotation, filter_method=self.filter)
    
    def print_run_start_info(self):
        """
        Simple utility function to announce the start of a run
        """
        print("=============================================================================================")
        print(f"Starting run {self.current_run}.")
        print("=============================================================================================")

    def train_model(self, train_loader, val_loader, loggers, callbacks):
        """Trains the given model using the given data.

        Args:
            model (LightningModule): The model to train.
            train_loader (DataLoader): The dataloader to use for training data.
            val_loader (DataLoader): The dataloader to use for validation data.
            loggers (List): The set of loggers to use during training.
            callbacks (List): The callbacks to use during training.
        """
        self.trainer = pl.Trainer(
            accelerator="auto",
            deterministic=False,
            max_epochs=self.epochs,
            log_every_n_steps=10,
            logger=loggers,
            callbacks=callbacks
        )
        self.trainer.fit(self.cnn, train_dataloaders=train_loader, val_dataloaders=val_loader)
        fit_duration = self.cnn.fit_end - self.cnn.fit_start
        self.cnn = self.MODEL_LOADERS[self.model](callbacks[1].best_model_path)
        self.cnn.fit_duration = fit_duration

    def execute_model_training(self):
        """
        This function is used to conduct model training
        """
        # Fetch dataloaders
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()
        
        # Determine the name of the run
        rot_label = self.ROTATION_LABEL[self.rotation]
        filt_label = self.FILTER_LABEL[self.filter]
        self.group_name = f"{self.model}_{self.weight}_{rot_label}{filt_label}"
        self.run_name = f"{self.group_name}run{self.current_run}"
        
        # Construct the models with appropriate parameters
        self.cnn, callbacks = self.MODEL_CONSTRUCTORS[self.model](self.current_run, self.run_name, weight=self.weight)
        
        # Create loggers for experiment tracking
        csv_logger = pl.loggers.CSVLogger("lightning_logs", name=self.run_name)
        self.wandb_logger = pl.loggers.WandbLogger(project="FGTRSTorch", group=self.group_name, name=self.run_name, \
            log_model=True, reinit=True)
        loggers = [self.wandb_logger, csv_logger]

        self.train_model(train_loader, val_loader, loggers, callbacks)
    
    def test_model(self, test_loader):
        """Controls the behaviour for testing a model

        Args:
            test_loader (DataLoader): A dataloader containing the samples from the holdout test set.
        """
        self.trainer.test(model=self.cnn, dataloaders=test_loader)
        self.update_metric_dataframes()
    
    def execute_testing_loop(self):
        """
        This function handles the logic for a loop through the testing data.
        """
        # Set model to eval mode to ensure correct testing behaviour
        self.cnn.eval()
        test_loader = self.dm.test_dataloader()
        
        self.test_model(test_loader)

    def update_metric_dataframes(self):
        """
        Update the dataframe containing the metric information for all of the experimental runs
        """
        if hasattr(self, 'metric_df'):
            self.metric_df = pd.concat([self.metric_df, self.cnn.metric_df], ignore_index=True)
        else:
            self.metric_df = self.cnn.metric_df
    
    def save_confusion_matrix(self):
        """
        Controls the storing of individual confusion matrices after a run.
        """
        if hasattr(self, 'cm'):
            self.cm += self.cnn.cm
        else:
            self.cm = self.cnn.cm
        np.save(f"outputs/confusion/{self.run_name}_cm.npy", self.cnn.cm)
    
    def plot_aggregate_confusion_matrix(self):
        """
        Generates a plot for the aggregate confusion matrix of the experiment.
        """
        self.cm = self.cm/self.runs
        np.save(f"outputs/confusion/{self.group_name}_avg_cm.npy", self.cm)
        
        # Inspired by https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#evaluate_metrics
        plt.figure(figsize=(5,5))
        cls_names = ['Bent', 'Compact', 'FRI', 'FRII']
        ax = sns.heatmap(self.cm, annot=True, fmt=".2f", xticklabels=cls_names, yticklabels=cls_names)
        for t in ax.texts:
            t.set_text(t.get_text() + " %")
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig(f"outputs/plots/{self.group_name}_cm.eps", format='eps', bbox_inches="tight", pad_inches=0)
        plt.close()
    
    def execute_experiment(self):
        """
        This function serves as the main driver for the execution of the experiment.
        """
        self.setup_parameters()
        
        for r in range(1, self.runs+1):
            self.execute_run(r)
        
            if r != self.runs:
                # Setup for next run
                self.dm.create_next_train_val_sets()
                del self.cnn
                print(f"Completed run {r}")
                self.wandb_logger.finalize("success")
                wandb.finish()
            else:
                print("Experiment concluded")
                self.metric_df.to_csv(f"outputs/dataframes/{self.group_name}_metric_df.csv", index=False)
                self.plot_aggregate_confusion_matrix()
                self.wandb_logger.finalize("success")
                wandb.finish()
    
    def execute_run(self, run):
        """Executes a single experimental run

        Args:
            run (int): Indicates the run number.
        """
        self.current_run = run
        
        self.print_run_start_info()
        self.execute_model_training()
        self.execute_testing_loop()
        self.save_confusion_matrix()

if __name__ == "__main__":
    # =====================
    # Rotation experiments
    # =====================
    # No rotational adjustments
    runner = ExperimentDriver(runs=10, model="SCNN", rotation_method="none", filter_method="MAANF", epochs=100)
    runner.execute_experiment()
    
    # Standardisation
    runner = ExperimentDriver(runs=10, model="SCNN", rotation_method="standardise", filter_method="MAANF", epochs=100)
    runner.execute_experiment()
    
    # # Augment
    runner = ExperimentDriver(runs=10, model="SCNN", rotation_method="augment", filter_method="MAANF", epochs=100)
    runner.execute_experiment()
    
    # ==========================
    # Feature guided experiments
    # ==========================
    # MhCNN
    runner = ExperimentDriver(runs=10, model="MhCNN", rotation_method="none", filter_method="MAANF", epochs=100, weight="even")
    runner.execute_experiment()
    
    runner = ExperimentDriver(runs=10, model="MhCNN", rotation_method="none", filter_method="MAANF", epochs=100, weight="main")
    runner.execute_experiment()
    
    runner = ExperimentDriver(runs=10, model="MhCNN", rotation_method="none", filter_method="MAANF", epochs=100, weight="aux")
    runner.execute_experiment()
    
    # MCNN
    runner = ExperimentDriver(runs=10, model="MCNN", rotation_method="none", filter_method="MAANF", epochs=100, weight="even")
    runner.execute_experiment()
    
    runner = ExperimentDriver(runs=10, model="MCNN", rotation_method="none", filter_method="MAANF", epochs=100, weight="main")
    runner.execute_experiment()
    
    runner = ExperimentDriver(runs=10, model="MCNN", rotation_method="none", filter_method="MAANF", epochs=100, weight="aux")
    runner.execute_experiment()