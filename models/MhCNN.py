import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchsummary import summary
import pandas as pd
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, BinaryF1Score
from torchmetrics import MeanSquaredError
from time import time
from sklearn.metrics import confusion_matrix
import numpy as np

def initialize_weights(layer):
    """Initializes the weights of a given layer

    Args:
        layer (object): The PyTorch layer to initialize.
    """
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

class MhCNN(pl.LightningModule):
    """
    Multihead Convolution Neural Network that has an additional head for extracting certain features. Shares a convolutional base
    with the head that does classification.
    """
    def __init__(self, run, filters=64, nconv=2, lr=0.0001, main_weight=0.5, aux_weight=0.5):
        """Initializes the CNN.

        Args:
            run (int): Indicates the current run number.
            filters (int, optional): The number of filters to start with in first layer. Defaults to 64.
            nconv (int, optional): The number of convolutional blocks to use. Defaults to 2.
            lr (float, optional): The learning rate to use. Defaults to 0.001.
            main_weight (float, optional): The weight used for the main head's loss.
            aux_weight (float, optional): The weight used for the auxiliary head's loss.
        """
        super().__init__()
        # Save hyperparameters in case loading necessary.
        self.save_hyperparameters()
        
        conv_layers = []
        # Create layers
        conv_layers.append(
            nn.Conv2d(
                in_channels=1, # Monochrome
                out_channels=filters,
                kernel_size=14,
                padding="same"
            )
        )
        conv_layers.append(nn.ReLU())
        for i in range(nconv):
            # Add convolutional blocks
            prev_filters = filters
            filters <<= 1 # Double the number of filters
            conv_layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                ) 
            )
            conv_layers.append(
                nn.Conv2d(
                    in_channels=prev_filters, # Previous amount of filters
                    out_channels=filters,
                    kernel_size=7,
                    padding="same"
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(
                nn.Conv2d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=7,
                    padding="same"
                )
            )
            conv_layers.append(nn.ReLU())
        conv_layers.append(
            nn.MaxPool2d(
                kernel_size=2
            ) 
        )
        conv_layers.append(nn.Flatten())
        self.conv_base = nn.Sequential(*conv_layers)
        
        # Main head
        main_layers = []
        main_layers.append(
            nn.Linear(
                in_features=82944,
                out_features=150
            )
        )
        main_layers.append(nn.ELU())
        main_layers.append(nn.Dropout1d(0.5))
        main_layers.append(
            nn.Linear(
                in_features=150,
                out_features=75
            )
        )
        main_layers.append(nn.ELU())
        main_layers.append(nn.Dropout1d(0.5))
        self.main_head = nn.Sequential(*main_layers)
        # Keep final layer separate to make it easy to save 2nd last layers' outputs
        self.main_out = nn.Linear(
            in_features=75,
            out_features=4
        )
        
        # Aux head
        aux_layers = []
        aux_layers.append(
            nn.Linear(
                in_features=82944,
                out_features=150
            )
        )
        aux_layers.append(nn.ELU())
        aux_layers.append(nn.Dropout1d(0.5))
        aux_layers.append(
            nn.Linear(
                in_features=150,
                out_features=75
            )
        )
        aux_layers.append(nn.ELU())
        aux_layers.append(nn.Dropout1d(0.5))
        self.aux_head = nn.Sequential(*aux_layers)
        # Create the neurons responsible for the various feature outputs
        self.bent_out = nn.Linear(
            in_features=75,
            out_features=1
        )
        self.fr_out = nn.Linear(
            in_features=75,
            out_features=1
        )
        self.core_count_out = nn.Linear(
            in_features=75,
            out_features=1
        )
        self.core_ratio_out = nn.Linear(
            in_features=75,
            out_features=1
        )
        
        # Initialize weights
        self.conv_base.apply(initialize_weights)
        self.main_head.apply(initialize_weights)
        self.main_out.apply(initialize_weights)
        self.aux_head.apply(initialize_weights)
        self.bent_out.apply(initialize_weights)
        self.fr_out.apply(initialize_weights)
        self.core_count_out.apply(initialize_weights)
        self.core_ratio_out.apply(initialize_weights)
        
        # Metrics
        self.accuracy = MulticlassAccuracy(4, average='micro')
        self.precision = MulticlassPrecision(4, average=None)
        self.recall = MulticlassRecall(4, average=None)
        self.f1 = MulticlassF1Score(4, average=None)
        self.bin_f1 = BinaryF1Score(average=None)
        self.mse = MeanSquaredError()
        
    def forward(self, x):
        """Executes a forward pass of the given data through the model.

        Args:
            x (Tensor): The batch to pass through the model.

        Returns:
            Tuple: A tuple containing the final feature layer's output from the main head, the main head
                output and the auxiliary head output.
        """
        # Pass through conv base
        x = self.conv_base(x)
        # Pass through main head
        main_head_x = self.main_head(x)
        main_out_x = self.main_out(main_head_x)
        # Pass through aux head
        aux_head_x = self.aux_head(x)
        return main_head_x, main_out_x, self.bent_out(aux_head_x), self.fr_out(aux_head_x), self.core_count_out(aux_head_x), self.core_ratio_out(aux_head_x)
    
    def configure_optimizers(self):
        """Initializes the optimizer to use for this autoencoder.

        Returns:
            Object: The initialized optimizer.
        """
        optimizer = optim.NAdam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def entropy_loss(self, preds, target):
        """Calculate the cross-entropy loss between the predicted and actual class labels.

        Args:
            preds (Tensor): The class probabilities predicted by the model.
            target (Tensor): The actual class labels.

        Returns:
            Tensor: The cross-entropy loss.
        """
        return nn.CrossEntropyLoss()(preds, target)

    def binary_entropy_loss(self, preds, target):
        """Calculate the binary cross-entropy loss between the predicted and actual bent labels.

        Args:
            preds (Tensor): The bent probabilities predicted by the model.
            target (Tensor): The actual bent labels.

        Returns:
            Tensor: The binary cross-entropy loss.
        """
        return nn.BCEWithLogitsLoss()(preds, target)
    
    def mse_loss(self, preds, target):
        """Calculate the mean square error between the predicted and actual values

        Args:
            preds (Tensor): The predicted values.
            target (Tensor): The true values.

        Returns:
            Tensor: The mean squared error between the predicted and actual values.
        """
        return nn.MSELoss()(preds, target)

    def training_step(self, train_batch, batch_idx):
        """Defines what should happen during each step during training

        Args:
            train_batch (Tensor): A batch of images to pass through the model.
            batch_idx (int): The index of the given batch

        Returns:
            float: The loss for the given batch
        """
        img, lbls, feat_lbls = train_batch
        feat_lbls = feat_lbls.float()
        _, class_preds, bent_preds, fr_preds, core_count_preds, core_ratio_preds = self.forward(img)
        
        # Calculate metrics for gradient descent
        class_loss = self.entropy_loss(preds=class_preds, target=lbls)
        bent_entropy_loss = self.binary_entropy_loss(preds=bent_preds, target=feat_lbls[:, 0:1])
        fr_ratio_mse_loss = self.mse_loss(preds=fr_preds, target=feat_lbls[:, 1:2])
        core_count_mse_loss = self.mse_loss(preds=core_count_preds, target=feat_lbls[:, 2:3])
        core_ratio_mse_loss = self.mse_loss(preds=core_ratio_preds, target=feat_lbls[:, 3:4])
        comb_aux_loss = (bent_entropy_loss + fr_ratio_mse_loss + core_count_mse_loss + core_ratio_mse_loss)/4
        final_loss = self.hparams.main_weight*class_loss + self.hparams.aux_weight*comb_aux_loss
        
        # Calculate metrics for model evaluation
        self.eval()
        with torch.no_grad():
            # Calculate metrics
            _, class_preds, bent_preds, fr_preds, core_count_preds, core_ratio_preds = self.forward(img)
            eval_class_loss = self.entropy_loss(preds=class_preds, target=lbls)
            eval_bent_entropy_loss = self.binary_entropy_loss(preds=bent_preds, target=feat_lbls[:, 0:1])
            eval_fr_ratio_mse_loss = self.mse_loss(preds=fr_preds, target=feat_lbls[:, 1:2])
            eval_core_count_mse_loss = self.mse_loss(preds=core_count_preds, target=feat_lbls[:, 2:3])
            eval_core_ratio_mse_loss = self.mse_loss(preds=core_ratio_preds, target=feat_lbls[:, 3:4])
            eval_comb_aux_loss = (eval_bent_entropy_loss + eval_fr_ratio_mse_loss + eval_core_count_mse_loss + eval_core_ratio_mse_loss)/4
            eval_final_loss = self.hparams.main_weight*eval_class_loss + self.hparams.aux_weight*eval_comb_aux_loss
            acc = self.accuracy(class_preds, lbls)
            prec = self.precision(class_preds, lbls)
            rec = self.recall(class_preds, lbls)
            f1 = self.f1(class_preds, lbls)
            bent_f1 = self.bin_f1(bent_preds, feat_lbls[:, 0:1])
            fr_mse = self.mse(fr_preds, feat_lbls[:, 1:2])
            core_count_mse = self.mse(core_count_preds, feat_lbls[:, 2:3])
            core_ratio_mse = self.mse(core_ratio_preds, feat_lbls[:, 3:4])
            
            # Log metrics
            self.log('train_class_loss', eval_class_loss)
            self.log('train_aux_loss', eval_comb_aux_loss)
            self.log('train_loss', eval_final_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_prec', prec.mean())
            self.log('train_recall', rec.mean())
            self.log('train_f1', f1.mean(), prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_bent_f1', bent_f1)
            self.log('train_fr_mse', fr_mse)
            self.log('train_core_count_mse', core_count_mse)
            self.log('train_core_ratio_mse', core_ratio_mse)
        
        self.train()
        return final_loss
    
    def validation_step(self, val_batch, batch_idx):
        """Defines what should happen during each step during validation.

        Args:
            val_batch (Tensor): A batch of images to use for validation of the model.
            batch_idx (int): The index of the given batch.
        """
        img, lbls, feat_lbls = val_batch
        _, class_preds, bent_preds, fr_preds, core_count_preds, core_ratio_preds = self.forward(img)
        
        # Calculate metrics
        class_loss = self.entropy_loss(preds=class_preds, target=lbls)
        bent_entropy_loss = self.binary_entropy_loss(preds=bent_preds, target=feat_lbls[:, 0:1])
        fr_ratio_mse_loss = self.mse_loss(preds=fr_preds, target=feat_lbls[:, 1:2])
        core_count_mse_loss = self.mse_loss(preds=core_count_preds, target=feat_lbls[:, 2:3])
        core_ratio_mse_loss = self.mse_loss(preds=core_ratio_preds, target=feat_lbls[:, 3:4])
        comb_aux_loss = (bent_entropy_loss + fr_ratio_mse_loss + core_count_mse_loss + core_ratio_mse_loss)/4
        final_loss = self.hparams.main_weight*class_loss + self.hparams.aux_weight*comb_aux_loss
        acc = self.accuracy(class_preds, lbls)
        prec = self.precision(class_preds, lbls)
        rec = self.recall(class_preds, lbls)
        f1 = self.f1(class_preds, lbls)
        bent_f1 = self.bin_f1(bent_preds, feat_lbls[:, 0:1])
        fr_mse = self.mse(fr_preds, feat_lbls[:, 1:2])
        core_count_mse = self.mse(core_count_preds, feat_lbls[:, 2:3])
        core_ratio_mse = self.mse(core_ratio_preds, feat_lbls[:, 3:4])
        
        # Log metrics
        self.log('val_class_loss', class_loss)
        self.log('val_aux_loss', comb_aux_loss)
        self.log('val_loss', final_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_prec', prec.mean())
        self.log('val_recall', rec.mean())
        self.log('val_f1', f1.mean(), prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_bent_f1', bent_f1)
        self.log('val_fr_mse', fr_mse)
        self.log('val_core_count_mse', core_count_mse)
        self.log('val_core_ratio_mse', core_ratio_mse)
    
        return final_loss
    
    def on_fit_start(self):
        """
        Defines what should happen at the start of the training loop.
        """
        self.fit_start = time()
    
    def on_fit_end(self):
        """
        Defines what should happen at the end of the training loop.
        """
        self.fit_end = time()
    
    def test_step(self, test_batch, batch_idx):
        """Defines what should happen during each step during testing.

        Args:
            vest_batch (Tensor): A batch of images to use for testing of the model.
            batch_idx (int): The index of the given batch.
        """
        img, lbls, feat_lbls = test_batch
        feats, class_preds, bent_preds, fr_preds, core_count_preds, core_ratio_preds = self.forward(img)
        
        # Calculate metrics
        class_loss = self.entropy_loss(preds=class_preds, target=lbls)
        bent_entropy_loss = self.binary_entropy_loss(preds=bent_preds, target=feat_lbls[:, 0:1])
        fr_ratio_mse_loss = self.mse_loss(preds=fr_preds, target=feat_lbls[:, 1:2])
        core_count_mse_loss = self.mse_loss(preds=core_count_preds, target=feat_lbls[:, 2:3])
        core_ratio_mse_loss = self.mse_loss(preds=core_ratio_preds, target=feat_lbls[:, 3:4])
        comb_aux_loss = (bent_entropy_loss + fr_ratio_mse_loss + core_count_mse_loss + core_ratio_mse_loss)/4
        final_loss = self.hparams.main_weight*class_loss + self.hparams.aux_weight*comb_aux_loss
        acc = self.accuracy(class_preds, lbls)
        prec = self.precision(class_preds, lbls)
        rec = self.recall(class_preds, lbls)
        f1 = self.f1(class_preds, lbls)
        bent_f1 = self.bin_f1(bent_preds, feat_lbls[:, 0:1])
        fr_mse = self.mse(fr_preds, feat_lbls[:, 1:2])
        core_count_mse = self.mse(core_count_preds, feat_lbls[:, 2:3])
        core_ratio_mse = self.mse(core_ratio_preds, feat_lbls[:, 3:4])
        self.cm = confusion_matrix(
            y_true=lbls.detach().cpu().numpy(),
            y_pred=class_preds.argmax(axis=1).detach().cpu().numpy(),
            normalize="true"
        )*100

        self.log('test_class_loss', class_loss)
        self.log('test_aux_loss', comb_aux_loss)
        self.log('test_total_loss', final_loss)
        self.log('test_accuracy', acc)
        self.log('test_bent_prec', prec[0])
        self.log('test_comp_prec', prec[1])
        self.log('test_fri_prec', prec[2])
        self.log('test_frii_prec', prec[3])
        self.log('test_prec', prec.mean())
        self.log('test_bent_rec', rec[0])
        self.log('test_comp_rec', rec[1])
        self.log('test_fri_rec', rec[2])
        self.log('test_frii_rec', rec[3])
        self.log('test_rec', rec.mean())
        self.log('test_bent_f1', f1[0])
        self.log('test_comp_f1', f1[1])
        self.log('test_fri_f1', f1[2])
        self.log('test_frii_f1', f1[3])
        self.log('test_f1', f1.mean())
        self.log('test_bent_shape_f1', bent_f1)
        self.log('test_fr_mse', fr_mse)
        self.log('test_core_count_mse', core_count_mse)
        self.log('test_core_ratio_mse', core_ratio_mse)
        
        self.metric_df = pd.DataFrame({
            "Training Time": [self.fit_duration],
            "Loss": [final_loss.detach().cpu().numpy()],
            "Accuracy": [acc.detach().cpu().numpy()],
            "Precision": [prec.mean().detach().cpu().numpy()],
            "Recall": [rec.mean().detach().cpu().numpy()],
            "F1": [f1.mean().detach().cpu().numpy()],
            "Bent Precision": [prec[0].detach().cpu().numpy()],
            "Compact Precision": [prec[1].detach().cpu().numpy()],
            "FRI Precision": [prec[2].detach().cpu().numpy()],
            "FRII Precision": [prec[3].detach().cpu().numpy()],
            "Bent Recall": [rec[0].detach().cpu().numpy()],
            "Compact Recall": [rec[1].detach().cpu().numpy()],
            "FRI Recall": [rec[2].detach().cpu().numpy()],
            "FRII Recall": [rec[3].detach().cpu().numpy()],
            "Bent F1": [f1[0].detach().cpu().numpy()],
            "Compact F1": [f1[1].detach().cpu().numpy()],
            "FRI F1": [f1[2].detach().cpu().numpy()],
            "FRII F1": [f1[3].detach().cpu().numpy()],
        })
        self.metric_df['Model'] = 'MhCNN'
        self.metric_df['Run'] = self.hparams.run
        
        if hasattr(self, 'feats'):
            self.feats = torch.cat((self.feats, feats))
            self.lbls = torch.cat((self.lbls, lbls))
        else:
            self.feats = feats
            self.lbls = lbls
        
if __name__ == "__main__":
    # Check whether class creates model as expected
    cnn = MhCNN(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn.to(device)
    summary(model, (1, 150, 150))