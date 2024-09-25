import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchsummary import summary
import pandas as pd
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
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

class SCNN(pl.LightningModule):
    """
    Standard Convolution Neural Network that does not make use of any form of feature guidance.
    """
    def __init__(self, run, filters=64, nconv=2, lr=0.0001):
        """Initializes the CNN. 

        Args:
            run (int): Indicates the current run number.
            filters (int, optional): The number of filters to start with in first layer. Defaults to 64.
            nconv (int, optional): The number of convolutional blocks to use. Defaults to 2.
            lr (float, optional): The learning rate to use. Defaults to 0.0001.
        """
        super().__init__()
        # Save hyperparameters in case loading necessary.
        self.save_hyperparameters()
        
        layers = []
        # Create layers
        layers.append(
            nn.Conv2d(
                in_channels=1, # Monochrome
                out_channels=filters,
                kernel_size=14,
                padding="same"
            )
        )
        layers.append(nn.ReLU())
        for i in range(nconv):
            # Add convolutional blocks
            prev_filters = filters
            filters <<= 1 # Double the number of filters
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                ) 
            )
            layers.append(
                nn.Conv2d(
                    in_channels=prev_filters, # Previous amount of filters
                    out_channels=filters,
                    kernel_size=7,
                    padding="same"
                )
            )
            layers.append(nn.ReLU())
            layers.append(
                nn.Conv2d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=7,
                    padding="same"
                )
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.MaxPool2d(
                kernel_size=2
            ) 
        )
        layers.append(nn.Flatten())
        layers.append(
            nn.Linear(
                in_features=82944,
                out_features=150
            )
        )
        layers.append(nn.ELU())
        layers.append(nn.Dropout1d(0.5))
        layers.append(
            nn.Linear(
                in_features=150,
                out_features=75
            )
        )
        layers.append(nn.ELU())
        layers.append(nn.Dropout1d(0.5))
        
        # Create model containing the constructed layers
        self.network = nn.Sequential(*layers)
        # Keep final layer separate to make it easy to save final hidden layer's outputs
        self.final_layer = nn.Linear(
            in_features=75,
            out_features=4
        )
        # Initialize weights
        self.network.apply(initialize_weights)
        self.final_layer.apply(initialize_weights)
        
        # Metrics
        self.accuracy = MulticlassAccuracy(4, average='micro')
        self.precision = MulticlassPrecision(4, average=None)
        self.recall = MulticlassRecall(4, average=None)
        self.f1 = MulticlassF1Score(4, average=None)
        
    def forward(self, x):
        """Executes a forward pass of the given data through the model.

        Args:
            x (Tensor): The batch to pass through the model.

        Returns:
            Tuple: A tuple containing the final feature layer's output and the class probabilities.
        """
        x = self.network(x)
        return x, self.final_layer(x)
    
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

    def training_step(self, train_batch, batch_idx):
        """Defines what should happen during each step during training

        Args:
            train_batch (Tensor): A batch of images to pass through the model.
            batch_idx (int): The index of the given batch

        Returns:
            float: The loss for the given batch
        """
        img, lbl, _ = train_batch
        _, preds = self.forward(img)
        
        # Calculate metrics for gradient descent
        with_bn_loss = self.entropy_loss(preds=preds, target=lbl)
        
        # Calculate metrics for model evaluation
        self.eval()
        with torch.no_grad():
            # Calculate metrics
            _, preds = self.forward(img)
            without_bn_loss = self.entropy_loss(preds=preds, target=lbl)
            acc = self.accuracy(preds, lbl)
            prec = self.precision(preds, lbl)
            rec = self.recall(preds, lbl)
            f1 = self.f1(preds, lbl)
            
            # Log metrics
            self.log('train_loss', without_bn_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_prec', prec.mean())
            self.log('train_recall', rec.mean())
            self.log('train_f1', f1.mean(), prog_bar=True, on_step=False, on_epoch=True)
        
        self.train()
        return with_bn_loss
    
    def validation_step(self, val_batch, batch_idx):
        """Defines what should happen during each step during validation.

        Args:
            val_batch (Tensor): A batch of images to use for validation of the model.
            batch_idx (int): The index of the given batch.
        """
        img, lbl, _ = val_batch
        _, preds = self.forward(img)
        
        # Calculate metrics
        loss = self.entropy_loss(preds=preds, target=lbl)
        acc = self.accuracy(preds, lbl)
        prec = self.precision(preds, lbl)
        rec = self.recall(preds, lbl)
        f1 = self.f1(preds, lbl)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_prec', prec.mean())
        self.log('val_recall', rec.mean())
        self.log('val_f1', f1.mean(), prog_bar=True, on_step=False, on_epoch=True)
    
        return loss
    
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
        img, lbl, _ = test_batch
        feats, preds = self.forward(img)
        
        # Calculate metrics
        loss = self.entropy_loss(preds=preds, target=lbl)
        acc = self.accuracy(preds, lbl)
        prec = self.precision(preds, lbl)
        rec = self.recall(preds, lbl)
        f1 = self.f1(preds, lbl)
        self.cm = confusion_matrix(
            y_true=lbl.detach().cpu().numpy(),
            y_pred=preds.argmax(axis=1).detach().cpu().numpy(),
            normalize="true"
        )*100

        self.log('test_loss', loss)
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
        
        self.metric_df = pd.DataFrame({
            "Training Time": [self.fit_duration],
            "Loss": [loss.detach().cpu().numpy()],
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
            "Compact F1": [f1[0].detach().cpu().numpy()],
            "Bent F1": [f1[1].detach().cpu().numpy()],
            "FRI F1": [f1[2].detach().cpu().numpy()],
            "FRII F1": [f1[3].detach().cpu().numpy()],
        })
        self.metric_df['Model'] = 'SCNN'
        self.metric_df['Run'] = self.hparams.run
        
        if hasattr(self, 'feats'):
            self.feats = torch.cat((self.feats, feats))
            self.lbls = torch.cat((self.lbls, lbl))
        else:
            self.feats = feats
            self.lbls = lbl
        
if __name__ == "__main__":
    # Check whether class creates model as expected
    cnn = SCNN(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn.network.to(device)
    summary(model, (1, 150, 150))