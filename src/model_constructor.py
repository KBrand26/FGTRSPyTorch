import pytorch_lightning as pl
import sys
# tell interpreter where to look
sys.path.insert(0, "models/")
from SCNN import SCNN
from MhCNN import MhCNN
from MCNN import MCNN

# Various weight schemes for guided architectures
OUTPUT_WEIGHTS = {
    "even": {
        "main_weight": 0.5,
        "aux_weight": 0.5
    },
    "main": {
        "main_weight": 0.75,
        "aux_weight": 0.25
    },
    "aux": {
        "main_weight": 0.25,
        "aux_weight": 0.75
    },
}

# Learning rates to use for each variant
MHCNN_WEIGHT_LRS = {
    "even": 0.0001,
    "main": 0.00001,
    "aux": 0.0001
}
MCNN_WEIGHT_LRS = {
    "even": 0.00001,
    "main": 0.00001,
    "aux": 0.00001
}

def construct_scnn(run, run_name, lr = 0.0001, weight="even"):
    """Constructs a standard CNN and its necessary callbacks.

    Args:
        run (int): Indicates the current run of the experiment. Necessary for logging purposes.
        run_name (str): An identifier for the current run.
        lr (float, optional): Indicates what learning rate should be used. Defaults to 0.0001.
        weight (str, optional): Placeholder argument to fit signature necessary for other constructors.

    Returns:
        Tuple: A tuple containing the Torch model and its callbacks.
    """
    escb = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", patience=5)
    cpcb = pl.callbacks.ModelCheckpoint(dirpath="model_checkpoints/", filename=f"{run_name}", monitor="val_loss", mode="min")
    callbacks = [escb, cpcb]
    cnn = SCNN(run, lr=lr)
    return cnn, callbacks

def construct_mhcnn(run, run_name, lr = 0.0001, weight="even"):
    """Construct a multiheaded CNN and its necessary callbacks.

    Args:
        run (int): Indicates the current run of the experiment. Necessary for logging purposes.
        run_name (str): An identifier for the current run.
        lr (float, optional): Indicates what learning rate should be used. Defaults to 0.0001.
        weight (str, optional): The weight setting for the model, can be even, aux or main. Defaults to even.

    Returns:
        Tuple: A tuple containing the Torch model and its callbacks.
    """
    lr = MHCNN_WEIGHT_LRS[weight]
    escb = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", patience=5)
    cpcb = pl.callbacks.ModelCheckpoint(dirpath="model_checkpoints/", filename=f"{run_name}", monitor="val_loss", mode="min")
    callbacks = [escb, cpcb]
    cnn = MhCNN(run, lr=lr, **(OUTPUT_WEIGHTS[weight]))
    return cnn, callbacks

def construct_mcnn(run, run_name, lr = 0.00001, weight="even"):
    """Construct a merged CNN and its necessary callbacks.

    Args:
        run (int): Indicates the current run of the experiment. Necessary for logging purposes.
        run_name (str): An identifier for the current run.
        lr (float, optional): Indicates what learning rate should be used. Defaults to 0.0001.
        weight (str, optional): The weight setting for the model, can be even, aux or main. Defaults to even.

    Returns:
        Tuple: A tuple containing the Torch model and its callbacks.
    """
    lr = MCNN_WEIGHT_LRS[weight]
    escb = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", patience=5)
    cpcb = pl.callbacks.ModelCheckpoint(dirpath="model_checkpoints/", filename=f"{run_name}", monitor="val_loss", mode="min")
    callbacks = [escb, cpcb]
    cnn = MCNN(run, lr=lr, **(OUTPUT_WEIGHTS[weight]))
    return cnn, callbacks

def load_scnn(path):
    """Loads a SCNN from a checkpoint.

    Args:
        path (str): The path to the model checkpoint

    Returns:
        SCNN: The loaded SCNN.
    """
    return SCNN.load_from_checkpoint(path)

def load_mhcnn(path):
    """Loads a MhCNN from a checkpoint.

    Args:
        path (str): The path to the model checkpoint

    Returns:
        MhCNN: The loaded MhCNN.
    """
    return MhCNN.load_from_checkpoint(path)

def load_mcnn(path):
    """Loads a MCNN from a checkpoint.

    Args:
        path (str): The path to the model checkpoint

    Returns:
        MCNN: The loaded MCNN.
    """
    return MCNN.load_from_checkpoint(path)