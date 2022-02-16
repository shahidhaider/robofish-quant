# Quantised Robofish Network
----

This repository is a demonstration of a network that is pretrained for emitter localisation of merFISH images from the molecular oncology department at the BC Cancer Centre under Dr. Aparicio. 

In addition, it demonstrates the impact of quantisation methods available through PyTorch on this network in terms of detection accuracy and the Jaccard Index. The quantisation methods demonstrated are:

1. Dynamic
1. Static
1. Quantisation Aware Training

NOTE: All the instructions and environment here to run this network locally on an M1 CPU computer. 

## Usage
----

Prepare the environment

```
 conda env create -f environment.yaml
 conda activate robofish-nn `
```

NOTE: To run this on azure and track with mlflow 

```
 conda env create -f environment_azure.yaml
 conda activate robofish-azure-nn `
```
While you the necessary files are here for you to remotely train on azure, you will be missing the credentials to do so with the defauly parameters. Replace the `config.json` and the computer target in the `run-train.py` files to get your run to work on AzureML. The `upload_data.py` requires similar precaution.

### Training
----
The network can be trained using the `train.py` file. 

```
python src/train.py --help
usage: train.py [-h] [--data_path DATA_PATH] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--gpu GPU]
                [--recompile_ds RECOMPILE_DS] [--print_rate PRINT_RATE] [--lr LR] [--grad_clip GRAD_CLIP]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the training data (default: None)
  --epochs EPOCHS       Number of Epochs to train (default: None)
  --batch_size BATCH_SIZE
                        Size of batch (default: 16)
  --gpu GPU             GPU: 1 for gpu, 0 for cpu (default: 0)
  --recompile_ds RECOMPILE_DS
                        Recache the dataset. 1 to recache, 0 to use cached (default: 0)
  --print_rate PRINT_RATE
                        The rate of batch samples to print loss functions (default: 10)
  --lr LR               Learning Rate (default: 0.001)
  --grad_clip GRAD_CLIP
                        Gradient Clip (default: 0.01)
``` 
Example data for training is stored in `Data\50.tgz`. Unzip this and point to the directory in the `--data_path` argument in `train.py` 

Models and state dictionaries are saved at the end of each epoch when there is a decrease in the best validation loss in the `mlflow` run.
### Quantization Aware Training
Quantization Aware Training can be done here as well by running the `quant_train.py` file. 

```
python src/quant_train.py --help
usage: quant_train.py [-h] [--data_path DATA_PATH] [--epochs EPOCHS]
                      [--batch_size BATCH_SIZE] [--gpu GPU]
                      [--recompile_ds RECOMPILE_DS] [--print_rate PRINT_RATE]
                      [--lr LR] [--grad_clip GRAD_CLIP] [--qbackend QBACKEND]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the training data (default: None)
  --epochs EPOCHS       Number of Epochs to train (default: None)
  --batch_size BATCH_SIZE
                        Size of batch (default: 16)
  --gpu GPU             GPU: 1 for gpu, 0 for cpu (default: 0)
  --recompile_ds RECOMPILE_DS
                        Recache the dataset. 1 to recache, 0 to use cached
                        (default: 0)
  --print_rate PRINT_RATE
                        The rate of batch samples to print loss functions
                        (default: 10)
  --lr LR               Learning Rate (default: 0.001)
  --grad_clip GRAD_CLIP
                        Gradient Clip (default: 0.01)
  --qbackend QBACKEND   Quant Backend (default: qnnpack)
  ```
### Testing
---

To test the network, `src/test.py` can be run to produce images demonstrating the accuracy and jaccard index of the emitter localisation.

```
python src/test.py
usage: test.py [-h] [--data_path DATA_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the training data (default: ./Data/50/)
 
```


### Testing the Quantised Models
----

To test the performance of Quantised models versus the floating point alternative, the `quant_test.py` can be run. The performance is compared with accuracy, jaccard index, the time for inference, and the size of the model parameters. 

```
python src/quant_test.py
usage: quant_test.py [-h] [--qbackend QBACKEND] [--data_path DATA_PATH] 

optional arguments:
  -h, --help            show this help message and exit
  --qbackend QBACKEND   Quantisation Backend (default: qnnpack)
  --data_path DATA_PATH
                        Path to the training data (default: ./Data/50/)
 
```

NOTE: The `qnnbackend` is utilised for mobile (typically ARM) backends. This is what is needed for working on M1 processors. For x86 processors with AVX2 support or higher, pass `fbgemm` into `--qbackend` backend.

## Results
----

The robofish model was trained over 30 epochs with the same learning rate, gradient clipping, and optimizer in both the floating point and a quantization aware form. A manual seed was set to ensure similar random weight and bias initialisation in the model. 


|Model Type| Average Inference Time (5 runs) | Size (KB)| Size Reduction|
|----|----|----|----|----|----|
|Floating Point|0.0996s|19744.379||
|Dynamic Quant|00.0741s|19744.379| 1.00x|
|Static Quant|0.060s|4977.777|3.97x|
|QAT|0.060s|4977.777|3.97x|

|Model Type| Pos Accuracy (>0.5) | Pos Jaccard Index (>0.5) | Barcode Accuracy>0.9 |  Barcode Jaccard Index (>0.9)|
|----|----|----|----|----| ----|
|Floating Point|0.923|0.898| 0.877 | 0.731|
|Dynamic Quant|0.923|0.898| 0.877 | 0.731|
|Static Quant|0.914|0.890|0.847|0.683|
|QAT| 0.924|0.901|0.809|0.668|

<!-- 
|Model Type| Accuracy | Jaccard Index | Average Inference Time (5 runs) | Size (KB)| Size Reduction|
|----|----|----|----|----|----|
|Floating Point|0.9248876081607481|0.8787601233670466|0.09964547157287598s|19744.379||
|Dynamic Quant|0.9248876081607481|0.8787601233670466|0.07407598495483399s|19744.379| 1.00x|
|Static Quant|0.927303934691360|0.8567536397414837|0.059986209869384764s|5118.575|3.86x| -->

