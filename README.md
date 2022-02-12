# Quantised Robofish Network
----

This repository is a demonstration of a network that is pretrained for emitter localisation of merFISH images from the molecular oncology department at the BC Cancer Centre under Dr. Aparicio. 

In addition, it demonstrates the impact of quantisation methods available through PyTorch on this network in terms of detection accuracy and the Jaccard Index. The quantisation methods demonstrated are:

1. Dynamic
1. Static
1. Quantisation Aware Training (TBD)

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

### Training
----
The network can be trained using the `train.py` file. 

```
python src/test.py --help
usage: train.py [-h] [--data_path DATA_PATH] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--gpu GPU] [--recompile_ds RECOMPILE_DS]
                [--print_rate PRINT_RATE] [--lr LR] [--grad_clip GRAD_CLIP]

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

State dictionaries are saved at the end of each epoch when there is a decrease in the best validation loss in the `mlflow` run.

### Testing
---

To test the network, `src/test.py` can be run to produce images demonstrating the accuracy and jaccard index of the emitter localisation.

```
python src/test.py
usage: test.py [-h] [--data_path DATA_PATH] [--state_dict STATE_DICT]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the training data (default: ./Data/50/)
  --state_dict STATE_DICT
                        Path to the training data (default: ./best_state_dicts/feb_11/state_dict.pth)
```


### Testing the Quantised Models
----

To test the performance of Quantised models versus the floating point alternative, the `quant_test.py` can be run. The performance is compared with accuracy, jaccard index, the time for inference, and the size of the model parameters. 

```
python src/quant_test.py
usage: quant_test.py [-h] [--qbackend QBACKEND] [--data_path DATA_PATH] [--state_dict STATE_DICT]

optional arguments:
  -h, --help            show this help message and exit
  --qbackend QBACKEND   Quantisation Backend (default: qnnpack)
  --data_path DATA_PATH
                        Path to the training data (default: ./Data/50/)
  --state_dict STATE_DICT
                        Path to the training data (default: ./best_state_dicts/feb_11/state_dict.pth)

```

NOTE: The `qnnbackend` is utilised for mobile (typically ARM) backends. This is what is needed for working on M1 processors. For x86 processors with AVX2 support or higher, pass `fbgemm` into `--qbackend` backend.

## Results
----


|Model Type| Accuracy | Jaccard Index | Average Inference Time (5 runs) | Size (KB)| Size Reduction|
|----|----|----|----|----|----|
|Floating Point|0.925|0.879|0.0996s|19744.379||
|Dynamic Quant|0.925|0.879|0.0741s|19744.379| 1.00x|
|Static Quant|0.927|0.857|0.060s|5118.575|3.86x|

<!-- 
|Model Type| Accuracy | Jaccard Index | Average Inference Time (5 runs) | Size (KB)| Size Reduction|
|----|----|----|----|----|----|
|Floating Point|0.9248876081607481|0.8787601233670466|0.09964547157287598s|19744.379||
|Dynamic Quant|0.9248876081607481|0.8787601233670466|0.07407598495483399s|19744.379| 1.00x|
|Static Quant|0.927303934691360|0.8567536397414837|0.059986209869384764s|5118.575|3.86x| -->

