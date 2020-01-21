# cvd-binary-classifier-demo
Kaggle cats v. dogs binary classifier using convolutional neural networks in PyTorch.
Dataset can be found in this [Kaggle Link](https://www.kaggle.com/c/dogs-vs-cats/).
The pre-trained model state dict can be downloaded [here](https://drive.google.com/open?id=1twFkUCyEM5fP1ATRL7On3fqGrP1Vuxx6).

## Usage
1. Install torch, torchvision, opencv-python
2. Download the model state dict [here](https://drive.google.com/open?id=1twFkUCyEM5fP1ATRL7On3fqGrP1Vuxx6).
3. Load the model state dict into a new instance of the model class from model.py.
```
model = ConvNet().to(device)   # add 'cpu' if there is no gpu
model.load_state_dict(torch.load('path_to_state_dict'))
model.eval()
```

## Results

The classifier during training reached 99.0%+ accuracy with validation set. Not sure how it will perform with testing set, due to the testing set from Kaggle is lacking labels. Will output accuracy over the whole 25,000 training images later.

## Graphs
![alt text](https://github.com/rae0924/cvd-binary-classifier-demo/blob/master/save/epochs_vs_loss.png)
![alt text](https://github.com/rae0924/cvd-binary-classifier-demo/blob/master/save/epochs_vs_accuracy.png)

