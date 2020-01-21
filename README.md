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


