import data
from model import ConvNet
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def test_validation():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ConvNet().to(device)
    dataset = data.CatsVSDogsDataset(train=True, 
        transform=transforms.Compose([
            data.Rescale(100),
            data.ToTensor()
        ]))
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    criterion = torch.nn.NLLLoss()
    with torch.no_grad():
        batch = iter(data_loader).next()
        images, labels = batch['image'], batch['label']
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        accuracy = (outputs.exp().argmax(dim=1) == labels).sum().float() / labels.shape[0]
        loss = criterion(outputs, labels)
        print(accuracy, loss)

def test_model():
    return  # waiting implementation 

if __name__ == "__main__":
    test_validation()