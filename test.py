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
        images, labels = batch['image'].to(device), batch['label'].to(device)
        outputs = model(images)
        accuracy = (outputs.exp().argmax(dim=1) == labels).sum().float() / labels.shape[0]
        loss = criterion(outputs, labels)
        print(accuracy, loss)

def test_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ConvNet().to(device)
    model = model.load_state_dict(torch.load('./save/cats_vs_dogs.pth'))
    model.eval()
    dataset = data.CatsVSDogsDataset(train=False,   # requires testing set
        transform=transforms.Compose([
            data.Rescale(100),
            data.ToTensor()
        ]))
    return  # waiting implementation 

if __name__ == "__main__":
    test_validation()