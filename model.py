import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import data, torch, pickle

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=384, out_features=2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x    
    

def train(model, trianing_set, num_epochs=10, device='cpu'):
    train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=training_set, batch_size=1000, shuffle=True, num_workers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    print_every = 100  # print every 100 batches
    running_loss = 0
    train_data = []
    for epoch in range(num_epochs):
        if epoch == 1:
            break
        for idx_batch, batch in enumerate(train_loader):
            images, labels = batch['image'], batch['label']
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (idx_batch+1) % print_every == 0:
                with torch.no_grad():
                    batch = iter(val_loader).next()
                    images, labels = batch['image'], batch['label']
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    equality = (outputs.exp().argmax(dim=1) == labels).sum()
                    accuracy = equality.float() / labels.shape[0]
                    val_loss = criterion(outputs, labels)
                train_data.append([epoch, running_loss/print_every, val_loss, accuracy])
                print(f"epoch: {epoch}, training_loss: {running_loss/print_every}", 
                f"val_loss: {val_loss} val_accuracy: {accuracy}")
                running_loss = 0
        running_loss = 0
    torch.save(model.state_dict(), 'model_stat_dict.pt')
    pickle.dump(train_data, open('training_data.pickle', 'wb'))

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    training_set = data.CatsVSDogsDataset(train=True, 
        transform=transforms.Compose([
            data.Rescale(100),
            data.ToTensor()
        ]))
    model = ConvNet().to(device) 
    train(model, training_set, num_epochs=50, device=device)
    
    # for i_batch, sample_batched in enumerate(train_loader):
    #     if i_batch == 0:
    #         print(sample_batched['image'][0])
    #         data.show_image(sample_batched['image'][0][0], sample_batched['label'][0])
    #         plt.show()
    #         break