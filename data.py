from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os, cv2, torch, random, numpy

'''
The dataset for this 
This is implementation of dataset and dataloading classes of pytorch.
see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
for how to load in datasets.
'''

class CatsVSDogsDataset(Dataset):

    def __init__(self, root_dir='./data', train=True, transform=None):
        if train:
            root_dir = os.path.join(root_dir, 'train')
        else:
            root_dir = os.path.join(root_dir, 'test')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if 'cat' in image_path:
            label = 0
        else:
            label = 1
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if isinstance(self.output_size, int):
            image = cv2.resize(image, dsize=(self.output_size, self.output_size))
        else:
            dx, dy = self.output_size
            image = cv2.resize(image, dsize=(dx, dy))
        image = image / 255
        return {'image': image, 'label': label}

class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        assert isinstance(image, numpy.ndarray)
        assert isinstance(label, int)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)  # only if gray-scale
        # image = image.permute(2, 0, 1)  # only if color
        label = torch.tensor(label, dtype=torch.long)
        return {'image': image, 'label': label}

def show_image(image, label):
    print(label)
    if label == 0:
        label = 'cat'
    else:
        label = 'dog'
    plt.title('label: {}'.format(label))
    plt.imshow(image, cmap='gray')

def visualize_dataset(dataset):
    fig = plt.figure()
    for i in range(len(dataset)):
        rand = random.randint(0, len(dataset))
        sample = dataset[rand]
        print(rand, sample['image'].shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(rand))
        ax.axis('off')
        show_image(**sample)
        if i == 3:
            plt.show()
            break

# run this file to see that the dataset is working
if __name__ == "__main__":
    dataset = CatsVSDogsDataset()
    visualize_dataset(dataset)