import os
import torch 
import torchvision
import torch.nn as nn
import numpy as np
import pickle
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torchviz import make_dot
from torch.utils.data import Dataset, DataLoader



#Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 80
learning_rate = 0.001

# Hyper parameters
num_epochs = 80
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Image preprocessing modules
pre_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32)])

post_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 


# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', 
                                           train=True, 
                                           transform=None,  
                                           download=False)
transforms1 = transforms.Compose([transforms.ToTensor()])
class RotatedSet(Dataset, angle):
    def __init__(self, root_dir, transforms = None):
        self.root_dir = root_dir
        self.transforms = transforms
        l, k = train_dataset[0]
        print(k)
        self.xs = train_dataset.data
        self.ys = train_dataset.targets
        # self.x = []
        # self.y = []
        # for b in range(1, 6):
        #     f = os.path.join(self.root_dir, 'data_batch_%d' % (b,))
        #     X, Y = self.load_CIFAR_batch(f)
        #     self.x.append(X)
        #     self.y.append(Y)
        # print("xs",self.xs)
        # input()
        # print("x",self.x)
        # input()
        rot_img = l.rotate(15)
        plt.figure()
        plt.imshow(rot_img)
        plt.savefig("dummy_name.png")
        # print("ys",self.ys)
        # print("y",self.y)
        input()
        self.x_tr = self.rotate_img(np.concatenate(self.xs), 90)
        self.y_tr = self.ys
        # self.y_tr = np.concatenate(self.y)
        # print("y",self.y_tr)
        # input()
        self.length = len(self.x_tr)
        print("self.length",self.length)
        input()

    def __getitem__(self, idx):
        img = self.x_tr[idx]
        img = self.transforms(img)
        return img, self.y_tr[idx]

    def load_CIFAR_batch(self, filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = np.transpose(np.reshape(X,(10000, 3, 32,32)), (0,2,3,1))
            Y = np.array(Y)
            return X, Y
    def rotate_img_new(self, img):
        print("555")
    def rotate_img(self, img, rot):
        if rot == 0:                                      # 0 degrees rotation
            return img
        elif rot == 90:                                   # 90 degrees rotation
            return img.swapaxes(-2, -1)[..., ::-1, :]
        elif rot == 180:                                  # 180 degrees rotation
            return img[..., ::-1, ::-1]
        elif rot == 270:                                  # 270 degrees rotation / or -90
            return img.swapaxes(-2, -1)[..., ::-1]
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __len__(self):
        return self.length
rot_set = RotatedSet('./data/cifar10/cifar-10-batches-py', transforms1)
input()

test_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# STL10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='data/STL10/', 
#                                            train=True, 
#                                            transform=transform,  
#                                            download=False)

# test_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', 
#                                           train=False, 
#                                           transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
model = models.resnet50().to(device)
#model = model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')

def get_rotated_item():
