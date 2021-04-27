# %%
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
# from tensorboardX import SummaryWriter
# from torchviz import make_dot
from torch.utils.data import Dataset, DataLoader

#Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# %%
# # CIFAR-10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', 
#                                            train=True, 
#                                            transform=None,  
#                                            download=True)

# indices = np.random.choice(50000, 10000)
# train_dataset = [train_dataset[index] for index in indices]

transforms1 = transforms.Compose([transforms.ToTensor()])
class RotatedSet(Dataset):
    def __init__(self,mode, root_dir, angle=30, transforms = None):
        self.root_dir = root_dir
        self.transforms = transforms
        if mode =="train":
            self.dataset = train_dataset
        elif mode == "test":
            self.dataset = test_dataset
        l, k = self.dataset[0]
        print(k)
        rot_img = l.rotate(angle)
        plt.figure()
        plt.imshow(rot_img)
        plt.savefig("dummy_name.png")
        self.x_tr = [l.rotate(angle) for l,k in self.dataset]
        self.length = len(self.x_tr)
        print("self.length",self.length)
        

    def __getitem__(self, idx):
        img = self.x_tr[idx]
        img = self.transforms(img)
        #return img, self.y_tr[idx]
        return img
    def __len__(self):
        return self.length

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', 
                                           train=True, 
                                           transform=None,  
                                           download=True)
test_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', 
                                          train=False, 
                                          transform=None)
indices1 = np.random.choice(50000, 10000)
indices2 = np.random.choice(10000, 2000)
train_dataset = [train_dataset[index] for index in indices1]
test_dataset = [test_dataset[index] for index in indices2]
# rot_set = RotatedSet('./data/cifar10/cifar-10-batches-py', 30, post_transform)




# STL10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='data/STL10/', 
#                                            train=True, 
#                                            transform=transform,  
#                                            download=False)

# test_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', 
#                                           train=False, 
#                                           transform=transforms.ToTensor())


# %%
def obtain_rotated_image(angle_list, mode):
    lst = []
    for i, angle in enumerate (angle_list):
        print("angle",angle)
        print("label",i+1)
        model = RotatedSet(mode,'./', angle, post_transform)
        for instance in model:
            lst.append((instance, i+1))
    return lst
# %%
train_dataset_rotated = obtain_rotated_image([-30, -10, 0, 10, 30],"train")
test_dataset_rotated = obtain_rotated_image([-30, -10, 0, 10, 30],"test")
print(train_dataset_rotated[0])

# %%
print(len(train_dataset_rotated))
print(len(test_dataset_rotated))
# %%
train_dataset_rotated
# %%
train_dataset_rotated[1][0]

#split data
train_size = int(0.8 * len(train_dataset_rotated))
val_size = len(train_dataset_rotated) - train_size
train_dataset_rotated, val_dataset_rotated = torch.utils.data.random_split(train_dataset_rotated, [train_size, val_size])
print(len(train_dataset_rotated))
print(len(val_dataset_rotated))

# %%
# Data loader
train_loader = DataLoader(dataset=train_dataset_rotated,
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=2)


val_loader = DataLoader(dataset=val_dataset_rotated,
                          batch_size=batch_size, 
                          shuffle=False,
                          num_workers=2)

test_loader = DataLoader(dataset=test_dataset_rotated,
                          batch_size=batch_size, 
                          shuffle=False,
                          num_workers=2)
input()
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("d.png")
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
model = models.resnet50().to(device)
#model = model.to(device)
# # Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # For updating learning rate
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

# # Test the model
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
