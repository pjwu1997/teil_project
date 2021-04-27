## this is an util that extracts a pytorch model
## and output it's n-th layer weights

# ref: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.cl1 = nn.Linear(25, 60)
#         self.cl2 = nn.Linear(60, 16)
#         self.fc1 = nn.Linear(16, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
        
#     def forward(self, x):
#         x = F.relu(self.cl1(x))
#         x = F.relu(self.cl2(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.log_softmax(self.fc3(x), dim=1)
#         return x


# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook


# model = MyModel()
# model.fc2.register_forward_hook(get_activation('fc2'))
# x = torch.randn(1, 25)
# output = model(x)
# print(activation['fc2'])
#%%
import torch
import torch.nn as nn
from collections import OrderedDict

def extract_layer(model, layer_name):
    method = getattr(model, layer_name)
    #layer = model[layer_name]
    weight = method.weight.data
    bias = method.weight.data
    #print(weight.flatten())
    return torch.cat((weight.flatten(), bias.flatten()))

if __name__ == '__main__':
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1,20,5)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(20,64,5)),
        ('relu2', nn.ReLU())
    ]))
    extract_layer(model, 'conv1')

# %%
