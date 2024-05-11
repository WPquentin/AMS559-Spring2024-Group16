import torch

import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# normalization, pointwise gaussian
class UnitGaussianNormalizer:
    def __init__(self, x, eps=0.00001, reduce_dim=[0], verbose=True):
        super().__init__()
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze(0)
        self.std = torch.std(x, reduce_dim, keepdim=True).squeeze(0)
        self.eps = eps
        
        if verbose:
            print(f'UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}.')
            print(f'   Mean and std of shape {self.mean.shape}, eps={eps}')

    def encode(self, x):
        # x = x.view(-1, *self.sample_shape)
        x -= self.mean
        x /= (self.std + self.eps)
        # x = (x.view(-1, *self.sample_shape) - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x.view(self.sample_shape) * std) + mean
        # x = x.view(-1, *self.sample_shape)
        x *= std
        x += mean

        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def count_params(model):
    """Returns the number of parameters of a PyTorch model"""
    return sum([p.numel()*2 if p.is_complex() else p.numel() for p in model.parameters()])


def wandb_login(api_key_file='../config/wandb_api_key.txt'):
    with open(api_key_file, 'r') as f:
        key = f.read()
    wandb.login(key=key)

def set_wandb_api_key(api_key_file='../config/wandb_api_key.txt'):
    import os
    try:
        os.environ['WANDB_API_KEY']
    except KeyError:
        with open(api_key_file, 'r') as f:
            key = f.read()
        os.environ['WANDB_API_KEY'] = key.strip()

def get_wandb_api_key(api_key_file='../config/wandb_api_key.txt'):
    import os
    try:
        return os.environ['WANDB_API_KEY']
    except KeyError:
        with open(api_key_file, 'r') as f:
            key = f.read()
        return key.strip()

def visualize(epoch, visualize_samples, trainSamples = True):
    fig = plt.figure(figsize=(10, 10))
    row = 3
    col = 4
    for index in range(row):
        # Input x
        x = visualize_samples['x'][index]
        # Ground-truth
        y = visualize_samples['y'][index]
        # Model prediction
        out = visualize_samples['out'][index]
        # start point
        if x.shape[-1] > out.shape[-1]:
            h = 1.0 / (x.shape[-1]//2)
            coordinate = torch.linspace(h, 1.0, x.shape[-1]//2)
            x = x[:,:x.shape[-1]//2]
        else:
            h = 1.0 / (x.shape[-1])
            coordinate = torch.linspace(h, 1.0, x.shape[-1])
            x = x[:,:x.shape[-1]]

        out = out.squeeze(0)

        ax = fig.add_subplot(row, col, index*col + 1)
        ax.plot(coordinate, x.squeeze().numpy())
        if index == 0: 
            ax.set_title('Input x')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())
    
        ax = fig.add_subplot(row, col, index*col + 2)
        ax.plot(coordinate, y.squeeze().numpy(), label = 'Ground-truth y')
        ax.plot(coordinate, out.squeeze().detach().numpy(), label = 'Model prediction')
        if index == 0: 
            ax.set_title('GT y and model prediction')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())


        ax = fig.add_subplot(row, col, index*col + 3)
        ax.plot(coordinate, out.squeeze().detach().numpy())
        if index == 0: 
            ax.set_title('Model prediction')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())

        ax = fig.add_subplot(row, col, index * col + 4)
        ax.plot(coordinate, np.abs(y.squeeze().numpy() - out.squeeze().detach().numpy()))
        if index == 0: 
            ax.set_title('abs(y - prediction)')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())

    title =  'epoch_' + str(epoch) + '_train' if trainSamples else '_test'
    fig.suptitle('Inputs, ground-truth output and prediction.' + title , y=0.98)
    plt.tight_layout()
    if trainSamples:
        wandb.log({'epoch': epoch, 'training samples visualize': wandb.Image(fig)})
    else:
        wandb.log({'epoch': epoch, 'testing samples visualize': wandb.Image(fig)})
    plt.close()


def visualize_test(visualize_samples, bc_index = 1):
    fig = plt.figure(figsize=(10, 10))
    row = 3
    col = 4
    for index in range(row):
        x = visualize_samples['x'][index]
        # Ground-truth
        y = visualize_samples['y'][index]
        # Model prediction
        out = visualize_samples['out'][index]
        # start point
        if x.shape[-1] > out.shape[-1]:
            h = 1.0 / (x.shape[-1]//2)
            coordinate = torch.linspace(h, 1.0, x.shape[-1]//2)
            x = x[:,:x.shape[-1]//2]
        else:
            h = 1.0 / (x.shape[-1])
            coordinate = torch.linspace(h, 1.0, x.shape[-1])
            x = x[:,:x.shape[-1]]

        out = out.squeeze(0)

        ax = fig.add_subplot(row, col, index * col + 1)
        ax.plot(coordinate, x.squeeze().numpy())
        if index == 0: 
            ax.set_title('Input x')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())

        ax = fig.add_subplot(row, col, index * col + 2)
        ax.plot(coordinate, y.squeeze().numpy(), label = 'Ground-truth y')
        ax.plot(coordinate, out.squeeze().detach().numpy(), label = 'Model prediction')
        if index == 0: 
            ax.set_title('GT y and model prediction')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())

        ax = fig.add_subplot(row, col, index * col + 3)
        ax.plot(coordinate, out.squeeze().detach().numpy())
        if index == 0: 
            ax.set_title('Model prediction')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())

        ax = fig.add_subplot(row, col, index * col + 4)
        ax.plot(coordinate, np.abs(y.squeeze().numpy() - out.squeeze().detach().numpy()))
        if index == 0: 
            ax.set_title('abs(y - prediction)')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        
    fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
    plt.tight_layout()
    
    wandb.log({"Final Visualization on Test": wandb.Image(fig), 'bc_index': bc_index})
