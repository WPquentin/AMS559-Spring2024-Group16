"""
Training a neural operator on diffusion
========================================
"""
# %%
# 
import torch
import wandb
import datetime
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
from neuralop.models import TFNO, FNO1d_with_BC
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small, load_burgers, load_diffusion_pt, load_diffusion_pt_withoutBC
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss, weighted_LpLoss

def bp():
    import pdb;pdb.set_trace()

device = 'cpu'
# read a parameter, transform it into string and attach to data_path
parser = argparse.ArgumentParser(description='train fno with bc.')

parser.add_argument('-const', '--const', type=float, default=0.01, help='constant boundary conditon')
parser.add_argument('-use_wandb', '--use_wandb', type=bool, default=False, help='whether use wandb')

args = parser.parse_args()
argConst = args.const
wandb_log = args.use_wandb

label = str(argConst).split('.')[0] + str(argConst).split('.')[1]
path_name = 'data_dict_constBC_'+ label + '.pickle'

data_path = 'C:\\Users\\jizhang\\Desktop\\projects\\ai4science\\electrochemistry\\NumericalDiffusionScheme-master\\Code_radial_normlized'
data_path = data_path + '\\' + path_name

# %%
# Loading the diffusion dataset
train_loader, test_loaders = load_diffusion_pt(
        n_train=90, batch_size=5, test_batch_size=10, data_path=data_path
)


# %%
# We create a tensorized FNO model

model = FNO1d_with_BC(n_modes_height=16, in_channels=1, hidden_channels=32, projection_channels=64, factorization=None, rank=0.42)
model = model.to(device)

n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Creating the losses
l2loss = weighted_LpLoss(d=1, p=2)
h1loss = H1Loss(d=1)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
wandb_log = True
if wandb_log:
    wandb.init(project = "operator_diffusion", name = datetime.datetime.now().strftime("%Y%m%d%H%M") + 'FNO_bc_constBC_' + label, config = {'const': argConst})
trainer = Trainer(model, n_epochs=50,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=wandb_log,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset
output_encoder = None
trainer.train(train_loader, test_loaders,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)


# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders.dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))
    # start point
    h = 1.0 / (x.shape[-1]//2)
    coordinate = torch.linspace(h, 1.0, x.shape[-1]//2)
    x = x[:,:x.shape[-1]//2]
    out = out.squeeze(0)

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.plot(coordinate, x.squeeze().numpy())
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.plot(coordinate, y.squeeze().numpy())
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.plot(coordinate, out.squeeze().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])
    
fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()

# save the parameters to a file
para_path = '.\\fno_bc_para' + label + '.pth'
torch.save(model.state_dict(), para_path)
new_model = FNO1d_with_BC(n_modes_height=16, in_channels=1, hidden_channels=32, projection_channels=64, factorization=None, rank=0.42)
new_model.load_state_dict(torch.load(para_path))

if wandb_log:
    wandb.log({"Visualization": wandb.Image(fig)})
    wandb.finish()
