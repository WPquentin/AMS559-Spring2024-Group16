"""
Training a neural operator on diffusion
========================================
"""
import torch
import wandb
import datetime
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
import argparse
from pathlib import Path
import matplotlib.ticker as ticker
from neuralop.models import TFNO, FNO1d_with_BC, FNO1d
from neuralop import Trainer
from neuralop.datasets import load_diffusion_pt, load_diffusion_pt_withoutBC
from neuralop.utils import count_params, visualize_test
from neuralop import LpLoss, H1Loss, weighted_LpLoss
from neuralop.sam import SAM
import hydra
from omegaconf import DictConfig, OmegaConf

def bp():
    import pdb;pdb.set_trace()

@hydra.main(version_base = None, config_path="conf", config_name="config")
# @hydra.main(config_name=".//conf//config")
def train_model(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = {k: (v if v != 'None' else None) for k, v in cfg_dict.items()}
    cfg = OmegaConf.create(cfg_dict)

    argConst = cfg.const
    wandb_log = cfg.use_wandb
    using_BC = cfg.use_BC
    epochs = cfg.epochs
    learning_rate = cfg.learning_rate
    weight_decay = cfg.weight_decay
    n_modes = cfg.n_modes
    domain_padding = cfg.domain_padding
    whether_shuffle = cfg.whether_shuffle
    norm_type = cfg.norm_type
    train_ratio = cfg.train_ratio
    split_train_test_randomly = cfg.split_train_test_randomly
    split_bc = cfg.split_bc
    device = 'cpu'

    label = str(argConst).split('.')[0] + str(argConst).split('.')[1]
    path_name = 'data_dict_constBC_'+ label + '.pickle'

    data_path = 'C:\\Users\\jizhang\\Desktop\\projects\\ai4science\\electrochemistry\\NumericalDiffusionScheme-master\\Code_radial_normlized'
    data_path = data_path + '\\' + path_name
    label += '_modes' + str(n_modes) + '_epochs' + str(epochs)

    # Loading the diffusion dataset and create the model
    if not using_BC:
        print("Using FNO without BC!")
        train_loader, test_loaders = load_diffusion_pt_withoutBC(
                train_ratio=train_ratio, batch_size=5, test_batch_size=3, data_path=data_path, split_train_test_randomly = split_train_test_randomly, whether_shuffle= whether_shuffle
        )
        model = FNO1d(n_modes_height = n_modes, in_channels=1, hidden_channels=32, projection_channels=64, factorization=None, rank=0.42, domain_padding=domain_padding, norm=norm_type)
    else:
        print("Using FNO BC!")
        train_loader, test_loaders = load_diffusion_pt(
                train_ratio=train_ratio, batch_size=5, test_batch_size=3, data_path=data_path, split_train_test_randomly = split_train_test_randomly, whether_shuffle= whether_shuffle
        )
        model = FNO1d_with_BC(n_modes_height = n_modes, in_channels=1, hidden_channels=32, projection_channels=64, factorization=None, rank=0.42, domain_padding=domain_padding, split_bc = split_bc, norm=norm_type)

    model = model.to(device)

    n_params = count_params(model)
    print(f'\nOur model has {n_params} parameters.')
    sys.stdout.flush()

    #Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=learning_rate, 
                                    weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= epochs)
    

    # Creating the losses
    l2loss = weighted_LpLoss(d=1, p=2)
    h1loss = H1Loss(d=1)

    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}

    print('\n### MODEL ###\n', model)
    # print('\n### OPTIMIZER ###\n', optimizer)
    # print('\n### SCHEDULER ###\n', scheduler)
    # print('\n### LOSSES ###')
    # print(f'\n * Train: {train_loss}')
    # print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()

    # Create the trainer
    if wandb_log:
        if using_BC:
            wandb.init(project = "operator_diffusion", name = datetime.datetime.now().strftime("%m%d%H%M") + 'FNO_constBC_' + label, config=cfg)
        else:
            wandb.init(project = "operator_diffusion", name = datetime.datetime.now().strftime("%Y%m%d%H%M") + 'FNO_noBC_constBC_' + label, config=cfg)
        wandb.config.update(cfg)
    trainer = Trainer(model, n_epochs=epochs,
                    device=device,
                    mg_patching_levels=0,
                    wandb_log=wandb_log,
                    log_test_interval=3,
                    use_distributed=False,
                    verbose=True)

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

    # Plot the prediction, and compare with the ground-truth
    if wandb_log:
        test_samples = test_loaders.dataset
        test_sample_set = {'x':[], 'y':[], 'out':[]}
        for index in range(3):
            data = test_samples[index]
            # Input x
            x = data['x']
            test_sample_set['x'].append(x)
            # Ground-truth
            y = data['y']
            test_sample_set['y'].append(y)
            # Model prediction
            test_sample_set['out'].append(model(x.unsqueeze(0)))
        visualize_test(test_sample_set)

    # save the parameters to a file
    if using_BC:
        par_path = Path(__file__).resolve().parent.joinpath('para')
        file_name = 'fno_para_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '_' + label + '.pth'
        para_path = par_path.joinpath(file_name)
    else:
        par_path = Path(__file__).resolve().parent.joinpath('para')
        file_name = 'fno_bc_para_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '_' + label + '.pth'
        para_path = par_path.joinpath(file_name)
    
    torch.save(model.state_dict(), para_path)

    if wandb_log:
        wandb.finish()

if __name__ == "__main__":
    train_model()