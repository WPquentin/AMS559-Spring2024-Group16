import torch
from timeit import default_timer
import wandb
import sys 

import neuralop.mpu.comm as comm

from .patching import MultigridPatching2D
from .losses import LpLoss
from ..utils import visualize

def bp():
    import pdb;pdb.set_trace()

class Trainer:
    def __init__(self, model, n_epochs, wandb_log=True, device=None,
                 mg_patching_levels=0, mg_patching_padding=0, mg_patching_stitching=True,
                 log_test_interval=1, log_output=False, use_distributed=False, verbose=True):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        mg_patching_levels : int, default is 0
            if 0, no multi-grid domain decomposition is used
            if > 0, indicates the number of levels to use
        mg_patching_padding : float, default is 0
            value between 0 and 1, indicates the fraction of size to use as padding on each side
            e.g. for an image of size 64, padding=0.25 will use 16 pixels of padding on each side
        mg_patching_stitching : bool, default is True
            if False, the patches are not stitched back together and the loss is instead computed per patch
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is True
        """
        self.n_epochs = n_epochs
        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.mg_patching_levels = mg_patching_levels
        self.mg_patching_stitching = mg_patching_stitching
        self.use_distributed = use_distributed
        self.device = device

        if mg_patching_levels > 0:
            self.mg_n_patches = 2**mg_patching_levels
            if verbose:
                print(f'Training on {self.mg_n_patches**2} multi-grid patches.')
                sys.stdout.flush()
        else:
            self.mg_n_patches = 1
            mg_patching_padding = 0
            if verbose:
                print(f'Training on regular inputs (no multi-grid patching).')
                sys.stdout.flush()

        self.mg_patching_padding = mg_patching_padding
        self.patcher = MultigridPatching2D(model, levels=mg_patching_levels, padding_fraction=mg_patching_padding,
                                           use_distributed=use_distributed, stitching=mg_patching_stitching)

    def train(self, train_loader, test_loaders, output_encoder,
              model, optimizer, scheduler, regularizer, 
              training_loss=None, eval_losses=None, using_sam=False):
        """Trains the given model on the given datasets"""
        n_train = len(train_loader.dataset)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if self.verbose:
            print(f'Training on {n_train} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        if output_encoder is not None:
            output_encoder.to(self.device)
        
        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 
        
        for epoch in range(self.n_epochs):
            avg_loss = 0
            avg_lasso_loss = 0
            model.train()
            t1 = default_timer()
            train_err = 0.0

            data_length = len(train_loader)
            visualize_data = {'x':[], 'y':[], 'out':[]}

            # bp()
            for ind, sample in enumerate(train_loader):
                # when considering boundary condition, the last half of x in the last dimension is bc
                x, y = sample['x'], sample['y']
                
                x, y = self.patcher.patch(x, y)
                x = x.to(self.device)
                y = y.to(self.device)

                if regularizer:
                    regularizer.reset()

                out = model(x)
                out, y = self.patcher.unpatch(out, y)

                if ind == 0 or ind == data_length - 1 or ind == data_length // 2:
                    # if len(x.shape) == 3, it denotes that we use 1d data
                    if len(x.shape) == 3:
                        visualize_data['x'].append(x[0])
                        visualize_data['y'].append(y[0])
                        visualize_data['out'].append(out[0])
                    # if len(x.shape) == 4, it denotes that we use 2d data
                    elif len(x.shape) == 4:
                        # If we use 2d data，then x.shape is [batch, 1, t_len, width]
                        # So we should take 3 time steps from each batch
                        for b in range(x.shape[0]):
                            for time_step in range(3):
                                t_len = x.shape[2]
                                vis_x = x[b, 0, int(t_len * time_step / 2.0), :]
                                vis_y = y[b, 0, int(t_len * time_step / 2.0), :]
                                vis_out = out[b, 0, int(t_len * time_step / 2.0), :]
                                visualize_data['x'].append(vis_x)
                                visualize_data['y'].append(vis_y)
                                visualize_data['out'].append(vis_out)
                    

                # Output encoding only works if output is stiched
                if output_encoder is not None and self.mg_patching_stitching:
                    out = output_encoder.decode(out)
                    y = output_encoder.decode(y)

                if not using_sam:
                    optimizer.zero_grad(set_to_none=True)
                    loss = training_loss(out.float(), y)

                    if regularizer:
                        loss += regularizer.loss

                    loss.backward()   
                    optimizer.step()
                else:
                    loss = training_loss(out.float(), y)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    out = model(x)
                    out, y = self.patcher.unpatch(out, y)
                    # second forward-backward step
                    training_loss(out.float(), y).backward()
                    optimizer.second_step(zero_grad=True)

                train_err += loss.item()

                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1
            del x, y

            train_err/= n_train
            avg_loss /= self.n_epochs
            
            if epoch % self.log_test_interval == 0: 
                
                msg = f'[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.5f}, train_err={train_err:.5f}'

                values_to_log = dict(train_err=train_err, time=epoch_train_time, avg_loss=avg_loss)

                for loader_name, loader in test_loaders.items():
                    if epoch == self.n_epochs - 1 and self.log_output:
                        to_log_output = True
                    else:
                        to_log_output = False

                    errors = self.evaluate(model, eval_losses, loader, epoch, output_encoder, log_prefix=loader_name)

                    for loss_name, loss_value in errors.items():
                        msg += f', {loss_name}={loss_value:.5f}'
                        values_to_log[loss_name] = loss_value

                if regularizer:
                    avg_lasso_loss /= self.n_epochs
                    msg += f', avg_lasso={avg_lasso_loss:.5f}'

                if self.verbose and is_logger:
                    print(msg)
                    sys.stdout.flush()

                # Wandb loging
                if self.wandb_log and is_logger:
                    for pg in optimizer.param_groups:
                        lr = pg['lr']
                        values_to_log['lr'] = lr
                    wandb.log(values_to_log, step=epoch, commit=True)
                    visualize(epoch = epoch, visualize_samples = visualize_data, trainSamples = True)


    def evaluate(self, model, loss_dict, data_loader, epoch, output_encoder=None, 
                 log_prefix=''):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        model : model to evaluate
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        output_encoder : used to decode outputs if not None
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        model.eval()

        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}

        data_length = len(data_loader)
        visualize_data = {'x':[], 'y':[], 'out':[]}
        n_samples = 0
        with torch.no_grad():
            for it, sample in enumerate(data_loader):
                x, y = sample['x'], sample['y']

                n_samples += x.size(0)
                
                x, y = self.patcher.patch(x, y)
                y = y.to(self.device)
                x = x.to(self.device)
                
                out = model(x)
        
                out, y = self.patcher.unpatch(out, y, evaluation=True)

                if output_encoder is not None:
                    out = output_encoder.decode(out)

                if it == 0 or it == data_length - 1 or it == data_length // 2:
                    visualize_data['x'].append(x[0])
                    visualize_data['y'].append(y[0])
                    visualize_data['out'].append(out[0])

                if (it == 0) and self.log_output and self.wandb_log and is_logger:
                    if out.ndim == 2:
                        img = out
                    else:
                        img = out.squeeze()[0]
                    wandb.log({f'image_{log_prefix}': wandb.Image(img.unsqueeze(-1).cpu().numpy())}, commit=False)
                
                for loss_name, loss in loss_dict.items():
                    errors[f'{log_prefix}_{loss_name}'] += loss(out, y).item()
                
        if epoch % self.log_test_interval == 0 and self.wandb_log and is_logger:
            visualize(epoch = epoch, visualize_samples = visualize_data, trainSamples = False)

        del x, y, out

        for key in errors.keys():
            errors[key] /= n_samples

        return errors


