import torch
import pickle
from pathlib import Path

from .tensor_dataset import TensorDataset

def bp():
    import pdb; pdb.set_trace()

def load_diffusion_pt(train_ratio,
                batch_size, test_batch_size,
                data_path = Path(__file__).resolve().parent.joinpath('data').joinpath(f'data_dict.pickle'),
                scale = 10.0,
                whether_shuffle = True,
                split_train_test_randomly = True,
                channel_dim=1):
    """Load the diffusion dataset
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    # After unsqueeze, x.shape should be (N, 1, width)
    x = torch.from_numpy(data['x']['u']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
    bc = torch.from_numpy(data['x']['bc']).type(torch.float32).clone() * scale
    bc_padded = torch.zeros_like(x)
    bc_padded[:,:,-1:] = bc.unsqueeze(1).unsqueeze(1)
    y = torch.from_numpy(data['y']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
    n = x.shape[0]
    n_train = int(n * train_ratio)

    if split_train_test_randomly:
        perm = torch.randperm(n)
        x = x[perm, ...]
        y = y[perm, ...]
        bc_padded = bc_padded[perm, ...]

    x_train = x[0:n_train, ...]
    bc_train = bc_padded[0:n_train, ...]
    x_train = torch.cat([x_train, bc_train], dim = -1)
    y_train = y[0:n_train, ...]

    x_test = x[n_train:, ...]
    bc_test = bc_padded[n_train:, ...]
    x_test = torch.cat([x_test, bc_test], dim = -1)
    y_test = y[n_train:, ...]

    del data, x, y, bc, bc_padded, bc_train, bc_test
    
    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=whether_shuffle,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, test_loader



def load_diffusion_pt_withoutBC(train_ratio,
                batch_size, test_batch_size,
                data_path = Path(__file__).resolve().parent.joinpath('data').joinpath(f'data_dict.pickle'),
                scale = 10.0,
                whether_shuffle = True,
                split_train_test_randomly = True,
                channel_dim=1):
    """Load the diffusion dataset
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # After unsqueeze, x.shape should be (N, 1, width)
    x = torch.from_numpy(data['x']['u']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
    y = torch.from_numpy(data['y']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
    n = x.shape[0]
    n_train = int(n * train_ratio)

    if split_train_test_randomly:
        perm = torch.randperm(n)
        x = x[perm, ...]
        y = y[perm, ...]

    x_train = x[0:n_train, ...]
    y_train = y[0:n_train, ...]
    
    x_test = x[n_train:, ...]
    y_test = y[n_train:, ...]

    del data, x, y
    
    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=whether_shuffle,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                            batch_size=test_batch_size, shuffle=False,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, test_loader
    

def load_diffusion_pt_withoutBC_mixed(train_ratio,
                batch_size, test_batch_size,
                data_path_parent = 'C:\\Users\\jizhang\\Desktop\\projects\\ai4science\\electrochemistry\\NumericalDiffusionScheme-master\\Code_radial_normlized',
                data_names = ['data_dict_constBC_001.pickle', 'data_dict_constBC_002.pickle'],
                scale = 10.0,
                split_train_test_randomly = True,
                whether_shuffle = True,
                channel_dim=1):
    """Load the diffusion dataset
    """
    for data_name in data_names:
        data_path = Path(data_path_parent).joinpath(data_name)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # After unsqueeze, x.shape should be (N, 1, width)
        x = torch.from_numpy(data['x']['u']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
        y = torch.from_numpy(data['y']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
        n = x.shape[0]
        n_train = int(n * train_ratio)

        if split_train_test_randomly:
            perm = torch.randperm(n)
            x = x[perm, ...]
            y = y[perm, ...]

        if data_name == data_names[0]:
            x_train = x[0:n_train, ...]
            y_train = y[0:n_train, ...]

            x_test = x[n_train:, ...]
            y_test = y[n_train:, ...]
        else:
            x_train = torch.cat([x_train, x[0:n_train, ...]], dim = 0)
            y_train = torch.cat([y_train, y[0:n_train, ...]], dim = 0)

            x_test = torch.cat([x_test, x[n_train:, ...]], dim = 0)
            y_test = torch.cat([y_test, y[n_train:, ...]], dim = 0)

        del data, x, y

    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=whether_shuffle,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, test_loader


def load_diffusion_pt_2D_withoutBC_mixed(train_ratio,
                batch_size, test_batch_size,
                data_path_parent = 'C:\\Users\\jizhang\\Desktop\\projects\\ai4science\\electrochemistry\\NumericalDiffusionScheme-master\\Code_radial_normlized',
                data_names = ['data_dict_constBC_001.pickle', 'data_dict_constBC_002.pickle'],
                scale = 10.0,
                split_train_test_randomly = True,
                whether_shuffle = True,
                channel_dim=1):
    """Load the diffusion dataset
    """
    for data_name in data_names:
        data_path = Path(data_path_parent).joinpath(data_name)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # After unsqueeze, x.shape should be (N, 1, width)
        x = torch.from_numpy(data['x']['u']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
        x = x.permute(1, 0, 2)
        # After unsqueeze, x.shape should be (1, 1, N, width)
        x = x.unsqueeze(0)
        y = torch.from_numpy(data['y']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
        y = y.permute(1, 0, 2)
        # After unsqueeze, y.shape should be (1, 1, N, width)
        y = y.unsqueeze(0)

        if data_name == data_names[0]:
            x_train = x
            y_train = y

        elif data_name == data_names[2]:
            x_test = x
            y_test = y

        else:
            x_train = torch.cat([x_train, x], dim = 0)
            y_train = torch.cat([y_train, y], dim = 0)

        del data, x, y

    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=whether_shuffle,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, test_loader

def load_diffusion_pt_2D_mixed(train_ratio,
                batch_size, test_batch_size,
                data_path_parent = 'C:\\Users\\jizhang\\Desktop\\projects\\ai4science\\electrochemistry\\NumericalDiffusionScheme-master\\Code_radial_normlized',
                data_names = ['data_dict_constBC_001.pickle', 'data_dict_constBC_002.pickle'],
                scale = 10.0,
                split_train_test_randomly = True,
                whether_shuffle = True,
                channel_dim=1):
    """Load the diffusion dataset
    """
    for data_name in data_names:
        data_path = Path(data_path_parent).joinpath(data_name)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # After unsqueeze, x.shape should be (N, 1, width)
        x = torch.from_numpy(data['x']['u']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
        bc = torch.from_numpy(data['x']['bc']).type(torch.float32).clone() * scale
        bc_padded = torch.zeros_like(x)
        bc_padded[:,:,-1:] = bc.unsqueeze(1).unsqueeze(1)
        x = x.permute(1, 0, 2)
        # After unsqueeze, x.shape should be (1, 1, N, width)
        x = x.unsqueeze(0)
        bc_padded = bc_padded.permute(1, 0, 2)
        bc_padded = bc_padded.unsqueeze(0)
    
        y = torch.from_numpy(data['y']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
        y = y.permute(1, 0, 2)
        # After unsqueeze, y.shape should be (1, 1, N, width)
        y = y.unsqueeze(0)

        if data_name == data_names[0]:
            x_train = x
            bc_train = bc_padded
            x_train = torch.cat([x_train, bc_train], dim = -1)
            y_train = y

        elif data_name == data_names[2]:
            x_test = x
            bc_test = bc_padded
            x_test = torch.cat([x_test, bc_test], dim = -1)
            y_test = y

        else:
            x_train_tmp = x
            bc_train = bc_padded
            x_train_tmp = torch.cat([x_train_tmp, bc_train], dim = -1)
            y_train_tmp = y

            x_train = torch.cat([x_train, x_train_tmp], dim = 0)
            y_train = torch.cat([y_train, y_train_tmp], dim = 0)
        del data, x, y

    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=whether_shuffle,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, test_loader

def load_diffusion_pt_mixed(train_ratio,
                batch_size, test_batch_size,
                data_path_parent = 'C:\\Users\\jizhang\\Desktop\\projects\\ai4science\\electrochemistry\\NumericalDiffusionScheme-master\\Code_radial_normlized',
                data_names = ['data_dict_constBC_001.pickle', 'data_dict_constBC_002.pickle'],
                scale = 10.0,
                split_train_test_randomly = True, 
                whether_shuffle = True,
                channel_dim=1):
    """Load the diffusion dataset
    """
    for data_name in data_names:
        data_path = Path(data_path_parent).joinpath(data_name)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        x = torch.from_numpy(data['x']['u']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
        bc = torch.from_numpy(data['x']['bc']).type(torch.float32).clone() * scale
        bc_padded = torch.zeros_like(x)
        bc_padded[:,:,-1:] = bc.unsqueeze(1).unsqueeze(1)
        y = torch.from_numpy(data['y']).unsqueeze(channel_dim).type(torch.float32).clone() * scale
        n = x.shape[0]
        n_train = int(n * train_ratio)

        if split_train_test_randomly:
            perm = torch.randperm(n)
            x = x[perm, ...]
            y = y[perm, ...]
            bc_padded = bc_padded[perm, ...]

        if data_name == data_names[0]:
            # After unsqueeze, x.shape should be (N, 1, width)
            x_train = x[0:n_train, ...]
            bc_train = bc_padded[0:n_train, ...]
            x_train = torch.cat([x_train, bc_train], dim = -1)
            y_train = y[0:n_train, ...]

            x_test = x[n_train:, ...]
            bc_test = bc_padded[n_train:, ...]
            x_test = torch.cat([x_test, bc_test], dim = -1)
            y_test = y[n_train:, ...]

            del data, x, y, bc, bc_padded, bc_train, bc_test
        else:
            x_train_tmp = x[0:n_train, ...]
            bc_train = bc_padded[0:n_train, ...]
            x_train_tmp = torch.cat([x_train_tmp, bc_train], dim = -1)
            y_train_tmp = y[0:n_train, ...]

            x_test_tmp = x[n_train:, ...]
            bc_test = bc_padded[n_train:, ...]
            x_test_tmp = torch.cat([x_test_tmp, bc_test], dim = -1)
            y_test_tmp = y[n_train:, ...]

            x_train = torch.cat([x_train, x_train_tmp], dim = 0)
            y_train = torch.cat([y_train, y_train_tmp], dim = 0)

            x_test = torch.cat([x_test, x_test_tmp], dim = 0)
            y_test = torch.cat([y_test, y_test_tmp], dim = 0)

            del data, x, y, bc, bc_padded, bc_train, bc_test

    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=whether_shuffle,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = load_diffusion_pt('./data/data_dict.pickle', 5, 5, 5, 5)
    for sample in train_loader:
        x = sample['x']
        y = sample['y']
        print(x)
        print(y)