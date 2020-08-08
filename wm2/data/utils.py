import numpy as np
import torch
from torch.utils.data import Subset

from wm2.utils import TensorNamespace


class SARI:
    def __init__(self, state, action, reward, done, info):
        self.state = state
        self.action = action
        if not isinstance(reward, np.ndarray):
            self.reward = np.array([reward])
        else:
            self.reward = reward
        self.has_reward = reward != 0
        self.done = done
        self.info = info
        self.pcont = 0


def one_hot(a, max, dtype=np.float32):
    hot = np.zeros(max + 1, dtype=dtype)
    hot[a] = 1.0
    return hot


def split(data, train_len, test_len):
    total_len = train_len + test_len
    train = Subset(data, range(0, train_len))
    test = Subset(data, range(train_len, total_len))
    return train, test


def chomp(seq, end, dim, bite_size):
    """
    Shortens a dimension by removing the head or tail
    :param seq: the input sequence as tensor
    :param end: chomp the head (index 0) or the tail (index last)
    :param dim: the time dimension to chomp along
    :param bite_size: the number of elements to chomp
    :return:
    """
    chomped_len = seq.size(dim) - bite_size
    if end == 'head':
        return torch.narrow(seq, dim, bite_size, chomped_len)
    if end == 'tail':
        return torch.narrow(seq, dim, 0, chomped_len)
    else:
        Exception('end parameter must be head or tail')


def pad_collate(batch):
    # longest = max([trajectory['state'].shape[0] for trajectory in batch])
    # data = pad(batch, longest)
    # dtype = data[next(iter(data))].dtype
    # mask = make_mask(batch, longest, dtype=dtype)
    # data['mask'] = mask
    # todo temporary fix until we support batches of trajectories
    data = {}
    for key in batch[0]:
        data[key] = torch.from_numpy(batch[0][key]).unsqueeze(0)
    return TensorNamespace(**data)


def pad_collate_old(batch):
    longest = max([trajectory['state'].shape[0] for trajectory in batch])
    data = pad(batch, longest)
    dtype = data[next(iter(data))].dtype
    mask = make_mask(batch, longest, dtype=dtype)
    data['mask'] = mask
    for key in data:
        data[key] = torch.from_numpy(data[key])
    return TensorNamespace(**data)


# def pad_collate_2(batch):
#     """
#     returns batch in [T, B, D] format (timesteps, Batch, dimensions)
#     :param batch:
#     :return:
#     """
#     data = {}
#     for key in batch[0]:
#         data[key] = torch.from_numpy(batch[0][key]).unsqueeze(1)
#     return TensorNamespace(**data)


# def pad_collate_2(batch):
#     """
#     returns batch in [T, B, D] format (timesteps, Batch, dimensions)
#     :param batch:
#     :return:
#     """
#     data = {}
#     for key in batch[0]:
#         data[key] = []
#     for element in batch:
#         for key in element:
#                 data[key] += [torch.from_numpy(element[key])]
#     for key in data:
#         data[key] = torch.stack(data[key]).permute(1, 0, 2)
#     return TensorNamespace(**data)

def pad_collate_3(batch):
    """
    returns batch in [T, B, D] format (timesteps, Batch, dimensions)
    :param batch:
    :return:
    """

    for b in batch:
        a = np.stack(b['action'])
        print(f' pad_collate_2 {a.max()} {a.min()}')
        # a = b['action']
        if a.max() > 2.0 or a.min() < -2.0:
            print(f' pad_collate_2 {a.max()} {a.min()}')


    lengths = [trajectory['state'].shape[0] for trajectory in batch]
    longest = max(lengths)
    data = {'pad': []}
    shape = {}
    dtype = {}

    # initialize parameters
    for key in batch[0]:
        data[key] = []
        shape[key] = batch[0][key].shape[1]
        dtype[key] = batch[0][key].dtype

    for i, element in enumerate(batch):
        data['pad'] += [np.concatenate(
            (np.ones(lengths[i], dtype=np.float32), np.zeros(longest - lengths[i], dtype=np.float32)))]

    for key in element:
        padding = np.zeros((longest - lengths[i], shape[key]), dtype=dtype[key])
        element[key] += np.concatenate((element[key], padding), axis=0)

    # populate the sequences from the batch
    for i, element in enumerate(batch):
        data['pad'][i] = torch.from_numpy(data['pad'][i]).view(-1, 1)
        for key in element:
            data[key] += [torch.from_numpy(element[key])]

    for key in data:
        data[key] = torch.stack(data[key]).permute(1, 0, 2)

    mx = data['action'].max()
    mn = data['action'].min()
    print(f' pad_collate_out {mx} {mn}')

    return TensorNamespace(**data)


def pad_tensor(lengths):
    """
    returns a tensor that indicates which sequences are padded by 0 and 1
    :param lengths: an array with the lengths of each sequence [30, 43, 56]
    :return: a tensor [max_len, batch_size, 1] of ones and zeros
    """
    T, B, D = max(lengths), len(lengths), 1
    pad = torch.zeros(T, B, D)
    for i, l in enumerate(lengths):
        pad[torch.arange(0, l), i] = 1.0
    return pad


def pad_collate_2(batch):
    """
    returns batch in [T, B, D] format (timesteps, Batch, dimensions)
    :param batch: a list of dictionaries containing numpy arrays of [T, D]
    :return: [T, B, D] formatted batch, zero padded at the sequence ends
    """

    lengths = [trajectory['state'].shape[0] for trajectory in batch]
    T, B = max(lengths), len(batch)

    shape = {}
    dtype = {}
    keys = [k for k in batch[0]]

    # initialize parameters
    for key in keys:
        shape[key] = batch[0][key].shape[1]

    data = {}

    # initialize zero tensors
    for key in keys:
        data[key] = torch.zeros(T, B, shape[key])

    # copy the data
    for i, (trajectory, length) in enumerate(zip(batch, lengths)):
        for key in trajectory:
            data[key][torch.arange(0, length), i] = torch.from_numpy(trajectory[key].astype(np.float32))

    # add indicators for the padding, can be used as a loss mask
    data['pad'] = pad_tensor(lengths)

    return TensorNamespace(**data)



def autoregress(state, action, reward, mask, target_start=0, target_length=None, advance=1):
    """

    :param state: (N, T, S)
    :param action: (N, T, A)
    :param reward: (N, T, 1)
    :param mask: (N, T, 1)
    :param target_start: start index of a slice across the state dimension to output as target
    :param target_length: length of slice across the state dimension to output as target
    :param advance: the amount of timesteps to advance the target, default 1
    :return:
    source: concatenated (state, action, reward),
    target: subset of the source advanced by 1,
    mask: loss_mask that is zero where padding was put, or where it makes no sense to make a prediction
    """
    source = torch.cat((state, reward), dim=2)
    target = source.clone()
    if target_length is not None:
        target = target.narrow(dim=2, start=target_start, length=target_length)

    ret = {}
    ret['source'] = chomp(source, 'tail', dim=1, bite_size=advance)
    ret['action'] = chomp(action, 'tail', dim=1, bite_size=advance)
    ret['target'] = chomp(target, 'head', dim=1, bite_size=advance)
    ret['mask'] = chomp(mask, 'head', dim=1, bite_size=advance)
    return TensorNamespace(**ret)


def chomp_and_pad(i, dim, bite_size=1, mode='head', pad_mode='zeros'):
    """
    chomps at the head or tail, and pads the opposite end
    :param i: input tensor
    :param dim: the dimension to chomp
    :param bite_size: number of elements to chomp
    :param mode: head or tail
    :param pad_mode: fill or zero
    :return:

        Example:

        >>> s = torch.tensor([1, 2, 3, 4])
        >>> chomp_and_pad(s, 0, mode='tail', pad_mode='fill')
        ... tensor([1, 1, 2, 3])
        >>> chomp_and_pad(s, 0, mode='head', pad_mode='fill')
        ... tensor([2, 3, 4, 4])


    """
    pad_size = list(i.shape)
    pad_size[dim] = 1
    length = i.size(dim)
    if pad_mode == 'zeros':
        pad = torch.zeros(*pad_size, dtype=i.dtype, device=i.device)
    elif pad_mode == 'fill':
        if mode == 'head':
            pad = torch.index_select(i, dim=dim, index=torch.tensor([length - 1], device=i.device))
        elif mode == 'tail':
            pad = torch.index_select(i, dim=dim, index=torch.tensor([0], device=i.device))
        else:
            raise Exception('mode: head or tail')
    else:
        raise Exception('pad_mode: zeros or fill')

    i_plus_1 = chomp(i.clone(), mode, dim=dim, bite_size=bite_size)
    if mode == 'head':
        i_plus_1 = torch.cat((i_plus_1, pad), dim=dim)
    elif mode == 'tail':
        i_plus_1 = torch.cat((pad, i_plus_1), dim=dim)
    else:
        raise Exception('mode: head or tail')

    return i_plus_1


def pad(batch, longest):
    params = {}
    for trajectory in batch:
        pad_len = longest - trajectory['state'].shape[0]
        padding = [(0, pad_len)]  # right pad only the first dim
        padding += [(0, 0) for _ in range(len(trajectory['state'].shape) - 1)]

        for key in trajectory:
            if key not in params:
                params[key] = []
            params[key] += [np.pad(trajectory[key], padding)]

    for key in params:
        params[key] = np.stack(params[key])
    return params


def make_mask(batch, longest, dtype=np.float):
    mask = []
    for trajectory in batch:
        l = trajectory['state'].shape[0]
        mask += [np.concatenate((np.ones(l, dtype=dtype), np.zeros(longest - l, dtype=dtype)))]
    mask = np.stack(mask)
    mask = np.expand_dims(mask, axis=2)
    return mask
