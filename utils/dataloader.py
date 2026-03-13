import bisect
import torch
import numpy as np
import os
import glob
import math
import torch.utils.data
from collections import OrderedDict
from torch.utils.data import Dataset, ConcatDataset

#数据归一化
class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std


    def fit_transform(self, data):
        axis = tuple(range(data.ndim - 1)) #维度减一，对最后一个维度进行归一化
        self.mean = np.mean(data, axis=axis, keepdims=True) #计算均值，保持维度
        self.std = np.std(data, axis=axis, keepdims=True) #计算标准差，保持维度
        return (data - self.mean) / (self.std + 1e-5) #归一化数据，添加小常数避免除零

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-5) #使用已计算的均值和标准差进行归一化，防止数据泄露

    def inverse_transform(self, data):
        #识别 5D 张量，强制 reshape 防止多通道广播错位
        if data.ndim == 5 and self.std.ndim == 4:
            std_5d = self.std.reshape(1, 1, -1, 1, 1)
            mean_5d = self.mean.reshape(1, 1, -1, 1, 1)
            return (data * (std_5d + 1e-5)) + mean_5d
        return (data * (self.std + 1e-5)) + self.mean #逆归一化数据，添加小常数避免乘零

#自动填充到 32x32 网格的函数，适用于非完美平方节点数的图数据。
def auto_pad_to_32(x_tensor, expected_size=32):
    N, T, Nodes, C = x_tensor.shape
    
    side = math.ceil(math.sqrt(Nodes))
    if side > expected_size:
        rem = side % 4
        expected_size = side + (4 - rem) if rem != 0 else side

    root = math.sqrt(Nodes)
    if int(root + 0.5) ** 2 == Nodes:
        H_inner, W_inner = int(root), int(root)
        x_reshaped = x_tensor
    elif Nodes == 1024: H_inner, W_inner = 32, 32; x_reshaped = x_tensor
    elif Nodes == 256: H_inner, W_inner = 16, 16; x_reshaped = x_tensor
    elif Nodes == 128: H_inner, W_inner = 16, 8; x_reshaped = x_tensor
    elif Nodes == 200: H_inner, W_inner = 10, 20; x_reshaped = x_tensor
    else:
        target_nodes = side * side
        H_inner, W_inner = side, side
        pad_len = target_nodes - Nodes
        zeros = torch.zeros(N, T, pad_len, C, device=x_tensor.device)
        x_reshaped = torch.cat([x_tensor, zeros], dim=2)

    x_grid = x_reshaped.permute(0, 1, 3, 2).view(N, T, C, H_inner, W_inner)
    
    canvas = torch.zeros(N, T, C, expected_size, expected_size, device=x_tensor.device)
    valid_mask = torch.zeros(1, 1, 1, expected_size, expected_size, device=x_tensor.device)
    
    h_cut = min(H_inner, expected_size)
    w_cut = min(W_inner, expected_size)
    
    canvas[..., :h_cut, :w_cut] = x_grid[..., :h_cut, :w_cut]
    valid_mask[..., :h_cut, :w_cut] = 1.0
    
    return canvas, valid_mask

def get_dataloaders_scaler(dataset_dir, batch_size=16, logger=None):
    data = {}
    datasets = {}
    dataloaders = {}
    
    if os.path.exists(os.path.join(dataset_dir, 'train.npz')):
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
            raw_x = cat_data['x'][..., :1]
            raw_y = cat_data['y'][..., :1]
            raw_x = np.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0)
            raw_y = np.nan_to_num(raw_y, nan=0.0, posinf=0.0, neginf=0.0)
            data['x_' + category] = raw_x
            data['y_' + category] = raw_y
    else:
        raise FileNotFoundError(f"Cannot find train.npz in {dataset_dir}")
        
    scaler = StandardScaler()
    
    data['x_train'] = scaler.fit_transform(data['x_train'])
    if 'x_val' in data: data['x_val'] = scaler.transform(data['x_val'])
    if 'x_test' in data: data['x_test'] = scaler.transform(data['x_test'])
    
    for category in ['train', 'val', 'test']:
        if 'y_' + category in data:
            data['y_' + category] = scaler.transform(data['y_' + category])

    global_valid_mask = None
    
    for category in ['train', 'val', 'test']:
        if 'x_' + category not in data: continue
        
        clipped_x = np.clip(data['x_' + category], -5.0, 5.0)
        
        if category in ['train', 'val']:
            clipped_y = np.clip(data['y_' + category], -5.0, 5.0)
        else:
            clipped_y = data['y_' + category] 
        
        x_tensor = torch.FloatTensor(clipped_x)
        y_tensor = torch.FloatTensor(clipped_y)

        x_padded, valid_mask = auto_pad_to_32(x_tensor)
        y_padded, _ = auto_pad_to_32(y_tensor)
        
        global_valid_mask = valid_mask
        datasets[category] = torch.utils.data.TensorDataset(x_padded, y_padded)
    
    if logger:
        logger.info(f"Data Loaded from {dataset_dir}")
        logger.info(f"Train: {len(datasets['train'])} Val: {len(datasets['val'])} Test: {len(datasets['test'])}")

    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True)
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, drop_last=False)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=False)

    return dataloaders, scaler, global_valid_mask

class DiskCacheDataset(Dataset):
    """
    Manifest-driven chunked dataset with bounded OrderedDict LRU chunk cache.
    Solves three I/O pathologies vs. the naive per-batch-file approach:
    1. __init__ reads only the tiny manifest file — no mass file opens at startup.
    2. Bounded LRU keeps N most-recently-used chunks in RAM; each DataLoader
       batch loads at most one new chunk instead of one per sample.
    3. Chunk files are saved in legacy (non-ZIP) torch format so OS-level mmap
       is truly effective (ZIP format requires in-memory decompression).
    """
    def __init__(self, cache_dir, lru_size=8):
        manifest_path = os.path.join(cache_dir, 'manifest.pt')
        if os.path.exists(manifest_path):
            manifest = torch.load(manifest_path, map_location='cpu', weights_only=True)
            chunk_count = int(manifest['chunk_count'])
            chunk_sizes = [int(s) for s in manifest['chunk_sizes']]
            self._chunk_files = [
                os.path.join(cache_dir, f'chunk_{i:04d}.pt')
                for i in range(chunk_count)
            ]
        else:
            # Legacy fallback: scan files and read sizes once (acceptable for small legacy caches).
            chunk_files = sorted(glob.glob(os.path.join(cache_dir, 'chunk_*.pt')))
            legacy = os.path.join(cache_dir, 'augmented_data_chunk.pt')
            self._chunk_files = chunk_files or ([legacy] if os.path.exists(legacy) else [])
            chunk_sizes = []
            for f in self._chunk_files:
                d = torch.load(f, map_location='cpu', weights_only=True)
                chunk_sizes.append(int(d['x'].shape[0]))
                del d

        self._chunk_sizes = chunk_sizes
        self._cumulative: list = [0] + list(np.cumsum(chunk_sizes).astype(int))
        self._lru: OrderedDict = OrderedDict()
        self._lru_size = lru_size

    def __len__(self):
        return self._cumulative[-1]

    def _get_chunk(self, chunk_idx: int):
        if chunk_idx in self._lru:
            self._lru.move_to_end(chunk_idx)
            return self._lru[chunk_idx]
        if len(self._lru) >= self._lru_size:
            self._lru.popitem(last=False)
        data = torch.load(self._chunk_files[chunk_idx],
                          map_location='cpu', weights_only=True)
        self._lru[chunk_idx] = data
        return data

    def __getitem__(self, idx):
        chunk_idx = bisect.bisect_right(self._cumulative, idx) - 1
        chunk_idx = max(0, min(chunk_idx, len(self._chunk_files) - 1))
        local_idx = idx - self._cumulative[chunk_idx]
        data = self._get_chunk(chunk_idx)
        return data['x'][local_idx], data['y'][local_idx]


class ChunkShuffleSampler(torch.utils.data.Sampler):
    """
    Chunk-aware sampler for DataLoader.  When the dataset is a ConcatDataset,
    in-memory sub-datasets are shuffled sample-by-sample; DiskCacheDataset
    sub-datasets are shuffled at chunk granularity so the LRU stays effective:
    all samples from the same chunk are requested consecutively, eliminating
    the cache-thrash caused by global shuffle=True.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # Collect indices as blocks, then shuffle blocks across sub-datasets
        # to interleave real and augmented data while preserving within-chunk
        # locality for LRU effectiveness.
        blocks = []
        subs = self.dataset.datasets if isinstance(self.dataset, ConcatDataset) else [self.dataset]
        offset = 0
        for sub in subs:
            n = len(sub)
            if isinstance(sub, DiskCacheDataset) and sub._chunk_files:
                for ci in torch.randperm(len(sub._chunk_files)).tolist():
                    lo = sub._cumulative[ci] + offset
                    hi = sub._cumulative[ci + 1] + offset
                    blocks.append((torch.randperm(hi - lo) + lo).tolist())
            else:
                # Split in-memory dataset into virtual blocks (~256 samples each)
                # so it interleaves with disk chunks at a fine granularity.
                perm = (torch.randperm(n) + offset).tolist()
                blk_size = 256
                for s in range(0, len(perm), blk_size):
                    blocks.append(perm[s:s + blk_size])
            offset += n
        # Shuffle block order to interleave real and augmented data.
        for bi in torch.randperm(len(blocks)).tolist():
            yield from blocks[bi]