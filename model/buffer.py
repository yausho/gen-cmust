import torch
import copy
import os
from torch.utils.data import ConcatDataset
from utils.dataloader import DiskCacheDataset

class CausalColdStartAugmenter:
    """
    生成式因果数据增强器：融合纯净的 Flow Matching ODE 采样与因果掩码。
    将小张量聚合成大文件存储，彻底解决 I/O 读写瓶颈。
    """
    def __init__(self, scheduler, device):
        self.scheduler = scheduler
        self.device = device

    def augment(self, model, causal_decipher, current_task_data, args, inference_steps=20):
        if args.sampling_p <= 0 or args.gen_num_r <= 0:
            return current_task_data
            
        if hasattr(causal_decipher, 'temperature'):
            causal_decipher.temperature = 5.0
        
        model.eval()
        x_real, y_real = current_task_data.tensors
        p = args.sampling_p
        r = args.gen_num_r
        
        cache_dir = os.path.join(args.log_dir, f"coldstart_cache_task{getattr(args, 'current_task_id', 0)}_p{p}_r{r}")
        os.makedirs(cache_dir, exist_ok=True)
        print(f" [Cold Start] Generating {r} augmented sequences. Caching to: {cache_dir}")
        
        scheduler = copy.deepcopy(self.scheduler)
        scheduler.set_timesteps(inference_steps)
        
        # Accumulate batches to chunk_size before flushing.  Avoids per-batch file
        # proliferation (e.g. 2000 files) while keeping peak CPU RAM per flush bounded.
        chunk_size = max(getattr(args, 'augment_chunk_size', 512), args.batch_size)
        pending_x: list = []
        pending_y: list = []
        pending_count = 0
        chunk_idx = 0
        chunk_sizes: list = []
        
        with torch.no_grad():
            for seq_idx in range(r):
                sample_mask = (torch.rand(len(x_real)) < p).bool()
                if not sample_mask.any():
                    continue
                    
                subset_x = x_real[sample_mask]
                subset_y = y_real[sample_mask]
                
                batch_size = args.batch_size 
                for b in range(0, len(subset_x), batch_size):
                    b_x = subset_x[b:b+batch_size].to(self.device)
                    b_y = subset_y[b:b+batch_size].to(self.device)
                    
                    t_dummy = torch.zeros(b_y.shape[0], device=self.device)
                    _, b_scores = model(b_y, t_dummy, b_x)
                    b_mask = causal_decipher(b_y, b_scores)

                    # Infer valid (non-padding) region from the data itself:
                    # padding pixels are exactly 0 in all T and C dims.
                    spatial_nonzero = (b_y.abs().sum(dim=(1, 2)) > 0).float()  # (B, H, W)
                    b_valid = spatial_nonzero.unsqueeze(1).unsqueeze(1)         # (B, 1, 1, H, W)
                    b_mask = b_mask * b_valid
                    
                    b_latents = torch.randn_like(b_y).to(self.device)
                    b_noise_ref = torch.randn_like(b_y).to(self.device)
                    
                    sigmas = scheduler.sigmas.to(self.device)
                    timesteps = scheduler.timesteps.to(self.device)

                    for i, t in enumerate(timesteps):
                        sigma_curr = sigmas[i]
                        sigma_next = sigmas[i + 1]
                        
                        t_model = t.expand(b_latents.shape[0])
                        model_output, _ = model(b_latents, t_model, b_x)
                        
                        # 标准 ODE 欧拉步
                        latents_next = b_latents + (sigma_next - sigma_curr) * model_output
                        latent_causal_next = (1 - sigma_next) * b_y + sigma_next * b_noise_ref
                        
                        # 理论修复：纯化 ODE 采样，废弃 DDPM 式强行加噪回退
                        b_latents = latent_causal_next * b_mask + latents_next * (1 - b_mask)
                    
                    # Accumulate into pending buffer; release GPU tensors immediately.
                    pending_x.append(b_x.cpu())
                    pending_y.append(b_latents.cpu())
                    pending_count += b_x.shape[0]
                    del b_x, b_y, b_scores, b_mask, b_latents, b_noise_ref

                    # Flush when pending buffer reaches chunk_size.
                    if pending_count >= chunk_size:
                        _mx = torch.cat(pending_x, dim=0)
                        _my = torch.cat(pending_y, dim=0)
                        _cp = os.path.join(cache_dir, f"chunk_{chunk_idx:04d}.pt")
                        torch.save({'x': _mx, 'y': _my}, _cp, _use_new_zipfile_serialization=False)
                        chunk_sizes.append(int(_mx.shape[0]))
                        chunk_idx += 1
                        pending_x, pending_y, pending_count = [], [], 0
                
        # Flush remaining samples that did not fill a full chunk.
        if pending_x:
            _mx = torch.cat(pending_x, dim=0)
            _my = torch.cat(pending_y, dim=0)
            _cp = os.path.join(cache_dir, f"chunk_{chunk_idx:04d}.pt")
            torch.save({'x': _mx, 'y': _my}, _cp, _use_new_zipfile_serialization=False)
            chunk_sizes.append(int(_mx.shape[0]))
            chunk_idx += 1

        # Write manifest so DiskCacheDataset can build its index without opening any chunk file.
        if chunk_idx > 0:
            torch.save({'chunk_count': chunk_idx, 'chunk_sizes': chunk_sizes},
                       os.path.join(cache_dir, 'manifest.pt'),
                       _use_new_zipfile_serialization=False)
            total_samples = sum(chunk_sizes)
            print(f" [Cold Start] Cached {total_samples} samples across {chunk_idx} chunk files.")
            augmented_dataset = DiskCacheDataset(cache_dir)
            return ConcatDataset([current_task_data, augmented_dataset])

        return current_task_data