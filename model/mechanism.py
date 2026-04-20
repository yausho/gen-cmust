import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq

class CausalDecipher(nn.Module):
    """
    Nucleus Energy Masking: adaptively split causal / environment regions
    per sample based on cumulative attention energy, replacing the fixed
    env_ratio.  Analogous to nucleus (top-p) sampling in NLP —— accumulate
    patches from highest importance until their energy reaches
    `nucleus_energy_p` of the total, mark them as causal core, and treat
    the rest as environment.
    """
    def __init__(self, nucleus_energy_p=0.2, patch_size=4, temperature=10.0,
                 grad_scale=0.1, min_causal_ratio=0.05, max_causal_ratio=0.5):
        super().__init__()
        self.nucleus_energy_p = nucleus_energy_p
        self.min_causal_ratio = min_causal_ratio
        self.max_causal_ratio = max_causal_ratio
        self.patch_size = patch_size
        self.temperature = temperature
        self.grad_scale = grad_scale

    def forward(self, x, importance_scores, hard=True, threshold_offset=0.0):
        if importance_scores.dim() == 2:  # [B, N] from non-sequential DiT -> [B, 1, N]
            importance_scores = importance_scores.unsqueeze(1)
        B, T, Num_Patches = importance_scores.shape

        # --- Nucleus Energy Masking ---
        # Sort patches by importance descending; accumulate energy until the
        # cumulative fraction reaches nucleus_energy_p.  The number of patches
        # needed is sample-adaptive: concentrated attention → few causal patches,
        # spread-out attention → more causal patches.
        sorted_desc, _ = torch.sort(importance_scores, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_desc, dim=-1)             # (B, T, N)
        total_energy = cumsum[..., -1:]                         # (B, T, 1)
        normalized_cumsum = cumsum / (total_energy + 1e-8)      # (B, T, N)

        # Count how many top patches are needed to reach the energy threshold.
        # +1 to include the patch that crosses the boundary.
        causal_counts = (normalized_cumsum < self.nucleus_energy_p).long().sum(dim=-1) + 1  # (B, T)

        # Clamp to [min, max] causal ratio to prevent degenerate masks.
        min_k = max(1, int(Num_Patches * self.min_causal_ratio))
        max_k = max(min_k, int(Num_Patches * self.max_causal_ratio))
        causal_counts = causal_counts.clamp(min=min_k, max=max_k)

        # Retrieve the threshold score: the importance of the last patch still
        # inside the causal nucleus.  Patches with score >= this are causal.
        threshold_idx = (causal_counts - 1).unsqueeze(-1)       # (B, T, 1)
        env_threshold = sorted_desc.gather(-1, threshold_idx).detach()  # (B, T, 1)
        
        scores_for_grad = importance_scores * self.grad_scale + importance_scores.detach() * (1 - self.grad_scale)
        soft_mask = torch.sigmoid(self.temperature * (scores_for_grad - env_threshold))
        
        if hard:
            bin_threshold = 0.5 + threshold_offset
            hard_mask = (soft_mask > bin_threshold).float()
            mask_causal_1d = hard_mask - soft_mask.detach() + soft_mask
        else:
            mask_causal_1d = soft_mask

        # Derive true H_patch / W_patch from the actual spatial tensor.
        # x may be 5D (B, T, C, H, W) or 4D (B, C, H, W); never assume square.
        if x.dim() == 5:
            spatial_H, spatial_W = x.shape[3], x.shape[4]
        else:
            spatial_H, spatial_W = x.shape[2], x.shape[3]
        H_patch = spatial_H // self.patch_size
        W_patch = spatial_W // self.patch_size

        mask_2d = mask_causal_1d.view(B, T, 1, H_patch, W_patch)
        mask_spatial = F.interpolate(
            mask_2d.view(B * T, 1, H_patch, W_patch), 
            scale_factor=self.patch_size, 
            mode="nearest"
        ).view(B, T, 1, H_patch * self.patch_size, W_patch * self.patch_size)
        
        # Expand to full spatial channels: x.shape[2]=C for 5D, x.shape[1]=C for 4D.
        out_channels = x.shape[2] if x.dim() == 5 else x.shape[1]
        return mask_spatial.expand(-1, -1, out_channels, -1, -1)


class CausalMemoryController:
    def __init__(self, memory_capacity=1000, causal_keep_ratio=0.2, patch_size=4, logger=None):
        self.capacity = memory_capacity
        self.causal_keep_ratio = causal_keep_ratio
        self.patch_size = patch_size
        self.logger = logger
        
        # 性能修复：使用 O(log K) 的优先队列代替 O(N) 的线性扫描
        self.memory_heap = [] 
        self.tiebreaker = 0   
        self.seen_samples = 0

    @property
    def buffer_history(self):
        # Backward compatibility for older training loops.
        return self.memory_heap

    def proactive_augmentation(self, model, scheduler, causal_decipher, x_batch, y_batch, inference_steps=20, resample_steps=2):
        if hasattr(causal_decipher, 'temperature'):
            causal_decipher.temperature = 5.0
            
        model.eval()
        x, y = x_batch.to(next(model.parameters()).device), y_batch.to(next(model.parameters()).device)
        
        with torch.no_grad():
            _, scores = model(y, torch.zeros(y.shape[0], device=y.device), x)
            mask = causal_decipher(y, scores)
            latents, noise_ref = torch.randn_like(y), torch.randn_like(y)
            sigmas, timesteps = scheduler.sigmas.to(y.device), scheduler.timesteps.to(y.device)

            for i, t in enumerate(timesteps):
                model_output, _ = model(latents, t.expand(latents.shape[0]), x)
                latents_next = latents + (sigmas[i+1] - sigmas[i]) * model_output
                latent_causal_next = (1 - sigmas[i+1]) * y + sigmas[i+1] * noise_ref
                
                # 理论修复：剥离 DDPM 加噪，保留纯粹的 Flow Matching ODE 更新
                latents = latent_causal_next * mask + latents_next * (1 - mask)
                
        model.train()
        return latents.detach()

    def update_memory(self, history_batch, target_batch, importance_scores, causal_mask=None, valid_mask=None):
        if importance_scores.dim() == 2:  # [B, N] -> [B, 1, N]
            importance_scores = importance_scores.unsqueeze(1)
        B, T, N = importance_scores.shape
        k = max(1, int(N * self.causal_keep_ratio))
        topk_scores, _ = torch.topk(importance_scores, k, dim=-1)
        sample_causal_scores = topk_scores.reshape(B, -1).mean(dim=-1)  # robust: handles any T/k layout

        # Do NOT apply causal_mask to target_batch here. Zeroing env regions would
        # corrupt the noisy_target distribution during replay (DiT global attention
        # sees near-zero env patches instead of real data+noise). The replay loss in
        # engine.py already applies causal_mask_rep to focus on causal regions.

        for i in range(len(history_batch)):
            self.seen_samples += 1
            self.tiebreaker += 1
            score = sample_causal_scores[i].item()
            vm = valid_mask[i].detach().cpu() if valid_mask is not None else None
            item = (score, self.tiebreaker,
                    history_batch[i].detach().cpu(),
                    target_batch[i].detach().cpu(),
                    importance_scores[i].detach().cpu(),
                    vm)

            if len(self.memory_heap) < self.capacity:
                heapq.heappush(self.memory_heap, item)
            else:
                if score > self.memory_heap[0][0]:
                    heapq.heapreplace(self.memory_heap, item)

    def get_replay_data(self, batch_size):
        if not self.memory_heap: return None, None, None, None

        # Build size -> index list in one pass (O(N)).
        # Key is the full shape tuple of the target tensor to prevent heterogeneous
        # H×W tensors with the same width (e.g. 32×64 vs 64×64) from being grouped
        # together, which would cause torch.stack to crash on mismatched heights.
        size_to_indices: dict = {}
        for i, item in enumerate(self.memory_heap):
            s = tuple(item[3].shape)  # full shape, not just width
            size_to_indices.setdefault(s, []).append(i)

        sizes = list(size_to_indices.keys())
        counts = np.array([len(size_to_indices[s]) for s in sizes], dtype=np.float64)

        # Weighted size selection: probability proportional to count so that rare
        # resolutions are chosen infrequently, avoiding batch-size collapse.
        probs = counts / counts.sum()
        chosen_size = sizes[int(np.random.choice(len(sizes), p=probs))]
        valid_indices = size_to_indices[chosen_size]

        # Hard fallback: if the chosen bucket is too small (< batch_size // 2),
        # switch to the most populous size to keep gradient variance stable.
        min_replay = max(1, batch_size // 2)
        if len(valid_indices) < min_replay:
            chosen_size = sizes[int(np.argmax(counts))]
            valid_indices = size_to_indices[chosen_size]

        # When the pool is smaller than batch_size, use replacement sampling
        # to fill the full batch, preventing gradient-variance spikes from tiny batches.
        allow_replace = len(valid_indices) < batch_size
        indices = np.random.choice(valid_indices, batch_size, replace=allow_replace)

        batch_target, batch_history, batch_scores, batch_valid = [], [], [], []
        for i in indices:
            item = self.memory_heap[i]
            # Handle legacy 5-element items (no valid_mask) for backward compatibility
            if len(item) > 5:
                _, _, t_hist, t_tgt, t_score, t_valid = item
            else:
                _, _, t_hist, t_tgt, t_score = item
                t_valid = None
            batch_target.append(t_tgt)
            batch_history.append(t_hist)
            batch_scores.append(t_score)
            batch_valid.append(t_valid)

        valid_stacked = torch.stack(batch_valid) if batch_valid[0] is not None else None
        return torch.stack(batch_target), torch.stack(batch_history), torch.stack(batch_scores), valid_stacked


class CausalRoAdaController:
    def __init__(self, var_threshold=1e-4, min_grad=0.0, max_freeze_ratio=0.5,
                 logger=None, causal_env_ratio=0.8, causal_max_grad=None, env_min_grad=None):
        self.threshold = var_threshold
        self.min_grad = min_grad
        # Use sensible defaults: causal gradients should be small (~0.05), env should be active
        self.causal_max_grad = float(0.05 if causal_max_grad is None else causal_max_grad)
        self.env_min_grad = float(1e-4 if env_min_grad is None else env_min_grad)
        self.max_freeze_ratio = max_freeze_ratio
        self.causal_env_ratio = causal_env_ratio
        self.logger = logger
        self.grad_history_causal = {}
        self.grad_history_env = {}
        self.current_task_ema_causal = {}
        self.current_task_ema_env = {}
        self.frozen_flags = {} 

    def commit_task_signature(self, max_history: int = 3):
        for name, ema_grad in self.current_task_ema_causal.items():
            if name not in self.grad_history_causal:
                self.grad_history_causal[name] = []
            self.grad_history_causal[name].append(ema_grad.cpu().numpy())
            self.grad_history_causal[name] = self.grad_history_causal[name][-max_history:]
        self.current_task_ema_causal.clear()

        for name, ema_grad in self.current_task_ema_env.items():
            if name not in self.grad_history_env:
                self.grad_history_env[name] = []
            self.grad_history_env[name].append(ema_grad.cpu().numpy())
            self.grad_history_env[name] = self.grad_history_env[name][-max_history:]
        self.current_task_ema_env.clear()

    def update_dual_signature(self, model, loss_causal, loss_env, momentum=0.9, inv_scale=1.0):
        named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not named_params:
            return

        params = [p for _, p in named_params]
        grads_causal = torch.autograd.grad(loss_causal, params, retain_graph=True, allow_unused=True)
        grads_env = torch.autograd.grad(loss_env, params, retain_graph=True, allow_unused=True)

        for (name, _), grad_c, grad_e in zip(named_params, grads_causal, grads_env):
            if grad_c is not None:
                grad_c = grad_c.detach() * inv_scale  # undo AMP scaling to restore true magnitude
                if torch.isfinite(grad_c).all():
                    if name not in self.current_task_ema_causal:
                        self.current_task_ema_causal[name] = grad_c.clone()
                    else:
                        self.current_task_ema_causal[name] = (
                            momentum * self.current_task_ema_causal[name] + (1 - momentum) * grad_c
                        )

            if grad_e is not None:
                grad_e = grad_e.detach() * inv_scale  # undo AMP scaling
                if torch.isfinite(grad_e).all():
                    if name not in self.current_task_ema_env:
                        self.current_task_ema_env[name] = grad_e.clone()
                    else:
                        self.current_task_ema_env[name] = (
                            momentum * self.current_task_ema_env[name] + (1 - momentum) * grad_e
                        )

    def apply_freeze(self, model, optimizer=None, current_task_id=None, debug=False):
        candidate_items = []
        total_count = 0

        for name, param in model.named_parameters():
            if (
                name not in self.grad_history_causal
                or name not in self.grad_history_env
                or len(self.grad_history_causal[name]) < 1
                or len(self.grad_history_env[name]) < 1
            ):
                continue

            # Build variance stacks: include current_task_ema if available
            # (apply_freeze is called BEFORE commit_task_signature so current_task_ema
            # still holds the current task's EMA gradients, providing at least 2 data
            # points for meaningful variance computation starting from Task 1).
            causal_list = list(self.grad_history_causal[name])
            env_list = list(self.grad_history_env[name])

            if name in self.current_task_ema_causal:
                causal_list.append(self.current_task_ema_causal[name].cpu().numpy())
            if name in self.current_task_ema_env:
                env_list.append(self.current_task_ema_env[name].cpu().numpy())

            if len(causal_list) < 2 or len(env_list) < 2:
                continue

            total_count += 1
            causal_stack = np.stack(causal_list)
            env_stack = np.stack(env_list)

            var_causal = float(np.var(causal_stack, axis=0).mean())
            var_env = float(np.var(env_stack, axis=0).mean())
            mag_causal = float(np.abs(causal_stack[-1]).mean())
            mag_env = float(np.abs(env_stack[-1]).mean())

            env_dominance = mag_env / (mag_causal + 1e-12)

            if debug and total_count <= 10:
                print(f"  {name[:40]}: var_c={var_causal:.2e}<{self.threshold:.0e}?{cond1}, "
                      f"mag_c={mag_causal:.2e}<{self.causal_max_grad:.0e}?{cond2}, "
                      f"mag_e={mag_env:.2e}>{self.env_min_grad:.0e}?{cond3}, "
                      f"var_e={var_env:.2e}>{self.threshold:.0e}?{cond4}")

            if cond1 and cond2 and cond3 and cond4:
                score = var_causal + mag_causal
                candidate_items.append((score, name, param))

        max_freeze = int(max(0, self.max_freeze_ratio) * total_count)
        max_freeze = min(max_freeze, len(candidate_items))

        if debug:
            print(f"\n[RoAda Debug] Task {current_task_id}: {total_count} evaluable params, "
                  f"{len(candidate_items)} candidates, max_freeze={max_freeze}")

        frozen_count = 0
        if max_freeze > 0:
            candidate_items.sort(key=lambda x: x[0])
            for _, name, param in candidate_items[:max_freeze]:
                self.frozen_flags[name] = True
                param.requires_grad = False
                frozen_count += 1
                    
        if self.logger:
            self.logger.info(
                f"[Causal-Invariant RoAda] Task {current_task_id}: "
                f"{frozen_count} / {total_count} evaluable parameters frozen "
                f"(candidates={len(candidate_items)}, max_ratio={self.max_freeze_ratio:.2f}, "
                f"causal_max_grad={self.causal_max_grad:.2e}, env_min_grad={self.env_min_grad:.2e}, "
                f"causal_env_ratio={self.causal_env_ratio:.2f})."
            )
        return frozen_count