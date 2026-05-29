import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq


class EWC:
    """Elastic Weight Consolidation for protecting important parameters"""
    def __init__(self, model, device, ewc_lambda=5000):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.params = {n: p.clone().detach().to(device) for n, p in model.named_parameters() if p.requires_grad}
        self._fisher = None

    def compute_fisher(self, dataloader, device, num_samples=200, task_id=None):
        """Compute Fisher Information Matrix diagonal approximation"""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        samples = 0
        for batch_x, batch_y in dataloader:
            if samples >= num_samples:
                break
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            self.model.zero_grad()
            output = self.model(batch_y, torch.zeros(batch_y.shape[0], device=device), batch_x, task_id=task_id)
            loss = output[0].mean() if isinstance(output, tuple) else output.mean()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2
            samples += batch_x.size(0)

        for n in fisher:
            fisher[n] /= samples
        self._fisher = fisher
        self.model.train()
        return fisher

    def penalty(self):
        """EWC penalty: sum_i Fisher_i * (param_i - old_param_i)^2"""
        if self._fisher is None:
            return torch.tensor(0.0, device=self.device)
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._fisher:
                loss += (self._fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.ewc_lambda * loss

    def update_params(self):
        """Save current parameters as old parameters for next task"""
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

class CausalDecipher(nn.Module):
    """
    Nucleus Energy Masking: adaptively split causal / environment regions
    per sample based on cumulative attention energy, replacing the fixed
    env_ratio.  Analogous to nucleus (top-p) sampling in NLP —— accumulate
    patches from highest importance until their energy reaches
    `nucleus_energy_p` of the total, mark them as causal core, and treat
    the rest as environment.
    """
    def __init__(self, nucleus_energy_p=0.2, patch_size=4, temperature=1.0,
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
    def __init__(self, memory_capacity=1000, causal_keep_ratio=0.2, patch_size=4,
                 logger=None, memory_policy="hybrid", memory_per_task_capacity=0,
                 task_balanced_replay=True, replay_age_alpha=0.0,
                 memory_random_ratio=0.5):
        self.capacity = memory_capacity
        self.causal_keep_ratio = causal_keep_ratio
        self.patch_size = patch_size
        self.logger = logger
        self.memory_policy = memory_policy
        self.memory_per_task_capacity = max(0, int(memory_per_task_capacity))
        self.task_balanced_replay = task_balanced_replay
        self.replay_age_alpha = max(0.0, float(replay_age_alpha))
        self.memory_random_ratio = max(0.0, min(float(memory_random_ratio), 1.0))

        # 修复：每个任务维护独立的 buffer，避免新任务数据覆盖旧任务数据
        self.task_buffers: dict[int, list] = {}  # task_id -> list of samples
        self.task_sample_counts: dict[int, int] = {}  # task_id -> 已采样次数（用于轮询）
        self.tiebreaker = 0
        self.seen_samples = 0

    def _per_task_capacity(self):
        if self.memory_per_task_capacity > 0:
            return self.memory_per_task_capacity
        num_tasks = max(1, len(self.task_buffers))
        return max(1, int(np.ceil(float(self.capacity) / float(num_tasks))))

    def _rebalance_capacity(self):
        if self.capacity <= 0:
            return
        per_task_capacity = self._per_task_capacity()
        for task_id, buf in self.task_buffers.items():
            if len(buf) > per_task_capacity:
                self.task_buffers[task_id] = heapq.nlargest(per_task_capacity, buf)
                heapq.heapify(self.task_buffers[task_id])

    @property
    def buffer_history(self):
        # Backward compatibility: return flattened all buffers
        all_items = []
        for buf in self.task_buffers.values():
            all_items.extend(buf)
        return all_items

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

    def update_memory(self, history_batch, target_batch, importance_scores, causal_mask=None, valid_mask=None, task_id=0):
        """保存样本到指定任务的 buffer，旧任务数据不会被新任务覆盖"""
        if self.capacity == 0:
            return  # memory_capacity=0 时禁用存储
        if importance_scores.dim() == 2:
            importance_scores = importance_scores.unsqueeze(1)
        B, T, N = importance_scores.shape
        k = max(1, int(N * self.causal_keep_ratio))
        topk_scores, _ = torch.topk(importance_scores, k, dim=-1)
        sample_causal_scores = topk_scores.reshape(B, -1).mean(dim=-1)

        # 初始化任务的 buffer
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = []
            self.task_sample_counts[task_id] = 0
            self._rebalance_capacity()

        # 修复：每个任务有独立的固定容量，不受任务数量影响
        # 用堆的负分数实现最大堆（heapq是最小堆）
        per_task_capacity = self._per_task_capacity()

        for i in range(len(history_batch)):
            self.seen_samples += 1
            self.tiebreaker += 1
            causal_score = float(sample_causal_scores[i].item())
            random_score = float(np.random.random())
            if self.memory_policy == "reservoir":
                score = random_score
            elif self.memory_policy == "hybrid":
                score = (
                    (1.0 - self.memory_random_ratio) * causal_score
                    + self.memory_random_ratio * random_score
                )
            else:
                score = causal_score
            vm = valid_mask[i].detach().cpu() if valid_mask is not None else None
            # 添加 task_id 标识
            item = (score, self.tiebreaker,
                    history_batch[i].detach().cpu(),
                    target_batch[i].detach().cpu(),
                    importance_scores[i].detach().cpu(),
                    vm,
                    task_id)  # 添加 task_id

            if len(self.task_buffers[task_id]) < per_task_capacity:
                heapq.heappush(self.task_buffers[task_id], item)
            else:
                if score > self.task_buffers[task_id][0][0]:
                    heapq.heapreplace(self.task_buffers[task_id], item)

    def _legacy_get_replay_data(self, batch_size, current_task_id=0):
        """从所有旧任务（task_id < current_task_id）的 buffer 中均匀采样回放数据"""
        # 收集所有旧任务的 buffer
        old_task_buffers = {tid: buf for tid, buf in self.task_buffers.items() if tid < current_task_id}

        if not old_task_buffers or all(len(buf) == 0 for buf in old_task_buffers.values()):
            return None, None, None, None

        # 收集所有旧任务的有效样本
        available_tasks = [tid for tid, buf in old_task_buffers.items() if len(buf) > 0]
        if not available_tasks:
            return None, None, None, None

        # 策略1: 从每个旧任务均匀采样（如果样本充足）
        samples_per_task = max(1, batch_size // len(available_tasks))
        min_task_samples = min(len(old_task_buffers[tid]) for tid in available_tasks)

        if min_task_samples >= samples_per_task:
            # 每个任务独立采样然后合并
            all_replay_targets, all_replay_history, all_replay_scores, all_replay_valid = [], [], [], []
            for tid in available_tasks:
                buf = old_task_buffers[tid]
                indices = np.random.choice(len(buf), samples_per_task, replace=False)
                for idx in indices:
                    item = buf[idx]
                    if len(item) > 6:
                        _, _, t_hist, t_tgt, t_score, t_valid, _ = item
                    else:
                        _, _, t_hist, t_tgt, t_score = item
                        t_valid = None
                    all_replay_targets.append(t_tgt)
                    all_replay_history.append(t_hist)
                    all_replay_scores.append(t_score)
                    all_replay_valid.append(t_valid)

            # 打乱顺序
            perm_idx = np.random.permutation(len(all_replay_targets))
            batch_target = [all_replay_targets[i] for i in perm_idx]
            batch_history = [all_replay_history[i] for i in perm_idx]
            batch_scores = [all_replay_scores[i] for i in perm_idx]
            batch_valid = [all_replay_valid[i] for i in perm_idx]
        else:
            # 策略2: 合并所有任务样本，统一采样
            all_items = []
            for buf in old_task_buffers.values():
                all_items.extend(buf)

            if len(all_items) == 0:
                return None, None, None, None

            # 按任务分层采样（尽量保持均衡）
            num_samples = min(batch_size, len(all_items))
            selected_items = []

            if len(available_tasks) > 0:
                samples_per_task_actual = max(1, num_samples // len(available_tasks))
                for tid in available_tasks:
                    buf = old_task_buffers[tid]
                    n = min(samples_per_task_actual, len(buf))
                    indices = np.random.choice(len(buf), n, replace=False)
                    selected_items.extend([buf[i] for i in indices])

            # 补充采样（如果不够）
            while len(selected_items) < num_samples and len(all_items) > len(selected_items):
                remaining = [item for item in all_items if item not in selected_items]
                if not remaining:
                    break
                selected_items.append(remaining[np.random.randint(0, len(remaining))])

            selected_items = selected_items[:num_samples]
            np.random.shuffle(selected_items)

            batch_target, batch_history, batch_scores, batch_valid = [], [], [], []
            for item in selected_items:
                if len(item) > 6:
                    _, _, t_hist, t_tgt, t_score, t_valid, _ = item
                else:
                    _, _, t_hist, t_tgt, t_score = item
                    t_valid = None
                batch_target.append(t_tgt)
                batch_history.append(t_hist)
                batch_scores.append(t_score)
                batch_valid.append(t_valid)

        valid_stacked = torch.stack(batch_valid) if batch_valid[0] is not None else None
        return torch.stack(batch_target), torch.stack(batch_history), torch.stack(batch_scores), valid_stacked

    def _unpack_item(self, item):
        if len(item) > 6:
            _, _, t_hist, t_tgt, t_score, t_valid, _ = item
            t_task_id = item[-1]
        else:
            _, _, t_hist, t_tgt, t_score = item
            t_valid = None
            t_task_id = -1
        return t_hist, t_tgt, t_score, t_valid, t_task_id

    def get_replay_data(self, batch_size, current_task_id=0):
        """Return a replay mini-batch with balanced coverage over previous tasks."""
        old_task_buffers = {tid: buf for tid, buf in self.task_buffers.items() if tid < current_task_id}
        available_tasks = [tid for tid, buf in old_task_buffers.items() if len(buf) > 0]
        if not available_tasks:
            return None, None, None, None

        selected_items = []
        if self.task_balanced_replay:
            shuffled_tasks = list(available_tasks)
            np.random.shuffle(shuffled_tasks)
            if self.replay_age_alpha > 0:
                weights = np.asarray(
                    [max(1, current_task_id - tid) ** self.replay_age_alpha for tid in shuffled_tasks],
                    dtype=np.float64,
                )
                weights = weights / max(weights.sum(), 1e-12)
                raw_counts = weights * batch_size
                counts = np.floor(raw_counts).astype(np.int64)
                if batch_size >= len(shuffled_tasks):
                    counts = np.maximum(counts, 1)
                diff = int(batch_size - counts.sum())
                if diff > 0:
                    frac_order = np.argsort(-(raw_counts - np.floor(raw_counts)))
                    for j in frac_order[:diff]:
                        counts[j] += 1
                elif diff < 0:
                    for j in np.argsort(raw_counts - np.floor(raw_counts)):
                        if diff == 0:
                            break
                        min_count = 1 if batch_size >= len(shuffled_tasks) else 0
                        if counts[j] > min_count:
                            counts[j] -= 1
                            diff += 1
            else:
                base = max(1, batch_size // len(shuffled_tasks))
                remainder = max(0, batch_size - base * len(shuffled_tasks))
                counts = np.asarray([
                    base + (1 if idx < remainder else 0)
                    for idx in range(len(shuffled_tasks))
                ], dtype=np.int64)

            for tid, n in zip(shuffled_tasks, counts):
                if n <= 0:
                    continue
                buf = old_task_buffers[tid]
                replace = len(buf) < n
                indices = np.random.choice(len(buf), n, replace=replace)
                selected_items.extend([buf[i] for i in indices])
        else:
            all_items = []
            for buf in old_task_buffers.values():
                all_items.extend(buf)
            if not all_items:
                return None, None, None, None
            n = min(batch_size, len(all_items))
            indices = np.random.choice(len(all_items), n, replace=False)
            selected_items = [all_items[i] for i in indices]

        if not selected_items:
            return None, None, None, None

        selected_items = selected_items[:batch_size]
        np.random.shuffle(selected_items)

        batch_target, batch_history, batch_scores, batch_valid, batch_task_ids = [], [], [], [], []
        for item in selected_items:
            t_hist, t_tgt, t_score, t_valid, t_task_id = self._unpack_item(item)
            batch_target.append(t_tgt)
            batch_history.append(t_hist)
            batch_scores.append(t_score)
            batch_valid.append(t_valid)
            batch_task_ids.append(int(t_task_id))

        valid_stacked = torch.stack(batch_valid) if all(v is not None for v in batch_valid) else None
        task_ids = torch.tensor(batch_task_ids, dtype=torch.long)
        return torch.stack(batch_target), torch.stack(batch_history), torch.stack(batch_scores), valid_stacked, task_ids


class CausalRoAdaController:
    def __init__(self, var_threshold=1e-4, env_var_threshold=1e-3, min_grad=0.0, max_freeze_ratio=0.5,
                 logger=None, causal_env_ratio=2.5, causal_max_grad=None, env_min_grad=None,
                 protection_mode="hard", selection_mode="percentile",
                 protect_ratio=0.1, tensor_protect_ratio=0.2,
                 soft_scale=0.3, reg_lambda=10.0,
                 use_replay_aware=True, replay_protect_ratio=0.03,
                 replay_soft_scale=0.5, replay_conflict_weight=1.0,
                 use_conflict_surgery=False, conflict_soft_scale=0.0):
        self.threshold = var_threshold  # for causal gradient variance
        self.env_var_threshold = env_var_threshold  # separate threshold for env gradient variance
        self.min_grad = min_grad
        # Use sensible defaults: causal gradients should be small (~0.05), env should be active
        self.causal_max_grad = float(0.05 if causal_max_grad is None else causal_max_grad)
        self.env_min_grad = float(1e-4 if env_min_grad is None else env_min_grad)
        self.max_freeze_ratio = max_freeze_ratio
        self.causal_env_ratio = causal_env_ratio
        self.protection_mode = protection_mode
        self.selection_mode = selection_mode
        self.protect_ratio = max(0.0, min(float(protect_ratio), 1.0))
        self.tensor_protect_ratio = max(0.0, min(float(tensor_protect_ratio), 1.0))
        self.soft_scale = max(0.0, min(float(soft_scale), 1.0))
        self.reg_lambda = float(reg_lambda)
        self.use_replay_aware = bool(use_replay_aware)
        self.replay_protect_ratio = max(0.0, min(float(replay_protect_ratio), 1.0))
        self.replay_soft_scale = max(0.0, min(float(replay_soft_scale), 1.0))
        self.replay_conflict_weight = max(0.0, float(replay_conflict_weight))
        self.use_conflict_surgery = bool(use_conflict_surgery)
        self.conflict_soft_scale = max(0.0, min(float(conflict_soft_scale), 1.0))
        self.logger = logger
        self.grad_history_causal = {}
        self.grad_history_env = {}
        self.current_task_ema_causal = {}
        self.current_task_ema_env = {}
        self.replay_grad_ema = {}
        self.replay_signed_grad_ema = {}
        self.replay_protected_masks = {}
        self.replay_mask_refresh_count = 0
        self.frozen_flags = {}
        self.protected_masks = {}
        self.anchor_params = {}

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

    def update_replay_signature(self, model, replay_loss, current_loss=None,
                                momentum=0.95, conflict_weight=None):
        """Track old-task importance from replay gradients and refresh soft masks."""
        if not self.use_replay_aware or replay_loss is None or not getattr(replay_loss, 'requires_grad', False):
            return 0

        named_params = [
            (name, p) for name, p in model.named_parameters()
            if p.requires_grad and not any(name.startswith(prefix) for prefix in self._protected_prefixes())
        ]
        if not named_params:
            return 0

        params = [p for _, p in named_params]
        grads_replay = torch.autograd.grad(
            replay_loss, params, retain_graph=True, allow_unused=True
        )
        grads_current = None
        if current_loss is not None and getattr(current_loss, 'requires_grad', False):
            grads_current = torch.autograd.grad(
                current_loss, params, retain_graph=True, allow_unused=True
            )

        if conflict_weight is None:
            conflict_weight = self.replay_conflict_weight

        updated = 0
        for idx, ((name, _), grad_replay) in enumerate(zip(named_params, grads_replay)):
            if grad_replay is None:
                continue
            grad_replay = grad_replay.detach()
            if not torch.isfinite(grad_replay).all():
                continue

            grad_replay_f = grad_replay.float()
            score = grad_replay_f.abs()
            if grads_current is not None and grads_current[idx] is not None and conflict_weight > 0:
                grad_current = grads_current[idx].detach()
                if torch.isfinite(grad_current).all():
                    grad_current_f = grad_current.float()
                    conflict = torch.relu(-(grad_replay_f * grad_current_f))
                    conflict = conflict / (grad_replay_f.abs() * grad_current_f.abs() + 1e-12)
                    score = score * (1.0 + conflict_weight * conflict.clamp(max=1.0))

            score = score.detach().cpu()
            signed_grad = grad_replay_f.detach().cpu()
            if name not in self.replay_grad_ema:
                self.replay_grad_ema[name] = score
                self.replay_signed_grad_ema[name] = signed_grad
            else:
                self.replay_grad_ema[name] = (
                    momentum * self.replay_grad_ema[name] + (1.0 - momentum) * score
                )
                if name in self.replay_signed_grad_ema and self.replay_signed_grad_ema[name].shape == signed_grad.shape:
                    self.replay_signed_grad_ema[name] = (
                        momentum * self.replay_signed_grad_ema[name] + (1.0 - momentum) * signed_grad
                    )
                else:
                    self.replay_signed_grad_ema[name] = signed_grad
            updated += 1

        if updated > 0:
            self._refresh_replay_masks(model)
        return updated

    def _refresh_replay_masks(self, model):
        if self.replay_protect_ratio <= 0:
            self.replay_protected_masks.clear()
            return 0

        protected_tensors = 0
        protected_elements = 0
        protected_names = []
        for name, param in model.named_parameters():
            score = self.replay_grad_ema.get(name)
            if score is None or score.shape != param.shape:
                continue

            flat_score = score.reshape(-1)
            finite = torch.isfinite(flat_score)
            if not bool(finite.any().item()):
                continue

            eligible_idx = torch.nonzero(finite, as_tuple=False).flatten()
            k = int(np.ceil(self.replay_protect_ratio * eligible_idx.numel()))
            k = max(1, min(k, eligible_idx.numel()))
            eligible_scores = flat_score[eligible_idx]
            top_local = torch.topk(eligible_scores, k=k, largest=True).indices
            selected = eligible_idx[top_local]

            mask = torch.zeros(param.numel(), dtype=torch.bool)
            mask[selected] = True
            mask = mask.reshape(param.shape)
            self.replay_protected_masks[name] = mask
            protected_tensors += 1
            protected_elements += int(mask.sum().item())
            protected_names.append(name)

        self.replay_mask_refresh_count += 1
        should_log = (
            self.replay_mask_refresh_count <= 3
            or self.replay_mask_refresh_count % 100 == 0
        )
        if self.logger and protected_tensors > 0 and should_log:
            self.logger.info(
                f"[Replay-Aware RoAda] refreshed masks: "
                f"{protected_tensors} tensors, {protected_elements} elements, "
                f"ratio={self.replay_protect_ratio:.3f}, soft_scale={self.replay_soft_scale:.2f}"
            )
            protected_by_layer = {}
            for name in protected_names:
                layer_type = name.split('.')[0] if '.' in name else name
                protected_by_layer[layer_type] = protected_by_layer.get(layer_type, 0) + 1
            self.logger.info(f"[Replay-Aware RoAda] protected tensors by layer: {protected_by_layer}")

        return protected_elements

    @staticmethod
    def _protected_prefixes():
        return (
            'condition_encoder',
            'task_adapter',
            'task_heads',
            'x_embedder',
            'pos_embed',
            't_embedder',
            'temporal_pos_embed',
        )

    @staticmethod
    def _robust_norm(array):
        array = np.asarray(array, dtype=np.float32)
        finite = np.abs(array[np.isfinite(array)])
        if finite.size == 0:
            return np.zeros_like(array, dtype=np.float32)

        scale = float(np.median(finite))
        if scale < 1e-12:
            scale = float(np.mean(finite)) + 1e-12
        return array / (scale + 1e-12)

    def _get_signature_lists(self, name):
        causal_list = list(self.grad_history_causal.get(name, []))
        env_list = list(self.grad_history_env.get(name, []))

        if name in self.current_task_ema_causal:
            causal_list.append(self.current_task_ema_causal[name].detach().cpu().numpy())
        if name in self.current_task_ema_env:
            env_list.append(self.current_task_ema_env[name].detach().cpu().numpy())

        return causal_list, env_list

    def _build_element_scores(self, causal_list, env_list):
        eps = 1e-12

        if len(causal_list) >= 2 and len(env_list) >= 2:
            causal_stack = np.stack(causal_list, axis=0).astype(np.float32, copy=False)
            env_stack = np.stack(env_list, axis=0).astype(np.float32, copy=False)

            var_causal = np.var(causal_stack, axis=0)
            var_env = np.var(env_stack, axis=0)
            mag_causal = np.abs(causal_stack[-1])
            mag_env = np.abs(env_stack[-1])
            dominance = mag_env / (mag_causal + eps)

            score = (
                np.log1p(self._robust_norm(mag_env))
                + np.log1p(self._robust_norm(dominance))
                - np.log1p(self._robust_norm(mag_causal))
                - np.log1p(self._robust_norm(var_causal))
                - 0.5 * np.log1p(self._robust_norm(var_env))
            )
            threshold_mask = (
                (var_causal < self.threshold)
                & (mag_causal < self.causal_max_grad)
                & (mag_env > self.env_min_grad)
                & (var_env < self.env_var_threshold)
            )
        else:
            mag_causal = np.abs(np.asarray(causal_list[-1], dtype=np.float32))
            mag_env = np.abs(np.asarray(env_list[-1], dtype=np.float32))
            dominance = mag_env / (mag_causal + eps)

            score = (
                np.log1p(self._robust_norm(dominance))
                + 0.5 * np.log1p(self._robust_norm(mag_env))
                - np.log1p(self._robust_norm(mag_causal))
            )
            threshold_mask = dominance > self.causal_env_ratio

        return np.nan_to_num(
            score,
            nan=-np.inf,
            posinf=np.finfo(np.float32).max,
            neginf=-np.inf,
        ), threshold_mask

    def _select_local_indices(self, scores, threshold_mask):
        if self.protect_ratio <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        flat_scores = scores.reshape(-1)
        eligible = np.isfinite(flat_scores)

        if self.selection_mode == "threshold":
            eligible &= threshold_mask.reshape(-1)

        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        k = int(np.ceil(self.protect_ratio * flat_scores.size))
        k = max(1, min(k, eligible_indices.size))

        if self.selection_mode == "random_top_tensor":
            selected_indices = np.random.choice(eligible_indices, size=k, replace=False)
            random_scores = np.random.random(k).astype(np.float32)
            return selected_indices.astype(np.int64), random_scores

        eligible_scores = flat_scores[eligible_indices]

        if k < eligible_indices.size:
            local_choice = np.argpartition(eligible_scores, -k)[-k:]
            selected_indices = eligible_indices[local_choice]
        else:
            selected_indices = eligible_indices

        selected_scores = flat_scores[selected_indices]
        return selected_indices.astype(np.int64), selected_scores.astype(np.float32, copy=False)

    def _tensor_score(self, scores, threshold_mask):
        flat_scores = scores.reshape(-1)
        eligible = np.isfinite(flat_scores)

        if self.selection_mode == "threshold":
            eligible &= threshold_mask.reshape(-1)

        eligible_scores = flat_scores[eligible]
        if eligible_scores.size == 0:
            return -np.inf

        k = int(np.ceil(self.protect_ratio * eligible_scores.size))
        k = max(1, min(k, eligible_scores.size))
        if k < eligible_scores.size:
            top_scores = np.partition(eligible_scores, -k)[-k:]
        else:
            top_scores = eligible_scores

        return float(np.mean(top_scores))

    def apply_protection(self, model, optimizer=None, current_task_id=None, debug=False):
        if self.protection_mode == "hard":
            return self.apply_freeze(model, optimizer, current_task_id, debug)
        return self.apply_soft_protection(model, current_task_id=current_task_id, debug=debug)

    def _update_anchor(self, name, param, old_mask, new_mask):
        new_only_mask = new_mask if old_mask is None else (new_mask & ~old_mask)
        if not bool(new_only_mask.any().item()):
            return

        current_value = param.detach()
        if name not in self.anchor_params:
            anchor = current_value.clone()
        else:
            anchor = self.anchor_params[name].to(param.device).clone()
            anchor[new_only_mask] = current_value[new_only_mask]
        self.anchor_params[name] = anchor.detach()

    def regularization_loss(self, model):
        if self.protection_mode != "regularize" or not self.protected_masks:
            return None

        reg_loss = None
        total_elements = 0
        for name, param in model.named_parameters():
            mask = self.protected_masks.get(name)
            anchor = self.anchor_params.get(name)
            if mask is None or anchor is None or mask.shape != param.shape:
                continue

            mask = mask.to(param.device)
            if not bool(mask.any().item()):
                continue

            anchor = anchor.to(param.device)
            masked_loss = (param - anchor).pow(2).masked_select(mask).sum()
            reg_loss = masked_loss if reg_loss is None else reg_loss + masked_loss
            total_elements += int(mask.sum().item())

        if reg_loss is None or total_elements <= 0:
            return None

        return reg_loss / float(total_elements)

    def apply_soft_protection(self, model, current_task_id=None, debug=False):
        total_tensors = 0
        total_elements = 0
        skipped_no_history = 0
        skipped_no_env_history = 0
        skipped_no_enough_data = 0
        skipped_no_candidate = 0
        candidate_items = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in self._protected_prefixes()):
                continue

            if name not in self.grad_history_causal:
                skipped_no_history += 1
                continue
            if name not in self.grad_history_env:
                skipped_no_env_history += 1
                continue

            causal_list, env_list = self._get_signature_lists(name)
            if len(causal_list) < 1 or len(env_list) < 1:
                skipped_no_history += 1
                continue

            total_tensors += 1
            total_elements += param.numel()

            if len(causal_list) < 2 or len(env_list) < 2:
                skipped_no_enough_data += 1

            scores, threshold_mask = self._build_element_scores(causal_list, env_list)
            selected_indices, selected_scores = self._select_local_indices(scores, threshold_mask)

            old_mask = self.protected_masks.get(name)
            if old_mask is not None and selected_indices.size > 0:
                old_flat = old_mask.detach().cpu().numpy().reshape(-1)
                is_new = ~old_flat[selected_indices]
                selected_indices = selected_indices[is_new]
                selected_scores = selected_scores[is_new]

            if selected_indices.size == 0:
                skipped_no_candidate += 1
                continue

            tensor_score = self._tensor_score(scores, threshold_mask)
            if not np.isfinite(tensor_score):
                skipped_no_candidate += 1
                continue

            candidate_items.append((tensor_score, name, param, selected_indices, selected_scores))

        raw_candidate_tensors = len(candidate_items)
        skipped_tensor_budget = 0
        tensor_filter_modes = ("top_tensor", "bottom_tensor", "random_top_tensor")
        if self.selection_mode in tensor_filter_modes and raw_candidate_tensors > 0:
            max_candidate_tensors = int(np.ceil(self.tensor_protect_ratio * total_tensors))
            max_candidate_tensors = max(1, min(max_candidate_tensors, raw_candidate_tensors))
            skipped_tensor_budget = raw_candidate_tensors - max_candidate_tensors
            if self.selection_mode == "top_tensor":
                candidate_items.sort(key=lambda item: item[0], reverse=True)
                candidate_items = candidate_items[:max_candidate_tensors]
            elif self.selection_mode == "bottom_tensor":
                candidate_items.sort(key=lambda item: item[0])
                candidate_items = candidate_items[:max_candidate_tensors]
            else:
                selected = np.random.choice(
                    raw_candidate_tensors,
                    size=max_candidate_tensors,
                    replace=False,
                )
                candidate_items = [candidate_items[i] for i in selected]

        max_protected_elements = int(max(0.0, self.max_freeze_ratio) * total_elements)
        existing_protected_elements = sum(int(mask.sum().item()) for mask in self.protected_masks.values())
        remaining_budget = max(0, max_protected_elements - existing_protected_elements)
        candidate_elements = sum(item[3].size for item in candidate_items)
        skipped_budget = max(0, candidate_elements - remaining_budget)

        cutoff = -np.inf
        if remaining_budget <= 0:
            candidate_items = []
        elif candidate_elements > remaining_budget:
            all_scores = np.concatenate([item[4] for item in candidate_items], axis=0)
            cutoff = float(np.partition(all_scores, -remaining_budget)[-remaining_budget])

        new_protected_tensors = 0
        new_protected_elements = 0
        protected_names = []

        for _, name, param, selected_indices, selected_scores in candidate_items:
            if cutoff != -np.inf:
                keep = selected_scores >= cutoff
                selected_indices = selected_indices[keep]
            if selected_indices.size == 0:
                continue

            new_mask_np = np.zeros(param.numel(), dtype=np.bool_)
            new_mask_np[selected_indices] = True
            new_mask = torch.from_numpy(new_mask_np.reshape(param.shape)).to(param.device)

            old_mask = self.protected_masks.get(name)
            if old_mask is not None:
                old_mask = old_mask.to(param.device)
                before = int(old_mask.sum().item())
                combined_mask = old_mask | new_mask
                added = int(combined_mask.sum().item()) - before
                self.protected_masks[name] = combined_mask
            else:
                combined_mask = new_mask
                added = int(combined_mask.sum().item())
                self.protected_masks[name] = combined_mask

            if self.protection_mode == "regularize":
                self._update_anchor(name, param, old_mask, new_mask)

            if added > 0:
                new_protected_tensors += 1
                new_protected_elements += added
                protected_names.append(name)

        total_protected_elements = sum(int(mask.sum().item()) for mask in self.protected_masks.values())

        if self.logger:
            log_name = "CausalRoAda-Regularize" if self.protection_mode == "regularize" else "CausalRoAda-Soft"
            self.logger.info(
                f"[{log_name}] Task {current_task_id}: "
                f"{new_protected_tensors}/{total_tensors} tensors newly protected, "
                f"{new_protected_elements}/{total_elements} new elements protected, "
                f"{total_protected_elements}/{total_elements} total protected in eligible tensors "
                f"(mode={self.selection_mode}, tensor_ratio={self.tensor_protect_ratio:.2f}, "
                f"local_ratio={self.protect_ratio:.2f}, "
                f"max_ratio={self.max_freeze_ratio:.2f}, grad_scale={self.soft_scale:.2f}, "
                f"reg_lambda={self.reg_lambda:.2e}, "
                f"candidate_tensors={raw_candidate_tensors}, candidates={candidate_elements}, "
                f"skipped_tensor_budget={skipped_tensor_budget}, skipped_budget={skipped_budget}, "
                f"skipped(no_causal)={skipped_no_history}, "
                f"skipped(no_env)={skipped_no_env_history}, "
                f"skipped(no_data)={skipped_no_enough_data}, "
                f"skipped(no_candidate)={skipped_no_candidate})."
            )
            if protected_names:
                action = "Regularized" if self.protection_mode == "regularize" else "Soft-protected"
                protected_by_layer = {}
                for name in protected_names:
                    layer_type = name.split('.')[0] if '.' in name else name
                    protected_by_layer[layer_type] = protected_by_layer.get(layer_type, 0) + 1
                self.logger.info(f"[RoAda Debug] {action} params by layer: {protected_by_layer}")
                self.logger.info(
                    f"[RoAda Debug] {action} param names: {protected_names[:10]}"
                    f"{'...' if len(protected_names) > 10 else ''}"
                )

        return new_protected_tensors

    def apply_gradient_protection(self, model):
        protected_elements = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            if self.protection_mode == "soft":
                mask = self.protected_masks.get(name)
                if mask is not None and mask.shape == param.grad.shape:
                    mask = mask.to(param.grad.device)
                    if self.soft_scale <= 0:
                        param.grad.masked_fill_(mask, 0.0)
                    elif self.soft_scale < 1:
                        param.grad[mask] = param.grad[mask] * self.soft_scale
                    protected_elements += int(mask.sum().item())

            replay_mask = self.replay_protected_masks.get(name)
            if replay_mask is not None and replay_mask.shape == param.grad.shape:
                replay_mask = replay_mask.to(param.grad.device)
                if self.use_conflict_surgery:
                    replay_direction = self.replay_signed_grad_ema.get(name)
                    if replay_direction is not None and replay_direction.shape == param.grad.shape:
                        replay_direction = replay_direction.to(param.grad.device)
                        conflict_mask = replay_mask & ((param.grad.float() * replay_direction.float()) < 0)
                        if self.conflict_soft_scale <= 0:
                            param.grad.masked_fill_(conflict_mask, 0.0)
                        elif self.conflict_soft_scale < 1:
                            param.grad[conflict_mask] = param.grad[conflict_mask] * self.conflict_soft_scale
                        protected_elements += int(conflict_mask.sum().item())
                else:
                    if self.replay_soft_scale <= 0:
                        param.grad.masked_fill_(replay_mask, 0.0)
                    elif self.replay_soft_scale < 1:
                        param.grad[replay_mask] = param.grad[replay_mask] * self.replay_soft_scale
                    protected_elements += int(replay_mask.sum().item())

        return protected_elements

    def get_protected_ratio(self, model):
        total = sum(p.numel() for p in model.parameters())
        if total <= 0:
            return 0

        replay_protected = sum(int(mask.sum().item()) for mask in self.replay_protected_masks.values())
        if self.protection_mode == "hard":
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            return (frozen + replay_protected) / total

        protected = sum(int(mask.sum().item()) for mask in self.protected_masks.values())
        protected += replay_protected
        return protected / total

    def apply_freeze(self, model, optimizer=None, current_task_id=None, debug=False):
        candidate_items = []
        total_count = 0
        skipped_no_history = 0
        skipped_no_enough_data = 0

        # 不应该被冻结的模块（这些模块对所有任务都应该是可学习的）
        protected_prefixes = (
            'condition_encoder',  # 编码历史条件，任务无关
            'task_adapter',       # 任务适配器，专门负责任务分离
            'x_embedder',         # 输入嵌入层
            'pos_embed',           # 位置编码
            't_embedder',          # 时间步嵌入
            'temporal_pos_embed',  # 时间位置编码
        )

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # 跳过不应该被冻结的模块
            if any(name.startswith(prefix) for prefix in protected_prefixes):
                continue

            if (
                name not in self.grad_history_causal
                or name not in self.grad_history_env
                or len(self.grad_history_causal[name]) < 1
                or len(self.grad_history_env[name]) < 1
            ):
                skipped_no_history += 1
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
                skipped_no_enough_data += 1
                # 修复：当只有1个数据点时，简化判断逻辑
                # 核心思想：环境主导度越高，越应该冻结（因为是环境相关参数）
                if len(causal_list) >= 1 and len(env_list) >= 1:
                    total_count += 1
                    mag_causal = float(np.abs(causal_list[-1]).mean())
                    mag_env = float(np.abs(env_list[-1]).mean())
                    env_dominance = mag_env / (mag_causal + 1e-12)

                    # 阈值设为2.0：宽松于之前的3.0（冻结了3个），严格于0.5（冻结了42-45个）
                    # 目的是冻结适量参数，避免过度冻结挤压新任务学习空间
                    if env_dominance > 2.0:
                        score = env_dominance  # 越高越优先冻结
                        candidate_items.append((score, name, param))
                continue

            total_count += 1
            causal_stack = np.stack(causal_list)
            env_stack = np.stack(env_list)

            var_causal = float(np.var(causal_stack, axis=0).mean())
            var_env = float(np.var(env_stack, axis=0).mean())
            mag_causal = float(np.abs(causal_stack[-1]).mean())
            mag_env = float(np.abs(env_stack[-1]).mean())

            env_dominance = mag_env / (mag_causal + 1e-12)

            cond1 = var_causal < self.threshold
            cond2 = mag_causal < self.causal_max_grad
            cond3 = mag_env > self.env_min_grad
            cond4 = var_env < self.env_var_threshold
            # 移除cond5的环境主导比例限制：只要因果梯度稳定+环境梯度活跃，就纳入候选
            # 评分时综合考虑方差和幅值，越小越优先冻结

            if debug and total_count <= 10:
                print(f"  {name[:40]}: var_c={var_causal:.2e}<{self.threshold:.0e}?{cond1}, "
                      f"mag_c={mag_causal:.2e}<{self.causal_max_grad:.0e}?{cond2}, "
                      f"mag_e={mag_env:.2e}>{self.env_min_grad:.0e}?{cond3}, "
                      f"var_e={var_env:.2e}<{self.env_var_threshold:.0e}?{cond4}, "
                      f"dom={env_dominance:.2e}")

            # 核心条件：因果梯度稳定（低方差+低幅值）AND 环境梯度活跃（方差小+幅值够）
            if cond1 and cond2 and cond3 and cond4:
                score = var_causal + mag_causal
                candidate_items.append((score, name, param))

        max_freeze = int(max(0, self.max_freeze_ratio) * total_count)
        max_freeze = min(max_freeze, len(candidate_items))

        if debug:
            print(f"\n[RoAda Debug] Task {current_task_id}: {total_count} evaluable params, "
                  f"{len(candidate_items)} candidates, max_freeze={max_freeze}, "
                  f"skipped(no_history)={skipped_no_history}, skipped(no_data)={skipped_no_enough_data}")
            print(f"  [DEBUG] grad_history_causal keys: {len(self.grad_history_causal)}, "
                  f"current_task_ema_causal keys: {len(self.current_task_ema_causal)}")
            # 统计4个核心条件的通过情况（与主循环逻辑一致）
            cond_pass = 0
            stats_evaluated = 0
            for name, param in model.named_parameters():
                if name not in self.grad_history_causal or name not in self.grad_history_env:
                    continue
                causal_list = list(self.grad_history_causal.get(name, []))
                env_list = list(self.grad_history_env.get(name, []))
                if name in self.current_task_ema_causal:
                    causal_list.append(self.current_task_ema_causal[name].cpu().numpy())
                if name in self.current_task_ema_env:
                    env_list.append(self.current_task_ema_env[name].cpu().numpy())

                if len(causal_list) >= 1 and len(env_list) >= 1:
                    stats_evaluated += 1
                    if len(causal_list) >= 2 and len(env_list) >= 2:
                        # 2+数据点：检查4个条件
                        var_causal = float(np.var(np.stack(causal_list), axis=0).mean())
                        var_env = float(np.var(np.stack(env_list), axis=0).mean())
                        mag_causal = float(np.abs(causal_list[-1]).mean())
                        mag_env = float(np.abs(env_list[-1]).mean())
                        if (var_causal < self.threshold and mag_causal < self.causal_max_grad
                                and mag_env > self.env_min_grad and var_env < self.env_var_threshold):
                            cond_pass += 1
                    else:
                        # 1数据点：检查env_dominance > 2.0
                        mag_causal = float(np.abs(causal_list[-1]).mean())
                        mag_env = float(np.abs(env_list[-1]).mean())
                        if mag_env / (mag_causal + 1e-12) > 2.0:
                            cond_pass += 1
            print(f"  有效评估参数: {stats_evaluated}, passed(4conds)={cond_pass}, failed={stats_evaluated - cond_pass}")
            if self.logger:
                self.logger.info(
                    f"[RoAda Debug] Task {current_task_id}: skipped(no_history)={skipped_no_history}, "
                    f"skipped(no_data)={skipped_no_enough_data}, "
                    f"evaluated={stats_evaluated}, passed(4conds)={cond_pass}, "
                    f"failed={stats_evaluated - cond_pass}"
                )

        frozen_count = 0
        frozen_names = []
        if max_freeze > 0:
            candidate_items.sort(key=lambda x: x[0])
            for _, name, param in candidate_items[:max_freeze]:
                self.frozen_flags[name] = True
                param.requires_grad = False
                frozen_names.append(name)
                frozen_count += 1

        # 验证冻结是否生效
        frozen_param_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_param_count = sum(1 for _ in model.parameters())
        if self.logger:
            self.logger.info(f"[RoAda Debug] Model has {frozen_param_count}/{total_param_count} frozen params after freezing")

        # 统计冻结参数的分布
        frozen_by_layer = {}
        for name in frozen_names:
            layer_type = name.split('.')[0] if '.' in name else name
            frozen_by_layer[layer_type] = frozen_by_layer.get(layer_type, 0) + 1

        if self.logger:
            self.logger.info(
                f"[Causal-Invariant RoAda] Task {current_task_id}: "
                f"{frozen_count} / {total_count} evaluable parameters frozen "
                f"(candidates={len(candidate_items)}, max_ratio={self.max_freeze_ratio:.2f}, "
                f"var_thresh={self.threshold:.0e}, env_var_thresh={self.env_var_threshold:.0e}, "
                f"causal_max_grad={self.causal_max_grad:.2e}, env_min_grad={self.env_min_grad:.2e}, "
                f"causal_env_ratio={self.causal_env_ratio:.2f})."
            )
            self.logger.info(f"[RoAda Debug] Frozen params by layer: {frozen_by_layer}")
            self.logger.info(f"[RoAda Debug] Frozen param names: {frozen_names[:10]}{'...' if len(frozen_names) > 10 else ''}")
        return frozen_count
