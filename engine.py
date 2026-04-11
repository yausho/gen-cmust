import torch
import numpy as np
import copy
from tqdm import tqdm
from utils.metrics import masked_mae, masked_rmse, masked_mape, masked_ssim
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
try:
    from torch.amp import autocast as _autocast, GradScaler

    def amp_autocast(enabled=True, dtype=torch.float16):
        return _autocast(device_type='cuda', enabled=enabled, dtype=dtype)
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler

    def amp_autocast(enabled=True, dtype=torch.float16):
        return _autocast(enabled=enabled, dtype=dtype)

def train_flow_matching(model, scheduler, causal_decipher, causal_memory,
                        dataloader, optimizer, lr_scheduler, device, args,
                        current_epoch=0, 
                        global_mask=None, ema_model=None, teacher_model=None, logger=None,
                        amp_scaler=None, scheduler_cache=None, roada_controller=None):
    model.train()
    total_loss = []
    
    start_temp, end_temp = 5.0, 15.0
    annealed_temp = start_temp + (end_temp - start_temp) * (current_epoch / args.epochs)
    if hasattr(causal_decipher, 'temperature'):
        causal_decipher.temperature = annealed_temp

    if scheduler_cache is not None:
        scheduler_copy = scheduler_cache
    else:
        scheduler_copy = copy.deepcopy(scheduler)
        scheduler_copy.set_timesteps(args.diffusion_steps)

    use_amp = (amp_scaler is not None and device.type == 'cuda')

    iter_bar = tqdm(dataloader, desc=f"Epoch {current_epoch}", leave=False)
    for i, (x, y) in enumerate(iter_bar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        u = compute_density_for_timestep_sampling(
            args.weighting_scheme, y.shape[0], args.logit_mean, args.logit_std, args.mode_scale
        )
        
        # 极速 GPU 索引提取 sigmas 和 timesteps
        indices = (u * scheduler_copy.config.num_train_timesteps).long().to(device)
        timesteps = scheduler_copy.timesteps.to(device)[indices]
        sigmas = scheduler_copy.sigmas.to(device=device, dtype=y.dtype)[indices]
        while len(sigmas.shape) < y.ndim:
            sigmas = sigmas.unsqueeze(-1)
            
        noise = torch.randn_like(y)
        noisy_target = (1.0 - sigmas) * y + sigmas * noise

        with amp_autocast(enabled=use_amp, dtype=torch.float16):
            pred_velocity, _ = model(noisy_target, timesteps, x)
            target_velocity = noise - y

        if current_epoch < args.warmup_epochs:
            causal_mask = torch.ones_like(y).to(device)
            clean_importance_scores = None
        else:
            model.eval()
            _, clean_importance_scores = model(y, torch.zeros(y.shape[0], device=device), x)
            model.train()
            causal_mask = causal_decipher(y, clean_importance_scores)

        # Fix: exclude padding zeros (from auto_pad_to_32) from env loss computation.
        if global_mask is not None:
            valid = global_mask.to(device)
            causal_mask = causal_mask * valid
            env_mask = valid - causal_mask
            padding_mask = 1.0 - valid  # regions outside real spatial extent
        else:
            env_mask = 1 - causal_mask
            padding_mask = torch.zeros_like(causal_mask)  # no padding when no global_mask

        weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas).float()
        while len(weighting.shape) < causal_mask.ndim: 
            weighting = weighting.unsqueeze(-1)

        loss_matrix = (pred_velocity - target_velocity) ** 2
        
        causal_sum = torch.clamp(causal_mask.reshape(y.shape[0], -1).sum(dim=1), min=1.0)
        env_sum = torch.clamp(env_mask.reshape(y.shape[0], -1).sum(dim=1), min=1.0)

        sample_causal_risk_current = (weighting * loss_matrix * causal_mask).reshape(y.shape[0], -1).sum(dim=1) / causal_sum
        sample_env_risk_current = (weighting * loss_matrix * env_mask).reshape(y.shape[0], -1).sum(dim=1) / env_sum
        
        loss_causal_current = torch.mean(sample_causal_risk_current)
        loss_env_current = torch.mean(sample_env_risk_current)

        # IRM penalty (scale-trick): enforce invariance by minimizing
        # gradient magnitudes of environment-specific risks w.r.t. a shared scale.
        irm_penalty = (pred_velocity * 0.0).sum().float()
        if current_epoch >= args.warmup_epochs and args.lambda_irm > 0:
            irm_scale = torch.tensor(
                1.0, device=device, dtype=pred_velocity.dtype, requires_grad=True
            )
            irm_residual = irm_scale * pred_velocity - target_velocity

            sample_causal_risk_scaled = (
                weighting * (irm_residual ** 2) * causal_mask
            ).reshape(y.shape[0], -1).sum(dim=1) / causal_sum
            sample_env_risk_scaled = (
                weighting * (irm_residual ** 2) * env_mask
            ).reshape(y.shape[0], -1).sum(dim=1) / env_sum

            loss_causal_scaled = torch.mean(sample_causal_risk_scaled)
            loss_env_scaled = torch.mean(sample_env_risk_scaled)
            grad_causal = torch.autograd.grad(
                loss_causal_scaled, [irm_scale], create_graph=True, retain_graph=True
            )[0]
            grad_env = torch.autograd.grad(
                loss_env_scaled, [irm_scale], create_graph=True, retain_graph=True
            )[0]
            irm_penalty = 0.5 * (grad_causal.pow(2) + grad_env.pow(2))

        # do(env) consistency: perturb only env regions, enforce causal prediction invariance.
        # DDP-safe zero: tied to pred_velocity so all parameters are "used" in the graph.
        loss_do = (pred_velocity * 0.0).sum().float()
        if current_epoch >= args.warmup_epochs and args.lambda_do > 0:
            perm_idx = torch.randperm(y.shape[0], device=device)
            env_source = noisy_target[perm_idx]
            # Restore padding-area noise to prevent distribution shift in DiT global attention.
            noisy_do = noisy_target * causal_mask + env_source * env_mask + noisy_target * padding_mask
            with amp_autocast(enabled=use_amp, dtype=torch.float16):
                pred_velocity_do, _ = model(noisy_do, timesteps, x)

            sample_do = (
                weighting * (pred_velocity_do - pred_velocity.detach()) ** 2 * causal_mask
            ).reshape(y.shape[0], -1).sum(dim=1) / causal_sum
            loss_do = torch.mean(sample_do)

        replay_loss = (pred_velocity * 0.0).sum().float()  # DDP-safe zero placeholder
        # 记忆回放与防强迫性遗忘逻辑
        if causal_memory is not None and len(causal_memory.memory_heap) > 0 and args.replay_ratio > 0:
            rep_y, rep_x, rep_scores = causal_memory.get_replay_data(batch_size=x.shape[0])
            if rep_y is not None:
                rep_y, rep_x, rep_scores = rep_y.to(device), rep_x.to(device), rep_scores.to(device)
                u_rep = compute_density_for_timestep_sampling(
                    args.weighting_scheme, rep_y.shape[0], args.logit_mean, args.logit_std, args.mode_scale
                )
                indices_rep = (u_rep * scheduler_copy.config.num_train_timesteps).long().to(device)
                timesteps_rep = scheduler_copy.timesteps.to(device)[indices_rep]
                sigmas_rep = scheduler_copy.sigmas.to(device=device, dtype=rep_y.dtype)[indices_rep]
                while len(sigmas_rep.shape) < rep_y.ndim:
                    sigmas_rep = sigmas_rep.unsqueeze(-1)
                    
                noise_rep = torch.randn_like(rep_y)
                noisy_target_rep = (1.0 - sigmas_rep) * rep_y + sigmas_rep * noise_rep

                if current_epoch < args.warmup_epochs:
                    causal_mask_rep = torch.ones_like(rep_y).to(device)
                else:
                    causal_mask_rep = causal_decipher(rep_y, rep_scores)
                    # NOTE: do NOT apply current-task valid mask here — rep_y may have a
                    # different spatial resolution than the current task, so `valid` would
                    # cause a shape mismatch. Replay samples are already de-padded at store time.

                with amp_autocast(enabled=use_amp, dtype=torch.float16):
                    pred_velocity_rep, _ = model(noisy_target_rep, timesteps_rep, rep_x)
                
                loss_matrix_rep = (pred_velocity_rep - (noise_rep - rep_y)) ** 2
                weighting_rep = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas_rep).float()
                while len(weighting_rep.shape) < causal_mask_rep.ndim: 
                    weighting_rep = weighting_rep.unsqueeze(-1)

                causal_sum_rep = causal_mask_rep.reshape(rep_y.shape[0], -1).sum(dim=1)
                causal_sum_rep = torch.clamp(causal_sum_rep, min=1.0)
                
                sample_causal_risk_rep = (weighting_rep * loss_matrix_rep * causal_mask_rep).reshape(rep_y.shape[0], -1).sum(dim=1) / causal_sum_rep
                replay_loss = sample_causal_risk_rep.mean()

        # Forward augment stream: keep causal region, redraw env region for sparse-task adaptation.
        augment_loss = (pred_velocity * 0.0).sum().float()  # DDP-safe zero placeholder
        if current_epoch >= args.warmup_epochs and args.augment_ratio > 0:
            perm_idx_aug = torch.randperm(y.shape[0], device=device)
            env_redraw = y[perm_idx_aug] + args.env_redraw_scale * torch.randn_like(y)
            env_mix = (1.0 - args.augment_ratio) * y + args.augment_ratio * env_redraw
            # Restore padding-area values to keep input distribution consistent.
            y_aug = y * causal_mask + env_mix * env_mask + y * padding_mask

            u_aug = compute_density_for_timestep_sampling(
                args.weighting_scheme, y_aug.shape[0], args.logit_mean, args.logit_std, args.mode_scale
            )
            indices_aug = (u_aug * scheduler_copy.config.num_train_timesteps).long().to(device)
            timesteps_aug = scheduler_copy.timesteps.to(device)[indices_aug]
            sigmas_aug = scheduler_copy.sigmas.to(device=device, dtype=y_aug.dtype)[indices_aug]
            while len(sigmas_aug.shape) < y_aug.ndim:
                sigmas_aug = sigmas_aug.unsqueeze(-1)

            noise_aug = torch.randn_like(y_aug)
            noisy_target_aug = (1.0 - sigmas_aug) * y_aug + sigmas_aug * noise_aug
            with amp_autocast(enabled=use_amp, dtype=torch.float16):
                pred_velocity_aug, _ = model(noisy_target_aug, timesteps_aug, x)

            target_velocity_aug = noise_aug - y_aug
            weighting_aug = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas_aug).float()
            while len(weighting_aug.shape) < causal_mask.ndim:
                weighting_aug = weighting_aug.unsqueeze(-1)

            aug_sum = torch.clamp(env_mask.reshape(y.shape[0], -1).sum(dim=1), min=1.0)
            sample_aug = (
                weighting_aug * (pred_velocity_aug - target_velocity_aug) ** 2 * env_mask
            ).reshape(y.shape[0], -1).sum(dim=1) / aug_sum
            augment_loss = torch.mean(sample_aug)

        base_loss = loss_causal_current + args.lambda_inv * loss_env_current
        if current_epoch < args.warmup_epochs:
            aux_ramp = 0.0
        else:
            ramp_span = max(1, args.aux_ramp_epochs)
            aux_ramp = min(1.0, float(current_epoch - args.warmup_epochs + 1) / float(ramp_span))

        base_ref = base_loss.detach().abs() + 1e-6
        max_aux = args.max_aux_to_base * base_ref
        irm_penalty_capped = torch.minimum(irm_penalty, max_aux)
        replay_loss_capped = torch.minimum(replay_loss, max_aux)
        augment_loss_capped = torch.minimum(augment_loss, max_aux)
        loss_do_capped = torch.minimum(loss_do, max_aux)

        replay_ratio_eff = max(0.0, float(args.replay_ratio))
        lambda_irm_eff = args.lambda_irm * aux_ramp
        lambda_replay_eff = args.lambda_replay * replay_ratio_eff * aux_ramp
        lambda_augment_eff = args.lambda_augment * aux_ramp
        lambda_do_eff = args.lambda_do * aux_ramp

        loss_task = (
            base_loss
            + lambda_irm_eff * irm_penalty_capped
            + lambda_replay_eff * replay_loss_capped
            + lambda_augment_eff * augment_loss_capped
            + lambda_do_eff * loss_do_capped
        )

        if current_epoch >= args.warmup_epochs and roada_controller is not None:
            # Throttle: run dual-signature backward only every N batches.
            # EMA smoothing means infrequent sampling still tracks the gradient trend accurately,
            # while reducing backward passes from 3x/batch to ~1x + 2x/N on average.
            roada_interval = getattr(args, 'roada_update_interval', 10)
            if i % roada_interval == 0:
                scale_factor = amp_scaler.get_scale() if use_amp else 1.0
                roada_controller.update_dual_signature(
                    model,
                    loss_causal_current * scale_factor,
                    loss_env_current * scale_factor,
                    inv_scale=1.0 / max(scale_factor, 1.0),
                )

        # 单一反向传播释放算力黑洞
        if use_amp:
            amp_scaler.scale(loss_task).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss_task.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        total_loss.append(loss_task.item())
        iter_bar.set_postfix(
            loss=f"{loss_task.item():.4f}",
            base=f"{base_loss.item():.4f}",
            irm=f"{irm_penalty.item():.4f}",
            rep=f"{replay_loss.item():.4f}",
            aug=f"{augment_loss.item():.4f}",
            do=f"{loss_do.item():.4f}",
            ramp=f"{aux_ramp:.2f}",
        )

        if causal_memory and clean_importance_scores is not None:
            causal_memory.update_memory(x, y, clean_importance_scores.detach(), causal_mask.detach())

    if lr_scheduler:
        lr_scheduler.step()
    return model, np.mean(total_loss)


def test_flow_matching(model, scheduler, dataloader, scaler, device, args, global_mask=None, logger=None):
    model.eval()
    y_preds, y_trues = [], []
    num_ensemble = args.num_ensemble 
    
    scheduler_infer = copy.deepcopy(scheduler)
    scheduler_infer.set_timesteps(args.inference_steps)
    sigmas = scheduler_infer.sigmas.to(device)
    timesteps = scheduler_infer.timesteps.to(device)
    
    oom_threshold = getattr(args, 'ensemble_oom_threshold', 64)
    test_bar = tqdm(dataloader, desc="Testing", leave=False)
    with torch.no_grad():
        for x, y in test_bar:
            x, y = x.to(device), y.to(device)
            B = x.shape[0]

            # OOM guard: fall back to sequential ensemble when batch*ensemble is too large.
            if B * num_ensemble <= oom_threshold:
                repeat_dims = [num_ensemble] + [1] * (x.ndim - 1)
                x_expanded = x.repeat(*repeat_dims)
                latents = torch.randn(num_ensemble * B, *y.shape[1:], device=device, dtype=y.dtype)

                for i, t in enumerate(timesteps):
                    sigma_curr = sigmas[i]
                    sigma_next = sigmas[i + 1]
                    t_expanded = t.expand(latents.shape[0])
                    model_output, _ = model(latents, t_expanded, x_expanded)
                    latents = latents + (sigma_next - sigma_curr) * model_output

                latents = latents.view(num_ensemble, B, *y.shape[1:])
                ensemble_mean = torch.mean(latents, dim=0)
            else:
                # Sequential per-ensemble sampling to avoid OOM.
                preds_list = []
                for _ in range(num_ensemble):
                    latents = torch.randn_like(y)
                    for i, t in enumerate(timesteps):
                        sigma_curr = sigmas[i]
                        sigma_next = sigmas[i + 1]
                        t_expanded = t.expand(latents.shape[0])
                        model_output, _ = model(latents, t_expanded, x)
                        latents = latents + (sigma_next - sigma_curr) * model_output
                    preds_list.append(latents)
                ensemble_mean = torch.stack(preds_list, dim=0).mean(dim=0)

            y_preds.append(scaler.inverse_transform(ensemble_mean.cpu().numpy()))
            y_trues.append(scaler.inverse_transform(y.cpu().numpy()))
            
    y_preds = np.maximum(np.concatenate(y_preds, axis=0), 0.0) 
    y_trues = np.concatenate(y_trues, axis=0)
    y_trues[y_trues < 1e-4] = 0.0 

    # Compute SSIM on the full 2D grid BEFORE flattening by global_mask.
    ssim_val = masked_ssim(y_preds, y_trues, global_mask=global_mask)

    if global_mask is not None:
        mask_np = global_mask.squeeze().cpu().numpy().astype(bool)
        y_preds = y_preds[..., mask_np]
        y_trues = y_trues[..., mask_np]

    return masked_mae(torch.tensor(y_preds), torch.tensor(y_trues), null_val=0.0).item(), \
           masked_rmse(torch.tensor(y_preds), torch.tensor(y_trues), null_val=0.0).item(), \
           masked_mape(torch.tensor(y_preds), torch.tensor(y_trues), null_val=0.0).item(), \
           ssim_val