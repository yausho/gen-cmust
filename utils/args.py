import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='GEN-CMuST')

    # 基础与数据参数
    parser.add_argument("--dataset", type=str, default="NYC")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_only", action="store_true", default=False)
    
    # 模型架构参数
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--forecast_len", type=int, default=12)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 扩散模型参数
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--weighting_scheme", type=str, default='logit_normal')
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--num_ensemble", type=int, default=15)
    parser.add_argument("--ensemble_oom_threshold", type=int, default=64,
                        help="Max B*num_ensemble before falling back to sequential ensemble to avoid OOM")

    # 因果与持续学习参数
    parser.add_argument("--lambda_irm", type=float, default=0.1,
                        help="IRM penalty coefficient (applied after warmup with auxiliary ramp)")
    parser.add_argument("--lambda_inv", type=float, default=0.15)
    parser.add_argument("--nucleus_energy_p", type=float, default=0.2,
                        help="Cumulative energy threshold for nucleus causal masking (replaces fixed env_ratio)")
    parser.add_argument("--min_causal_ratio", type=float, default=0.05,
                        help="Floor for adaptive causal patch ratio to prevent degenerate all-env masks")
    parser.add_argument("--max_causal_ratio", type=float, default=0.5,
                        help="Ceiling for adaptive causal patch ratio to prevent degenerate all-causal masks")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--memory_capacity", type=int, default=2000)
    parser.add_argument("--replay_ratio", type=float, default=0.5,
                        help="Replay weight multiplier; effective replay coeff = lambda_replay * replay_ratio * replay_ramp")
    parser.add_argument("--replay_warmup_epochs", type=int, default=0,
                        help="Replay starts after this epoch index; 0 enables replay from task start")
    parser.add_argument("--replay_full_weight", type=float, default=1.0,
                        help="Full-region replay weight before lambda_replay scaling")
    parser.add_argument("--replay_causal_weight", type=float, default=2.0,
                        help="Causal-region replay weight before lambda_replay scaling")
    parser.add_argument("--replay_env_weight", type=float, default=0.2,
                        help="Environment-region replay weight before lambda_replay scaling")
    parser.add_argument("--memory_policy", type=str, default="hybrid",
                        choices=["causal", "reservoir", "hybrid"],
                        help="Memory priority policy: causal score, random reservoir-like, or mixed")
    parser.add_argument("--memory_per_task_capacity", type=int, default=0,
                        help="If >0, fixed memory capacity per task; otherwise memory_capacity is balanced across tasks")
    parser.add_argument("--task_balanced_replay", action="store_true", default=True,
                        help="Sample replay data evenly from each previous task")
    parser.add_argument("--no_task_balanced_replay", dest="task_balanced_replay", action="store_false")
    parser.add_argument("--replay_age_alpha", type=float, default=0.0,
                        help="If >0, oversample older tasks in task-balanced replay; weight=(current_task-task_id)^alpha")
    parser.add_argument("--use_task_balanced_replay_loss", action="store_true", default=True,
                        help="Average replay loss per previous task before averaging across tasks")
    parser.add_argument("--no_task_balanced_replay_loss", dest="use_task_balanced_replay_loss", action="store_false")
    parser.add_argument("--replay_loss_age_alpha", type=float, default=0.0,
                        help="If >0, age-weight per-task replay losses; weight=(current_task-task_id)^alpha")
    parser.add_argument("--replay_loss_focal_alpha", type=float, default=0.0,
                        help="If >0, upweight previous tasks whose detached replay loss is currently larger")
    parser.add_argument("--memory_random_ratio", type=float, default=0.5,
                        help="Random-diversity share in hybrid memory priority; 0 keeps purely causal priority")
    parser.add_argument("--causal_keep_ratio", type=float, default=0.5,
                        help="Ratio of causal region samples to keep in memory buffer (higher = more diverse replay)")
    parser.add_argument("--sampling_p", type=float, default=0.5)
    parser.add_argument("--gen_num_r", type=int, default=5)
    parser.add_argument("--augment_ratio", type=float, default=0.2)
    parser.add_argument("--augment_chunk_size", type=int, default=512,
                        help="Samples per cold-start chunk file; larger = fewer files, slightly more RAM per flush")
    parser.add_argument("--env_redraw_scale", type=float, default=0.1)
    parser.add_argument("--lambda_do", type=float, default=0.02)
    parser.add_argument("--lambda_replay", type=float, default=2.0,
                        help="Coefficient for memory replay loss (higher = more replay weight)")
    parser.add_argument("--lambda_augment", type=float, default=0.1)
    parser.add_argument("--aux_ramp_epochs", type=int, default=30)
    parser.add_argument("--max_aux_to_base", type=float, default=0.5)
    
    # RoAda 参数
    parser.add_argument("--roada_var_threshold", type=float, default=0.01,
                        help="Variance threshold for causal gradient freeze")
    parser.add_argument("--roada_env_var_threshold", type=float, default=0.01,
                        help="Variance threshold for environment gradient freeze (usually larger than causal)")
    parser.add_argument("--roada_min_grad", type=float, default=1e-07)
    parser.add_argument("--roada_causal_max_grad", type=float, default=0.20,
                        help="Upper bound for causal gradient magnitude in freeze candidacy")
    parser.add_argument("--roada_env_min_grad", type=float, default=1e-05,
                        help="Lower bound for environment gradient magnitude in freeze candidacy; None -> roada_min_grad")
    parser.add_argument("--roada_max_freeze_ratio", type=float, default=0.15)
    parser.add_argument("--roada_causal_env_ratio", type=float, default=2.5)
    parser.add_argument("--roada_protection_mode", type=str, default="hard",
                        choices=["hard", "soft", "regularize"],
                        help="CausalRoAda protection mode: hard freezes tensors, soft scales gradients, regularize anchors selected params")
    parser.add_argument("--roada_selection_mode", type=str, default="top_tensor",
                        choices=["percentile", "threshold", "top_tensor", "bottom_tensor", "random_top_tensor"],
                        help="Soft CausalRoAda selection mode")
    parser.add_argument("--roada_protect_ratio", type=float, default=0.1,
                        help="Max protected element ratio per parameter tensor for soft CausalRoAda")
    parser.add_argument("--roada_tensor_protect_ratio", type=float, default=0.2,
                        help="Max tensor ratio considered for top_tensor soft CausalRoAda")
    parser.add_argument("--roada_soft_scale", type=float, default=0.3,
                        help="Gradient multiplier for soft-protected elements")
    parser.add_argument("--roada_reg_lambda", type=float, default=10.0,
                        help="Penalty weight for regularize-mode CausalRoAda anchor drift")
    parser.add_argument("--roada_update_interval", type=int, default=5,
                        help="Call update_dual_signature every N batches to avoid 3x backward overhead")
    parser.add_argument("--roada_when", type=str, default="replay_or_augment",
                        choices=["always", "augment_only", "replay_or_augment", "never"],
                        help="When to activate RoAda protection")
    parser.add_argument("--use_replay_aware_roada", action="store_true", default=True,
                        help="Use replay-loss gradients to build RoAda protection masks")
    parser.add_argument("--no_replay_aware_roada", dest="use_replay_aware_roada", action="store_false")
    parser.add_argument("--roada_replay_protect_ratio", type=float, default=0.03,
                        help="Per-tensor top ratio protected by replay-aware RoAda")
    parser.add_argument("--roada_replay_soft_scale", type=float, default=0.5,
                        help="Gradient multiplier for replay-aware RoAda masks")
    parser.add_argument("--roada_replay_update_interval", type=int, default=5,
                        help="Update replay-aware RoAda masks every N batches")
    parser.add_argument("--roada_replay_momentum", type=float, default=0.95,
                        help="EMA momentum for replay-aware RoAda gradient importance")
    parser.add_argument("--roada_replay_conflict_weight", type=float, default=1.0,
                        help="Extra weight for parameters where replay and current gradients conflict")
    parser.add_argument("--use_roada_conflict_surgery", action="store_true", default=False,
                        help="Use replay-gradient signs to damp only current gradients that conflict with replay")
    parser.add_argument("--no_roada_conflict_surgery", dest="use_roada_conflict_surgery", action="store_false")
    parser.add_argument("--roada_conflict_soft_scale", type=float, default=0.0,
                        help="Gradient multiplier for replay-important elements whose update conflicts with replay gradients")
    parser.add_argument("--use_al_env_ratio", type=float, default=None)
    parser.add_argument("--proactive_inject_ratio", type=float, default=0.25,
                        help="Fraction of memory_capacity to fill with proactive augmentation")

    # EWC 参数
    parser.add_argument("--use_ewc", action="store_true", default=True,
                        help="Enable EWC regularization to protect important parameters")
    parser.add_argument("--no_ewc", dest="use_ewc", action="store_false")
    parser.add_argument("--ewc_lambda", type=float, default=5000,
                        help="EWC penalty coefficient (higher = stronger protection of old tasks)")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum LR for cosine annealing (eta_min)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--use_forward_causal_augment", action="store_true", default=False,
                        help="Enable causal cold-start augmentation before training each non-initial task")
    parser.add_argument("--no_forward_causal_augment", dest="use_forward_causal_augment", action="store_false")
    
    parser.add_argument("--use_proactive_aug", action="store_true", default=False)
    parser.add_argument("--no_proactive_aug", dest="use_proactive_aug", action="store_false")
    
    parser.add_argument("--use_causal_roada", action="store_true", default=True)
    parser.add_argument("--no_causal_roada", dest="use_causal_roada", action="store_false")

    # 任务适配器参数（类似DOL的LSA）
    parser.add_argument("--use_task_adapter", action="store_true", default=False,
                        help="Enable task-aware low-rank adapter (like DOL's LSA) to reduce forgetting")
    parser.add_argument("--no_task_adapter", dest="use_task_adapter", action="store_false")
    parser.add_argument("--use_task_head", action="store_true", default=False,
                        help="Enable task-specific output heads for stronger task isolation")
    parser.add_argument("--no_task_head", dest="use_task_head", action="store_false")
    parser.add_argument("--isolate_task_modules", action="store_true", default=False,
                        help="Freeze old task adapters/heads and train only current task modules plus optional backbone")
    parser.add_argument("--no_isolate_task_modules", dest="isolate_task_modules", action="store_false")
    parser.add_argument("--freeze_backbone_after_task0", action="store_true", default=False,
                        help="Under task isolation, freeze shared backbone after the first task")
    parser.add_argument("--no_freeze_backbone_after_task0", dest="freeze_backbone_after_task0", action="store_false")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.02,
                        help="Backbone LR multiplier for task_id > 0 under task isolation")
    parser.add_argument("--lsa_dim", type=int, default=4,
                        help="Low-rank dimension for task adapter")
    parser.add_argument("--lsa_num", type=int, default=2,
                        help="Number of low-rank layers in task adapter")
    parser.add_argument("--max_tasks", type=int, default=20,
                        help="Maximum number of tasks (for adapter memory allocation)")

    return parser
