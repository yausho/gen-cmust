import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='GEN-CMuST')

    # 基础与数据参数
    parser.add_argument("--dataset", type=str, default="NYC")
    parser.add_argument("--gpu", type=int, default=0)
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
    parser.add_argument("--num_ensemble", type=int, default=20)
    parser.add_argument("--ensemble_oom_threshold", type=int, default=64,
                        help="Max B*num_ensemble before falling back to sequential ensemble to avoid OOM")

    # 因果与持续学习参数
    parser.add_argument("--lambda_irm", type=float, default=0.1,
                        help="IRM penalty coefficient (applied after warmup with auxiliary ramp)")
    parser.add_argument("--lambda_inv", type=float, default=0.15)
    parser.add_argument("--nucleus_energy_p", type=float, default=0.25,
                        help="Cumulative energy threshold for nucleus causal masking (replaces fixed env_ratio)")
    parser.add_argument("--min_causal_ratio", type=float, default=0.05,
                        help="Floor for adaptive causal patch ratio to prevent degenerate all-env masks")
    parser.add_argument("--max_causal_ratio", type=float, default=0.5,
                        help="Ceiling for adaptive causal patch ratio to prevent degenerate all-causal masks")
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--memory_capacity", type=int, default=2000)
    parser.add_argument("--replay_ratio", type=float, default=0.5,
                        help="Replay weight multiplier; effective replay coeff = lambda_replay * replay_ratio * replay_ramp")
    parser.add_argument("--replay_warmup_epochs", type=int, default=0,
                        help="Replay starts after this epoch index; 0 enables replay from task start")
    parser.add_argument("--causal_keep_ratio", type=float, default=0.2)
    parser.add_argument("--sampling_p", type=float, default=0.5)
    parser.add_argument("--gen_num_r", type=int, default=5)
    parser.add_argument("--augment_ratio", type=float, default=0.2)
    parser.add_argument("--augment_chunk_size", type=int, default=512,
                        help="Samples per cold-start chunk file; larger = fewer files, slightly more RAM per flush")
    parser.add_argument("--env_redraw_scale", type=float, default=0.1)
    parser.add_argument("--lambda_do", type=float, default=0.02)
    parser.add_argument("--lambda_replay", type=float, default=1.0,
                        help="Coefficient for memory replay loss")
    parser.add_argument("--lambda_augment", type=float, default=0.1)
    parser.add_argument("--aux_ramp_epochs", type=int, default=30)
    parser.add_argument("--max_aux_to_base", type=float, default=0.5)
    
    # RoAda 参数
    parser.add_argument("--roada_var_threshold", type=float, default=1e-04,
                        help="Variance threshold for freeze - relaxed for better freezing")
    parser.add_argument("--roada_min_grad", type=float, default=1e-07)
    parser.add_argument("--roada_causal_max_grad", type=float, default=0.05,
                        help="Upper bound for causal gradient magnitude in freeze candidacy")
    parser.add_argument("--roada_env_min_grad", type=float, default=1e-04,
                        help="Lower bound for environment gradient magnitude in freeze candidacy; None -> roada_min_grad")
    parser.add_argument("--roada_max_freeze_ratio", type=float, default=0.35)
    parser.add_argument("--roada_causal_env_ratio", type=float, default=2.5)
    parser.add_argument("--roada_update_interval", type=int, default=5,
                        help="Call update_dual_signature every N batches to avoid 3x backward overhead")
    parser.add_argument("--use_al_env_ratio", type=float, default=None)
    parser.add_argument("--proactive_inject_ratio", type=float, default=0.25,
                        help="Fraction of memory_capacity to fill with proactive augmentation")

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
    
    parser.add_argument("--use_proactive_aug", action="store_true", default=True)
    parser.add_argument("--no_proactive_aug", dest="use_proactive_aug", action="store_false")
    
    parser.add_argument("--use_causal_roada", action="store_true", default=True)
    parser.add_argument("--no_causal_roada", dest="use_causal_roada", action="store_false")

    return parser