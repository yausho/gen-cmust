import os
import time
import copy
import torch
from utils.args import create_parser
from utils.logging import get_logger
from utils.utils import set_seed, EarlyStopping
from utils.dataloader import get_dataloaders_scaler, DiskCacheDataset, ChunkShuffleSampler
from torch.utils.data import DataLoader, ConcatDataset

from model.dit import CMuST_DiT 
from model.mechanism import CausalDecipher, CausalMemoryController, CausalRoAdaController
from model.buffer import CausalColdStartAugmenter
from engine import train_flow_matching, test_flow_matching
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


def _compute_af_bwt_error(metric_matrix):
    """
    For error metrics (MAE/RMSE/MAPE): lower is better.
    AF  = average(final - diagonal) over old tasks (>=0 means forgetting).
    BWT = average(diagonal - final) over old tasks (higher is better).
    """
    num_tasks = len(metric_matrix)
    if num_tasks <= 1:
        return None, None

    last_col = num_tasks - 1
    deltas = []
    for i in range(num_tasks - 1):
        diag_i = metric_matrix[i][i]
        final_i = metric_matrix[i][last_col]
        if diag_i is None or final_i is None:
            continue
        deltas.append(final_i - diag_i)

    if not deltas:
        return None, None

    af = sum(deltas) / len(deltas)
    bwt = -af
    return af, bwt


def _compute_af_bwt_score(metric_matrix):
    """
    For score metrics (SSIM): higher is better.
    AF  = average(diagonal - final) over old tasks (>=0 means forgetting).
    BWT = average(final - diagonal) over old tasks (higher is better).
    """
    num_tasks = len(metric_matrix)
    if num_tasks <= 1:
        return None, None

    last_col = num_tasks - 1
    deltas = []
    for i in range(num_tasks - 1):
        diag_i = metric_matrix[i][i]
        final_i = metric_matrix[i][last_col]
        if diag_i is None or final_i is None:
            continue
        deltas.append(diag_i - final_i)

    if not deltas:
        return None, None

    af = sum(deltas) / len(deltas)
    bwt = -af
    return af, bwt

def main():
    parser = create_parser() #获取命令行参数
    args = parser.parse_args() #解析命令行参数并存储在 args 对象中

    # 如果 use_al_env_ratio 被设置了，就覆盖 roada_causal_env_ratio。这样用户只需要一个参数就能调整 AL 和 RoAda 的因果环境权重，避免了混淆和错误配置。
    if getattr(args, 'use_al_env_ratio', None) is not None:
        args.roada_causal_env_ratio = args.use_al_env_ratio
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    args.data_path = os.path.join("./data", args.dataset)
    args.log_dir = os.path.join("./logs", args.dataset)
    os.makedirs(args.log_dir, exist_ok=True)

    log_name = f"{'TEST' if args.test_only else 'TRAIN'}_{time.strftime('%m%d_%H%M%S')}.log"
    logger = get_logger(args.log_dir, name='GEN-CMuST', log_filename=log_name)
    set_seed(args.seed)

    logger.info(f"Arguments: {args}")#记录所有参数设置，方便复现和调试
    logger.info(f"Dataset: {args.dataset} | Device: {args.device}")

    # 自动检测数据目录下的任务文件夹，支持任意数量的任务顺序训练
    task_names = sorted([f for f in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, f))])
    num_tasks = len(task_names)
    logger.info(f"Detected Tasks: {task_names}")

    # Continual-learning metric matrix: row=i (evaluated task), col=j (after training task j)
    # Each entry stores final test metrics on task i when the current model is at stage j.
    mae_matrix = [[None for _ in range(num_tasks)] for _ in range(num_tasks)]

    # 模型实例化
    model = CMuST_DiT(
        input_size=args.input_size, #输入时间序列的长度（例如过去12小时）
        patch_size=args.patch_size,#每个 patch 的长度（例如4小时），模型会将输入切分成多个 patch 进行处理
        in_channels=args.in_channels,#输入数据的通道数（例如1表示单变量时间序列）
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        history_len=args.history_len,#历史输入长度（例如12小时），模型会根据这个长度来构建位置编码和注意力机制
        forecast_len=args.forecast_len,#预测输出长度（例如12小时），模型会根据这个长度来构建输出层和损失计算
    ).to(args.device)

    #将数据切割成因果区和环境区，并计算每个位置的因果重要性分数，供后续的增强和记忆机制使用
    causal_decipher = CausalDecipher(
        nucleus_energy_p=args.nucleus_energy_p,
        patch_size=args.patch_size,
        min_causal_ratio=args.min_causal_ratio,
        max_causal_ratio=args.max_causal_ratio,
    ).to(args.device)

    #负责存放旧知识的记忆库 在训练新任务时会把部分具有代表性的样本喂给模型 以防止灾难性遗忘
    #解决冷启动瓶颈，当模型接触到数据极少的新任务时，利用旧知识的逻辑，结合扩散模型，生成额外的训练样本来提升初始性能，缓解性能骤降问题
    #解决虚假相关性，只有旧知识和新知识保持一致的规律，才会被生成器利用来生成增强样本，避免引入与新任务无关的虚假相关性导致性能崩溃
    causal_memory = CausalMemoryController(
        memory_capacity=args.memory_capacity,#记忆库的最大容量，超过后会根据重要性分数丢弃旧样本
        causal_keep_ratio=args.causal_keep_ratio,#在记忆库满时，保留因果区样本的比例（例如0.2表示优先保留20%的因果区样本），剩余部分从环境区样本中丢弃
        patch_size=args.patch_size,#每个样本的 patch 大小，影响重要性分数的计算和样本的存储结构
        logger=logger 
    ) #只存储因果重要性评分高的样本，优先保留因果区样本，防止记忆库被环境区样本占满导致遗忘加剧


    #负责任务切换时冻结哪些参数，防止灾难性遗忘
    #如果一个参数在因果区域表现得非常稳定，但在环境区域剧烈波动，则认为是因果参数，冻结
    causal_roada = CausalRoAdaController(
        var_threshold=args.roada_var_threshold, #参数冻结的方差阈值，越小越激进
        min_grad=args.roada_min_grad, #冻结参数的最小梯度，防止过度冻结
        max_freeze_ratio=args.roada_max_freeze_ratio, #最大冻结比例，防止模型过度冻结导致性能崩溃
        logger=logger,  #日志记录器，用于输出冻结决策和统计信息
        causal_env_ratio=args.roada_causal_env_ratio #在计算冻结优先级时，因果区样本的权重相对于环境区的倍数 (e.g. 2.5 = 250% more priority for causal env samples)
    )
    
    #扩散模型的时间步长调度器，负责在训练和推理过程中根据预设的时间步长序列来调整噪声水平，支持不同的权重方案和动态调整
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=args.diffusion_steps)

    #冷启动启动增强器：在新任务上生成额外的训练样本以缓解初始性能下降
    coldstart_augmenter = CausalColdStartAugmenter(scheduler=scheduler, device=args.device)

    # AMP 梯度缩放器：在使用混合精度训练时，动态调整梯度的缩放因子以防止数值下溢和溢出，提升训练稳定性和性能
    amp_scaler = GradScaler() if args.device.type == 'cuda' else None
    train_scheduler_cache = copy.deepcopy(scheduler)
    train_scheduler_cache.set_timesteps(args.diffusion_steps) #保存好1000步的 scheduler 以供训练时使用，避免每次 forward 都重新设置时间步长导致的性能损失

    # Keep one optimizer across tasks to preserve Adam moments (exp_avg/exp_avg_sq).
    # Frozen params (requires_grad=False) are naturally skipped during optimizer.step().
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    final_rmse = None

    # 训练和评估循环：自动检测任务文件夹，依次训练每个任务，并在每个任务结束后评估之前的任务以检查遗忘情况
    for task_id, task_name in enumerate(task_names):
        args.current_task_id = task_id
        task_data_path = os.path.join(args.data_path, task_name)
        logger.info(f"\n" + "="*30 + f" Starting Task {task_id}: {task_name} " + "="*30)

        dataloaders, scaler, global_mask = get_dataloaders_scaler(
            task_data_path, args.batch_size, logger
        )
        base_train_dataset = dataloaders['train'].dataset
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']

        # 冷启动增强：只有在非第一个任务且启用了增强的情况下才执行，避免不必要的计算和潜在的性能风险
        if task_id > 0 and args.use_forward_causal_augment:
            logger.info(f"Task {task_id}: Executing causal cold-start augmentation (p={args.sampling_p}, r={args.gen_num_r})...")
            try:
                train_dataset = coldstart_augmenter.augment(
                    model=model, causal_decipher=causal_decipher,
                    current_task_data=base_train_dataset, args=args, inference_steps=args.inference_steps
                )
            except Exception as e:
                logger.warning(f"Augmentation failed, falling back to original data: {e}")
                train_dataset = base_train_dataset
        else:
            train_dataset = base_train_dataset

        # 如果训练数据是由多个磁盘缓存数据集组成的连接数据集，使用 ChunkShuffleSampler 来打乱样本顺序，
        # 避免全局 shuffle 导致的 LRU 抖动问题；否则正常使用 shuffle=True 的 DataLoader。
        if isinstance(train_dataset, ConcatDataset) and any(
                isinstance(d, DiskCacheDataset) for d in train_dataset.datasets):
            # Use chunk-aware sampler to prevent LRU thrashing from DataLoader's global shuffle.
            sampler = ChunkShuffleSampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      sampler=sampler, drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, drop_last=True)

        model_save_path = os.path.join(args.log_dir, f"best_model_task_{task_id}.pt")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=getattr(args, 'lr_min', 1e-6)
        )
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=model_save_path)

        # 任务切换时的主动增强：在训练新任务之前，利用当前模型和记忆库生成一些具有代表性的样本
        # 并将它们注入到记忆库中，以提升模型在新任务上的初始性能，缓解性能骤降问题
        if task_id > 0 and args.use_proactive_aug:
            logger.info(f"Task {task_id}: Executing proactive memory augmentation (capacity-based injection)...")
            try:
                scheduler_proactive = copy.deepcopy(scheduler)
                scheduler_proactive.set_timesteps(args.inference_steps)
                
                added_count = 0
                memory_capacity = getattr(args, 'memory_capacity', 200)
                inject_target = memory_capacity // 2
                
                for sample_x, sample_y in train_loader:
                    sample_x, sample_y = sample_x.to(args.device), sample_y.to(args.device)
                    
                    y_pseudo = causal_memory.proactive_augmentation(
                        model=model, scheduler=scheduler_proactive, causal_decipher=causal_decipher,
                        x_batch=sample_x, y_batch=sample_y, inference_steps=args.inference_steps
                    )
                    
                    # Use real model importance scores instead of dummy ones.
                    # Dummy all-ones scores cause CausalDecipher to produce an all-zero
                    # causal mask (sigmoid(0)=0.5 → hard_mask=0), making replay loss
                    # always zero AND giving artificially high heap priority.
                    model.eval()
                    with torch.no_grad():
                        _, real_scores = model(
                            sample_y, torch.zeros(sample_y.shape[0], device=args.device), sample_x
                        )
                    model.train()
                    
                    causal_memory.update_memory(sample_x.cpu(), y_pseudo.cpu(), real_scores.cpu())
                    
                    del y_pseudo
                    added_count += sample_x.shape[0]
                    if added_count >= inject_target:
                        break
                    
                logger.info(">>> Proactive augmentation successful.")
            except Exception as e:
                logger.warning(f">>> Proactive augmentation failed: {e}")

        if not args.test_only:
            for epoch in range(args.epochs):
                model, avg_loss = train_flow_matching(
                    model, scheduler, causal_decipher, causal_memory,
                    train_loader, optimizer, lr_scheduler, args.device, args,
                    current_epoch=epoch, global_mask=global_mask, logger=logger,
                    amp_scaler=amp_scaler, scheduler_cache=train_scheduler_cache,
                    roada_controller=causal_roada if args.use_causal_roada else None
                )
                
                val_mae, val_rmse, _, _ = test_flow_matching(
                    model, scheduler, val_loader, scaler, args.device, args, global_mask=global_mask
                )
                
                logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")
                
                early_stopping(val_mae, model)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered.")
                    break

        logger.info(f"Loading best weights for Task {task_id} for final testing...")
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=args.device, weights_only=True))
        elif args.test_only:
            logger.warning(
                f"Checkpoint not found at {model_save_path}. "
                f"Running test with current in-memory weights for Task {task_id}."
            )
        else:
            raise FileNotFoundError(f"Expected checkpoint not found: {model_save_path}")
        
        test_mae, test_rmse, _, _ = test_flow_matching(
            model, scheduler, test_loader, scaler, args.device, args, global_mask=global_mask
        )
        logger.info(f"[Final Result] Task {task_id} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")
        mae_matrix[task_id][task_id] = test_mae
        final_rmse = test_rmse

        if args.use_causal_roada:
            try:
                causal_roada.commit_task_signature()
                causal_roada.apply_freeze(model, optimizer, current_task_id=task_id)
            except Exception as e:
                logger.warning(f"RoAda parameter freezing failed: {e}")

        if task_id > 0:
            logger.info(f"\n========== Evaluating Memory Retention (Tasks 0 ~ {task_id-1}) ==========")
            for prev_id in range(task_id):
                prev_task_path = os.path.join(args.data_path, task_names[prev_id])
                prev_dataloaders, prev_scaler, prev_mask = get_dataloaders_scaler(prev_task_path, args.batch_size, logger)
                
                pm_mae, pm_rmse, _, _ = test_flow_matching(
                    model, scheduler, prev_dataloaders['test'], prev_scaler, args.device, args, global_mask=prev_mask
                )
                logger.info(f"   [Reviewing Task {prev_id}] | Post-Forgetting MAE: {pm_mae:.4f} | RMSE: {pm_rmse:.4f}")
                mae_matrix[prev_id][task_id] = pm_mae
            logger.info("="*70)

    # Continual-learning summary metrics after all tasks are completed.
    af_mae, bwt_mae = _compute_af_bwt_error(mae_matrix)

    if af_mae is not None:
        logger.info(
            f"[CL Summary] MAE / RMSE / AF / BWT | MAE: {mae_matrix[-1][-1]:.6f} | RMSE: {final_rmse:.6f} | AF: {af_mae:.6f} | BWT: {bwt_mae:.6f}"
        )

    logger.info("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()