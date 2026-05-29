import torch
import torch.nn as nn


class TaskAwareAdapter(nn.Module):
    """
    低秩任务感知适配器，类似DOL的LSA机制。

    核心思想：
    1. 基础参数对所有任务共享，捕获通用的时空模式
    2. 任务适配器负责捕获任务特定的因果模式
    3. 持续学习时冻结基础参数，只更新适配器，减少灾难性遗忘

    设计参考：
    - DOL的LSA: 低秩分解的任务嵌入
    - LoRA: 低秩适应的高效参数更新
    - AdapterFormer: Transformer的轻量级任务适配
    """

    def __init__(self, hidden_size, num_tasks=10, lsa_dim=4, lsa_num=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lsa_dim = lsa_dim
        self.lsa_num = lsa_num
        self.num_tasks = num_tasks
        self.dropout = dropout

        # 任务嵌入：每个任务一个低秩适配器
        # 使用多个低秩投影的组合，捕获不同方面的任务特征
        self.task_adapters = nn.ModuleList([
            self._create_adapter_for_task() for _ in range(num_tasks)
        ])

        # 全局任务嵌入，用于初始化
        self.register_buffer('zero_embed', torch.zeros(1, hidden_size))

    def _create_adapter_for_task(self):
        """为单个任务创建低秩适配器"""
        layers = []
        in_dim = self.hidden_size

        for i in range(self.lsa_num):
            out_dim = self.lsa_dim if i < self.lsa_num - 1 else self.hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < self.lsa_num - 1:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim

        return nn.Sequential(*layers)

    def forward(self, x, task_id):
        """
        Args:
            x: (B, N, hidden_size) 输入特征
            task_id: int, 当前任务ID

        Returns:
            适配后的特征 (B, N, hidden_size)
        """
        if task_id is None or task_id >= self.num_tasks:
            return x

        # 残差连接：adapter(x) + x
        return x + self.task_adapters[task_id](x)


class CausalTaskAdapter(nn.Module):
    """
    结合因果分离思想的增强版任务适配器。

    在适配器内部区分：
    1. 因果适配：捕获任务间共享的因果规律
    2. 环境适配：捕获任务特定的环境模式
    """

    def __init__(self, hidden_size, num_tasks=10, causal_dim=2, env_dim=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 因果适配器（更保守，捕获通用模式）
        self.causal_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, causal_dim),
                nn.SiLU(),
                nn.Linear(causal_dim, hidden_size)
            ) for _ in range(num_tasks)
        ])

        # 环境适配器（更灵活，捕获特定模式）
        self.env_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, env_dim),
                nn.SiLU(),
                nn.Linear(env_dim, hidden_size)
            ) for _ in range(num_tasks)
        ])

        # 可学习的因果/环境混合权重
        self.causal_ratio = 0.5

    def forward(self, x, task_id, causal_weight=1.0):
        """
        Args:
            x: (B, N, hidden_size) 输入特征
            task_id: int, 当前任务ID
            causal_weight: float, 因果适配的权重 (0-1)

        Returns:
            适配后的特征 (B, N, hidden_size)
        """
        if task_id is None:
            return x

        causal_out = self.causal_adapters[task_id](x)
        env_out = self.env_adapters[task_id](x)

        # 加权组合：causal_weight控制因果适配的比例
        adapter_out = causal_weight * causal_out + (1 - causal_weight) * env_out

        return x + adapter_out


class MultiHeadTaskAdapter(nn.Module):
    """
    多头任务适配器，将隐藏维度分成多个头分别适配。

    优势：
    1. 不同头可以学习不同方面的任务特征
    2. 头之间解耦，减少干扰
    3. 保持原有的表达能力
    """

    def __init__(self, hidden_size, num_tasks=10, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 每个任务、每个头一个适配器
        self.task_head_adapters = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.head_dim, self.head_dim // 2),
                    nn.SiLU(),
                    nn.Linear(self.head_dim // 2, self.head_dim)
                ) for _ in range(num_heads)
            ]) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id):
        """
        Args:
            x: (B, N, hidden_size) 输入特征
            task_id: int, 当前任务ID

        Returns:
            适配后的特征 (B, N, hidden_size)
        """
        if task_id is None:
            return x

        B, N, _ = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)

        adapted = []
        for h in range(self.num_heads):
            head_input = x[:, h]  # (B, N, head_dim)
            head_output = self.task_head_adapters[task_id][h](head_input)
            adapted.append(head_input + head_output)  # 残差连接

        adapted = torch.stack(adapted, dim=1)  # (B, H, N, head_dim)
        adapted = adapted.transpose(1, 2).reshape(B, N, self.hidden_size)  # (B, N, hidden_size)

        return adapted


def create_task_adapter(adapter_type='lora', **kwargs):
    """
    工厂函数，创建指定类型的任务适配器

    Args:
        adapter_type: 'lora', 'causal', 'multihead'
        **kwargs: 传递给适配器的参数

    Returns:
        TaskAdapter实例
    """
    if adapter_type == 'lora':
        return TaskAwareAdapter(**kwargs)
    elif adapter_type == 'causal':
        return CausalTaskAdapter(**kwargs)
    elif adapter_type == 'multihead':
        return MultiHeadTaskAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
