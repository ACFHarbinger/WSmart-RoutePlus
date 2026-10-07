"""
Eval Config module.

Attributes:
    datasets: Filename of the dataset(s) to evaluate.
    overwrite: Set true to overwrite.
    output_filename: Name of the results file to write.
    val_size: Number of instances used for reporting validation performance.
    offset: Offset where to start in dataset.
    eval_batch_size: Batch size to use during (baseline) evaluation.
    decoding: Decoding configuration.
    model: Model configuration.
    seed: Random seed to use.
    data_distribution: Data distribution of the dataset.
    no_cuda: Disable CUDA.
    compress_mask: Compress mask into long.
    max_calc_batch_size: Size for subbatches.
    results_dir: Name of evaluation results directory.
    multiprocessing: Use multiprocessing to parallelize over multiple GPUs.
    graph: Graph/instance configuration.
    reward: Objective/reward configuration.
    problem: Problem to evaluate.

Example:
    eval_config = EvalConfig(
        datasets=["test.pkl"],
        overwrite=True,
        output_filename="results.txt",
        val_size=12800,
        offset=0,
        eval_batch_size=256,
        decoding=DecodingConfig(),
        model=NeuralAgentConfig(),
        seed=42,
        data_distribution="gaussian",
        no_cuda=False,
        compress_mask=False,
        max_calc_batch_size=12800,
        results_dir="results",
        multiprocessing=False,
        graph=GraphConfig(),
        reward=ObjectiveConfig(),
        problem="vrpp",
    )
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.envs.env import EnvConfig
from logic.src.configs.models.decoding import DecodingConfig
from logic.src.configs.policies.na import NeuralAgentConfig


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        datasets: Filename of the dataset(s) to evaluate.
        overwrite: Set true to overwrite.
        output_filename: Name of the results file to write.
        val_size: Number of instances used for reporting validation performance.
        offset: Offset where to start in dataset.
        eval_batch_size: Batch size to use during (baseline) evaluation.
        decoding: Decoding configuration.
        model: Model configuration.
        seed: Random seed to use.
        data_distribution: Data distribution of the dataset.
        no_cuda: Disable CUDA.
        compress_mask: Compress mask into long.
        max_calc_batch_size: Size for subbatches.
        results_dir: Name of evaluation results directory.
        multiprocessing: Use multiprocessing to parallelize over multiple GPUs.
        graph: Graph/instance configuration.
        reward: Objective/reward configuration.
        problem: Problem to evaluate.
    """

    datasets: Optional[List[str]] = None
    overwrite: bool = False
    output_filename: Optional[str] = None
    val_size: int = 12_800
    offset: int = 0
    eval_batch_size: int = 256
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    policy: NeuralAgentConfig = field(default_factory=NeuralAgentConfig)
    seed: int = 42
    data_distribution: Optional[str] = None
    no_cuda: bool = False
    compress_mask: bool = False
    max_calc_batch_size: int = 12_800
    results_dir: str = "results"
    multiprocessing: bool = False
    env: Any = field(default_factory=EnvConfig)
    problem: str = "vrpp"
