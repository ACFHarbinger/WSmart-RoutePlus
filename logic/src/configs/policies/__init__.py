"""
Policy configuration dataclasses.
Attributes:
    ABCConfig: ABC policy configuration.
    ABPCHGConfig: ABC + PC + HG policy configuration.
    HyperHeuristicACOConfig: Hyper-heuristic ACO policy configuration.
    KSparseACOConfig: K-sparse ACO policy configuration.
    ADPRolloutConfig: ADP rollout policy configuration.

    AdaptiveKernelSearchConfig: Adaptive kernel search policy configuration.
    ALNSConfig: ALNS policy configuration.
    ALNSIPOConfig: ALNS-IPO policy configuration.
    AMPHHConfig: AMPHH policy configuration.
    ARCOConfig: ARCO policy configuration.
    BBConfig: BB policy configuration.
    BCConfig: BC policy configuration.
    BPConfig: BP policy configuration.
    BPCConfig: BPC policy configuration.
    CALMConfig: CALM policy configuration.
    CFRSConfig: CFRS policy configuration.
    CGHConfig: CGH policy configuration.
    CPSATConfig: CP-SAT policy configuration.
    CVRPConfig: CVRP policy configuration.
    DEConfig: DE policy configuration.
    MuCommaLambdaESConfig: Mu, lambda ES policy configuration.
    MuKappaLambdaESConfig: Mu, kappa, lambda ES policy configuration.
    MuPlusLambdaESConfig: Mu + lambda ES policy configuration.
    ExactSDPConfig: Exact SDP policy configuration.
    FAConfig: FA policy configuration.
    FILOConfig: FILO policy configuration.
    GAConfig: GA policy configuration.
    GIHHConfig: GIHH policy configuration.
    GLSConfig: GLS policy configuration.
    GPHHConfig: GPHH policy configuration.
    GP_MP_HH_Config: GP-MP-HH policy configuration.
    HGSConfig: HGS policy configuration.
    HGSADCConfig: HGS-ADC policy configuration.
    HGSALNSConfig: HGS-ALNS policy configuration.
    HGSRRConfig: HGS-RR policy configuration.
    HMMGDHHConfig: HMM-GD-HH policy configuration.
    HybridMemeticLargeNeighborhoodSearchConfig: Hybrid memetic large neighborhood search policy configuration.
    HNAPolicyConfig: HNA policy configuration.
    HSConfig: HS policy configuration.
    HULKConfig: HULK policy configuration.
    HVPLConfig: HVPL policy configuration.
    ILSConfig: ILS policy configuration.
    IntegerLShapedBendersConfig: Integer L-shaped Benders policy configuration.
    ILSRVNDSPConfig: ILS-RVND-SP policy configuration.
    JointGreedyConfig: Joint greedy policy configuration.
    JointSAConfig: Joint SA policy configuration.
    KGLSConfig: KGLS policy configuration.
    KernelSearchConfig: Kernel search policy configuration.
    LocalBranchingConfig: Local branching policy configuration.
    LocalBranchingVNSConfig: Local branching-VNS policy configuration.
    LBBDConfig: LBBD policy configuration.
    LCAConfig: LCA policy configuration.
    LKH3Config: LKH3 policy configuration.
    LRHConfig: LRH policy configuration.
    MAConfig: MA policy configuration.
    MemeticAlgorithmDualPopulationConfig: Memetic algorithm dual population policy configuration.
    MemeticAlgorithmIslandModelConfig: Memetic algorithm island model policy configuration.
    MemeticAlgorithmToleranceBasedSelectionConfig: Memetic algorithm tolerance-based selection policy configuration.
    MHHConfig: MHH policy configuration.
    MP_ACO_Config: MP-ACO policy configuration.
    MP_ILS_Config: MP-ILS policy configuration.
    MP_PSO_Config: MP-PSO policy configuration.
    MP_SA_Config: MP-SA policy configuration.
    NeuralAgentConfig: Neural agent policy configuration.
    NDSBRKGAConfig: NDSBRKGA policy configuration.
    PackingConfig: Packing policy configuration.
    RouteImprovingConfig: Route improving policy configuration.
    PHConfig: PH policy configuration.
    PHHConfig: PHH policy configuration.
    POPMUSICConfig: POPMUSIC policy configuration.
    PSOConfig: PSO policy configuration.
    DistancePSOConfig: Distance PSO policy configuration.
    PSOMAConfig: PSOMA policy configuration.
    QDEConfig: QDE policy configuration.
    RENSConfig: RENS policy configuration.
    RFOConfig: RFO policy configuration.
    RLALNSConfig: RL-ALNS policy configuration.
    RLGDHHConfig: RL-GD-HH policy configuration.
    RLHVPLConfig: RL-HVPL policy configuration.
    RTSConfig: RTS policy configuration.
    SAConfig: SA policy configuration.
    SANSConfig: SANS policy configuration.
    SCAConfig: SCA policy configuration.
    SHHConfig: SHH policy configuration.
    SISRConfig: SISR policy configuration.
    SLCConfig: SLC policy configuration.
    SRCConfig: SRC policy configuration.
    SSHHConfig: SSHH policy configuration.
    ScenarioTreeExtensiveFormConfig: Scenario tree extensive form policy configuration.
    SWCTCFConfig: SWCTCF policy configuration.
    TSConfig: TS policy configuration.
    TSPConfig: TSP policy configuration.
    VNSConfig: VNS policy configuration.
    VPLConfig: VPL policy configuration.
    ExactGuidedHeuristicConfig: Exact Guided Heuristic policy configuration.
    PGCLNSConfig: PG-CLNS policy configuration.

Example:
    >>> from logic.src.configs.policies import ARCOConfig
    >>> config = ARCOConfig()
    >>> print(config)
    ARCOConfig(name='arco', solve_timeout=0.0, solution_timeout=0.0, solve_time_limit=600.0, solution_time_limit=1800.0, workers=1, solve_workers=None, verbose=0, logging_level='INFO', debug=False, random_seed=None, cache_dir='data/models', dataset_dir='data/datasets', policy_config=None, model=None, model_config=<ModelConfig(name='am', encoder=<EncoderConfig(type='gat', embed_dim=128, hidden_dim=512, n_layers=3, n_heads=8, n_sublayers=None, normalization=<NormalizationConfig(embed_dim=128, use_layer_norm=True)>, activation=<ActivationConfig(type='gelu')>, dropout=0.1, mask_inner=True, mask_graph=False, spatial_bias=False, connection_type='residual', aggregation_graph='avg', aggregation_node='sum', spatial_bias_scale=1.0, hyper_expansion=4)>, decoder=<DecoderConfig(type='attention', embed_dim=128, hidden_dim=512, n_layers=3, n_heads=8, normalization=<NormalizationConfig(embed_dim=128, use_layer_norm=True)>, activation=<ActivationConfig(type='gelu')>, decoding=<DecodingConfig(strategy='greedy', beam_width=1, temperature=1.0, top_k=None, top_p=None, tanh_clipping=0.0, mask_logits=True, multistart=False, num_starts=1, select_best=False)>, dropout=0.1, mask_logits=True, connection_type='residual', n_predictor_layers=None, tanh_clipping=10.0, hyper_expansion=4)>, reward=<ObjectiveConfig(type='vrpp')>, temporal_horizon=0, policy_config=None, load_path=None)>, training_config=None, dataset_config=None, rollout_config=None, evaluation_config=None, problem_config=None)
"""

from .abc import ABCConfig
from .abpc_hg import ABPCHGConfig
from .aco_hh import HyperHeuristicACOConfig
from .aco_ks import KSparseACOConfig
from .adp import ADPRolloutConfig
from .aks import AdaptiveKernelSearchConfig
from .alns import ALNSConfig
from .alns_ipo import ALNSIPOConfig
from .amphh import AMPHHConfig
from .arco import ARCOConfig
from .bb import BBConfig
from .bc import BCConfig
from .bp import BPConfig
from .bpc import BPCConfig
from .calm import CALMConfig
from .cf_rs import CFRSConfig
from .cgh import CGHConfig
from .cp_sat import CPSATConfig
from .cvrp import CVRPConfig
from .de import DEConfig
from .egh import ExactGuidedHeuristicConfig
from .es_mcl import MuCommaLambdaESConfig
from .es_mkl import MuKappaLambdaESConfig
from .es_mpl import MuPlusLambdaESConfig
from .esdp import ExactSDPConfig
from .fa import FAConfig
from .filo import FILOConfig
from .ga import GAConfig
from .gihh import GIHHConfig
from .gls import GLSConfig
from .gp_hh import GPHHConfig
from .gp_mp_hh import GP_MP_HH_Config
from .hgs import HGSConfig
from .hgs_adc import HGSADCConfig
from .hgs_alns import HGSALNSConfig
from .hgs_rr import HGSRRConfig
from .hmlns import HybridMemeticLargeNeighborhoodSearchConfig
from .hmm_gd_hh import HMMGDHHConfig
from .hna import HNAPolicyConfig
from .hs import HSConfig
from .hulk import HULKConfig
from .hvpl import HVPLConfig
from .ils import ILSConfig
from .ils_bd import IntegerLShapedBendersConfig
from .ils_rvnd_sp import ILSRVNDSPConfig
from .jgo import JointGreedyConfig
from .jsa import JointSAConfig
from .kgls import KGLSConfig
from .ks import KernelSearchConfig
from .lasm import LASMPipelineConfig
from .lb import LocalBranchingConfig
from .lb_vns import LocalBranchingVNSConfig
from .lbbd import LBBDConfig
from .lca import LCAConfig
from .lkh3 import LKH3Config
from .lrh import LRHConfig
from .ma import MAConfig
from .ma_dp import MemeticAlgorithmDualPopulationConfig
from .ma_im import MemeticAlgorithmIslandModelConfig
from .ma_ts import MemeticAlgorithmToleranceBasedSelectionConfig
from .mhh import MHHConfig
from .mp_aco import MP_ACO_Config
from .mp_ils import MP_ILS_Config
from .mp_pso import MP_PSO_Config
from .mp_sa import MP_SA_Config
from .ms_bpc_sp import MSBPCSPConfig
from .na import NeuralAgentConfig
from .nds_brkga import NDSBRKGAConfig
from .other import PackingConfig, RouteImprovingConfig
from .pg_clns import PGCLNSConfig
from .ph import PHConfig
from .phh import PHHConfig
from .popmusic import POPMUSICConfig
from .pso import PSOConfig
from .psoda import DistancePSOConfig
from .psoma import PSOMAConfig
from .qde import QDEConfig
from .rens import RENSConfig
from .rfo import RFOConfig
from .rl_alns import RLALNSConfig
from .rl_gd_hh import RLGDHHConfig
from .rl_hvpl import RLHVPLConfig
from .rts import RTSConfig
from .sa import SAConfig
from .sans import SANSConfig
from .sca import SCAConfig
from .shh import SHHConfig
from .sisr import SISRConfig
from .slc import SLCConfig
from .src import SRCConfig
from .ss_hh import SSHHConfig
from .st_ef import ScenarioTreeExtensiveFormConfig
from .swc_tcf import SWCTCFConfig
from .ts import TSConfig
from .tsp import TSPConfig
from .vns import VNSConfig
from .vpl import VPLConfig

__all__ = [
    "ARCOConfig",
    "AdaptiveKernelSearchConfig",
    "ABPCHGConfig",
    "ADPRolloutConfig",
    "KSparseACOConfig",
    "HyperHeuristicACOConfig",
    "ALNSConfig",
    "ALNSIPOConfig",
    "BBConfig",
    "BCConfig",
    "BPConfig",
    "BPCConfig",
    "CALMConfig",
    "CFRSConfig",
    "CVRPConfig",
    "DistancePSOConfig",
    "FILOConfig",
    "GIHHConfig",
    "HGSConfig",
    "HGSADCConfig",
    "HGSALNSConfig",
    "HGSRRConfig",
    "HNAPolicyConfig",
    "ILSRVNDSPConfig",
    "IntegerLShapedBendersConfig",
    "HVPLConfig",
    "KGLSConfig",
    "LASMPipelineConfig",
    "LKH3Config",
    "MSBPCSPConfig",
    "NeuralAgentConfig",
    "RLALNSConfig",
    "RLGDHHConfig",
    "RLHVPLConfig",
    "SANSConfig",
    "SISRConfig",
    "TSPConfig",
    "SWCTCFConfig",
    "POPMUSICConfig",
    "PackingConfig",
    "RouteImprovingConfig",
    "HULKConfig",
    "QDEConfig",
    "PSOMAConfig",
    "HybridMemeticLargeNeighborhoodSearchConfig",
    "KernelSearchConfig",
    "ABCConfig",
    "FAConfig",
    "SCAConfig",
    "HSConfig",
    "SLCConfig",
    "MemeticAlgorithmToleranceBasedSelectionConfig",
    "LCAConfig",
    "MemeticAlgorithmIslandModelConfig",
    "MuCommaLambdaESConfig",
    "MuKappaLambdaESConfig",
    "MuPlusLambdaESConfig",
    "GPHHConfig",
    "HMMGDHHConfig",
    "SSHHConfig",
    "VNSConfig",
    "VPLConfig",
    "MemeticAlgorithmDualPopulationConfig",
    "PSOConfig",
    "MAConfig",
    "TSConfig",
    "SAConfig",
    "LocalBranchingConfig",
    "LocalBranchingVNSConfig",
    "RENSConfig",
    "GAConfig",
    "DEConfig",
    "GLSConfig",
    "ILSConfig",
    "RTSConfig",
    "ExactSDPConfig",
    "PHConfig",
    "ScenarioTreeExtensiveFormConfig",
    "LBBDConfig",
    "CPSATConfig",
    "SRCConfig",
    "AMPHHConfig",
    "CGHConfig",
    "GP_MP_HH_Config",
    "LRHConfig",
    "MHHConfig",
    "MP_ACO_Config",
    "MP_SA_Config",
    "MP_ILS_Config",
    "MP_PSO_Config",
    "PHHConfig",
    "RFOConfig",
    "SHHConfig",
    "NDSBRKGAConfig",
    "JointSAConfig",
    "JointGreedyConfig",
    "ExactGuidedHeuristicConfig",
    "PGCLNSConfig",
]
