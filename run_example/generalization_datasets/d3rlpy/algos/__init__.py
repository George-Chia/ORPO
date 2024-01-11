from typing import Any, Dict, Type

from .awac import AWAC
from .base import AlgoBase
from .baseORPOSim import AlgoBaseORPOSim
from .bc import BC, DiscreteBC
from .bcq import BCQ, DiscreteBCQ
from .bear import BEAR
from .combo import COMBO
from .cql import CQL, DiscreteCQL
from .cqlORPOSim import CQLORPOSim
from .crr import CRR
from .ddpg import DDPG
from .dqn import DQN, DoubleDQN
from .iql import IQL
from .iqlOneEpoch import IQLOneEpoch
from .iqlORPOSim import IQLORPOSim
from .mopo import MOPO
from .nfq import NFQ
from .plas import PLAS, PLASWithPerturbation
from .random_policy import DiscreteRandomPolicy, RandomPolicy
from .random_policyRecord2rewards import RandomPolicyRecord2rewards
from .sac import SAC, DiscreteSAC
from .sacOneEpoch import SACOneEpoch
from .sacORPOSim import SACORPOSim
from .sacRecord2rewards import SACRecord2rewards
from .td3 import TD3
from .td3_plus_bc import TD3PlusBC
from .td3_plus_bcOneEpoch import TD3PlusBCOneEpoch
from .td3_plus_bcORPOSim import TD3PlusBCORPOSim

__all__ = [
    "AlgoBase",
    "AlgoBaseORPOSim",
    "AWAC",
    "BC",
    "DiscreteBC",
    "BCQ",
    "DiscreteBCQ",
    "BEAR",
    "COMBO",
    "CQL",
    "CQLORPOSim",
    "DiscreteCQL",
    "CRR",
    "DDPG",
    "DQN",
    "DoubleDQN",
    "IQL",
    "IQLOneEpoch",
    "IQLORPOSim",
    "MOPO",
    "NFQ",
    "PLAS",
    "PLASWithPerturbation",
    "SAC",
    "SACOneEpoch",
    "SACORPOSim",
    "SACRecord2rewards",
    "DiscreteSAC",
    "TD3",
    "TD3PlusBC",
    "TD3PlusBCOneEpoch",
    "TD3PlusBCORPOSim",
    "RandomPolicy",
    "RandomPolicyRecord2rewards",
    "DiscreteRandomPolicy",
    "get_algo",
    "create_algo",
]


DISCRETE_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "bc": DiscreteBC,
    "bcq": DiscreteBCQ,
    "cql": DiscreteCQL,
    "dqn": DQN,
    "double_dqn": DoubleDQN,
    "nfq": NFQ,
    "sac": DiscreteSAC,
    "random": DiscreteRandomPolicy,
}

CONTINUOUS_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "awac": AWAC,
    "bc": BC,
    "bcq": BCQ,
    "bear": BEAR,
    "combo": COMBO,
    "cql": CQL,
    "CQLORPOSim":CQLORPOSim,
    "crr": CRR,
    "ddpg": DDPG,
    "iql": IQL,
    "iqlOneEpoch": IQLOneEpoch,
    "IQLORPOSim": IQLORPOSim,
    "mopo": MOPO,
    "plas": PLASWithPerturbation,
    "sac": SAC,
    "sacOneEpoch": SACOneEpoch,
    "SACORPOSim":TD3PlusBCORPOSim,
    "SACRecord2rewards":SACRecord2rewards,
    "td3": TD3,
    "td3_plus_bc": TD3PlusBC,
    "td3_plus_bcOneEpoch": TD3PlusBCOneEpoch,
    "TD3PlusBCORPOSim":TD3PlusBCORPOSim,
    "random": RandomPolicy,
}


def get_algo(name: str, discrete: bool) -> Type[AlgoBase]:
    """Returns algorithm class from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.

    Returns:
        type: algorithm class.

    """
    if discrete:
        if name in DISCRETE_ALGORITHMS:
            return DISCRETE_ALGORITHMS[name]
        raise ValueError(f"{name} does not support discrete action-space.")
    if name in CONTINUOUS_ALGORITHMS:
        return CONTINUOUS_ALGORITHMS[name]
    raise ValueError(f"{name} does not support continuous action-space.")


def create_algo(name: str, discrete: bool, **params: Any) -> AlgoBase:
    """Returns algorithm object from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.
        params (any): arguments for algorithm.

    Returns:
        d3rlpy.algos.base.AlgoBase: algorithm.

    """
    return get_algo(name, discrete)(**params)
