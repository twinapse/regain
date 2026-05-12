"""
Frozen-backbone repair controllers.

This package provides small post-hoc controllers trained on a limited repair set and applied at evaluation time,
without updating the backbone. Controllers cover logit calibration, static gain modulation (per unit or per
channel group), and input-conditioned gain modulation, plus a linear-probe baseline.

Gain-based controllers target ResNet and ViT backbones via stage/block unit discovery.
"""

from regain.models.controllers.repair.calibration import BiCController
from regain.models.controllers.repair.calibration import IL2MController
from regain.models.controllers.repair.calibration import LogitBiasController
from regain.models.controllers.repair.calibration import TCILLiteController
from regain.models.controllers.repair.calibration import TemperatureScalingController
from regain.models.controllers.repair.calibration import WeightAligningController
from regain.models.controllers.repair.channel_gains import ChannelBlockGainController
from regain.models.controllers.repair.channel_gains import ChannelStageGainController
from regain.models.controllers.repair.conditioned_gains import InputConditionedBlockGainController
from regain.models.controllers.repair.conditioned_gains import InputConditionedStageGainController
from regain.models.controllers.repair.probing import LinearProbeController
from regain.models.controllers.repair.probing import PrototypeBlendController
from regain.models.controllers.repair.scalar_gains import ScalarBlockGainController
from regain.models.controllers.repair.scalar_gains import ScalarStageGainController

__all__ = [
    'BiCController',
    'ChannelBlockGainController',
    'ChannelStageGainController',
    'IL2MController',
    'InputConditionedBlockGainController',
    'InputConditionedStageGainController',
    'LinearProbeController',
    'LogitBiasController',
    'PrototypeBlendController',
    'ScalarBlockGainController',
    'ScalarStageGainController',
    'TCILLiteController',
    'TemperatureScalingController',
    'WeightAligningController',
]
