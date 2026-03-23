import importlib
from types import MappingProxyType
from typing import Mapping

from regain.constants import PARAM_SCENARIO

__all__ = [
    'import_symbol',
    'list_scenarios',
    'list_backbones',
    'list_controllers',
    'list_lr_schedulers',
    'list_repair_buffer_policies',
    'get_scenario_builder_path',
    'get_backbone_path',
    'get_controller_path',
    'get_lr_scheduler_path',
    'get_repair_buffer_policy_path',
]


def import_symbol(full_path: str) -> type:
    """
    Import a symbol from a fully qualified import path.

    Args:
        full_path (str): Fully qualified import path (e.g., 'module.submodule.ClassName').

    Returns:
        type: Imported symbol.

    Raises:
        ValueError: If the path is not fully qualified or cannot be imported.
    """
    if not full_path or '.' not in full_path or full_path.startswith('.'):
        raise ValueError('`full_path` must be a fully qualified import path')
    module_path, _, symbol_name = full_path.rpartition('.')
    if not module_path or not symbol_name or module_path.startswith('.'):
        raise ValueError('`full_path` must be a fully qualified import path')
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ValueError(f'`{full_path}` not found') from exc
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f'`{full_path}` not found') from exc


###########
# Helpers #
###########


def _normalize_registry_key(*, value: object, label: str) -> str:
    """
    Normalize a registry key and validate its type.

    Args:
        value: Raw registry key.
        label: Error message label.

    Returns:
        str: Normalized registry key.

    Raises:
        ValueError: If the key is invalid.
    """
    if not isinstance(value, str):
        raise ValueError(f'{label} must be a string')
    key = value.strip()
    if not key:
        raise ValueError(f'{label} must be a non-empty string')
    return key


def _resolve_registry_entry(
    *,
    mapping: Mapping[str, str],
    key: str,
    label: str,
    supported_label: str,
) -> str:
    """
    Resolve a registry entry to its path.

    Args:
        mapping: Registry mapping of keys to paths.
        key: Normalized registry key.
        label: Error message label.
        supported_label: Error message label for supported values.

    Returns:
        str: Fully qualified path.

    Raises:
        ValueError: If the key is not registered.
    """
    path = mapping.get(key)
    if path is None:
        supported_names = ', '.join(sorted(mapping)) or 'none'
        raise ValueError(
            f'Unsupported {label}: {key}. Supported {supported_label}: {supported_names}.'
        )
    return path


def _list_registry_entries(mapping: Mapping[str, str]) -> tuple[str, ...]:
    """
    List registry keys in sorted order.

    Args:
        mapping: Registry mapping of keys to class paths.

    Returns:
        tuple[str, ...]: Sorted registry keys.
    """
    return tuple(sorted(mapping))


#############
# Scenarios #
#############

_SCENARIOS: Mapping[str, str] = MappingProxyType(
    {
        'split_cifar100': 'regain.avalanche_utils.scenarios.SplitCIFAR100ScenarioBuilder',
        'split_imagenet_r': 'regain.avalanche_utils.scenarios.SplitImageNetRScenarioBuilder',
        'split_tiny_imagenet': 'regain.avalanche_utils.scenarios.SplitTinyImageNetScenarioBuilder',
    }
)


def list_scenarios() -> tuple[str, ...]:
    """
    List available scenario registry names.

    Returns:
        tuple[str, ...]: Sorted scenario names.
    """
    return _list_registry_entries(_SCENARIOS)


def get_scenario_builder_path(scenario: str) -> str:
    """
    Resolve a scenario name to its fully qualified builder path.

    Args:
        scenario (str): Scenario registry name.

    Returns:
        str: Fully qualified path for the scenario builder.

    Raises:
        ValueError: If the scenario name is invalid or unsupported.
    """
    key = _normalize_registry_key(value=scenario, label='Scenario name')
    return _resolve_registry_entry(
        mapping=_SCENARIOS,
        key=key,
        label=PARAM_SCENARIO,
        supported_label='scenarios',
    )


#############
# Backbones #
#############

_BACKBONES: Mapping[str, str] = MappingProxyType(
    {
        'resnet18': 'regain.models.classifiers.ResNet18Classifier',
        'vit_base': 'regain.models.classifiers.ViTBaseClassifier',
        'vit_small': 'regain.models.classifiers.ViTSmallClassifier',
    }
)


def list_backbones() -> tuple[str, ...]:
    """
    List available backbone registry names.

    Returns:
        tuple[str, ...]: Sorted backbone names.
    """
    return _list_registry_entries(_BACKBONES)


def get_backbone_path(name: str) -> str:
    """
    Resolve a backbone name to its fully qualified path.

    Args:
        name (str): Backbone registry name.

    Returns:
        str: Fully qualified path for the backbone class.

    Raises:
        ValueError: If the backbone name is invalid or unsupported.
    """
    key = _normalize_registry_key(value=name, label='Backbone name')
    return _resolve_registry_entry(
        mapping=_BACKBONES,
        key=key,
        label='backbone name',
        supported_label='backbones',
    )


#################
# LR Schedulers #
#################

_LR_SCHEDULERS: Mapping[str, str] = MappingProxyType(
    {
        'multi_step': 'torch.optim.lr_scheduler.MultiStepLR',
        'warmup_cosine': 'regain.experiments.lr_schedulers.WarmupCosineLR',
    }
)


def list_lr_schedulers() -> tuple[str, ...]:
    """
    List available LR scheduler registry names.

    Returns:
        tuple[str, ...]: Sorted LR scheduler names.
    """
    return _list_registry_entries(_LR_SCHEDULERS)


def get_lr_scheduler_path(name: str) -> str:
    """
    Resolve an LR scheduler name to its fully qualified path.

    Args:
        name (str): LR scheduler registry name.

    Returns:
        str: Fully qualified path for the LR scheduler class.

    Raises:
        ValueError: If the LR scheduler name is invalid or unsupported.
    """
    key = _normalize_registry_key(value=name, label='LR scheduler name')
    return _resolve_registry_entry(
        mapping=_LR_SCHEDULERS,
        key=key,
        label='LR scheduler name',
        supported_label='LR schedulers',
    )


###############
# Controllers #
###############

_CONTROLLERS: Mapping[str, str] = MappingProxyType(
    {
        'bace': 'regain.models.controllers.prevention.BaCEController',
        'bic': 'regain.models.controllers.repair.BiCController',
        'channel_block': 'regain.models.controllers.repair.ChannelBlockGainController',
        'channel_stage': 'regain.models.controllers.repair.ChannelStageGainController',
        'cn': 'regain.models.controllers.prevention.ContinualNormalizationController',
        'conditioned_block': 'regain.models.controllers.repair.InputConditionedBlockGainController',
        'conditioned_stage': 'regain.models.controllers.repair.InputConditionedStageGainController',
        'il2m': 'regain.models.controllers.repair.IL2MController',
        'linear_probe': 'regain.models.controllers.repair.LinearProbeController',
        'logit_bias': 'regain.models.controllers.repair.LogitBiasController',
        'scalar_block': 'regain.models.controllers.repair.ScalarBlockGainController',
        'scalar_stage': 'regain.models.controllers.repair.ScalarStageGainController',
        'tbbn': 'regain.models.controllers.prevention.TaskBalancedBatchNormController',
    }
)


def list_controllers() -> tuple[str, ...]:
    """
    List available controller registry names.

    Returns:
        tuple[str, ...]: Sorted controller names.
    """
    return _list_registry_entries(_CONTROLLERS)


def get_controller_path(name: str) -> str:
    """
    Resolve a controller name to its fully qualified path.

    Args:
        name (str): Controller registry name.

    Returns:
        str: Fully qualified path for the controller.

    Raises:
        ValueError: If the controller name is invalid or unsupported.
    """
    key = _normalize_registry_key(value=name, label='Controller name')
    return _resolve_registry_entry(
        mapping=_CONTROLLERS,
        key=key,
        label='controller name',
        supported_label='controllers',
    )


###############################################################
# Repair buffer policies                                      #
#                                                             #
# Optional repair-buffer policy registry.                     #
###############################################################

_REPAIR_BUFFER_POLICIES: Mapping[str, str] = MappingProxyType(
    {
        'balanced_fifo': 'regain.models.controllers.sampling.RepairBufferBalancedFIFOPolicy',
        'fifo': 'regain.models.controllers.sampling.RepairBufferFIFOPolicy',
    }
)


def list_repair_buffer_policies() -> tuple[str, ...]:
    """
    List available repair buffer policy registry names.

    Returns:
        tuple[str, ...]: Sorted policy names.
    """
    return _list_registry_entries(_REPAIR_BUFFER_POLICIES)


def get_repair_buffer_policy_path(name: str) -> str:
    """
    Resolve a repair buffer policy name to its fully qualified path.

    Args:
        name (str): Repair buffer policy registry name.

    Returns:
        str: Fully qualified path for the repair buffer policy.

    Raises:
        ValueError: If the policy name is invalid or unsupported.
    """
    key = _normalize_registry_key(value=name, label='Repair buffer policy name')
    return _resolve_registry_entry(
        mapping=_REPAIR_BUFFER_POLICIES,
        key=key,
        label='repair buffer policy',
        supported_label='policies',
    )
