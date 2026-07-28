from configs.config_schema import PolicyConfig

from models.policy.iron_vla import IronVLA

POLICY_REGISTRY: dict[str, type[IronVLA]] = {
    'IRON_VLA': IronVLA,
    'HEAD_ONLY': IronVLA,
}


def build_model(policy_config: PolicyConfig) -> IronVLA:
    """Build a policy model from a validated PolicyConfig."""
    policy_class_name = policy_config.policy_class
    if policy_class_name not in POLICY_REGISTRY:
        raise ValueError(
            f'Policy class {policy_class_name!r} not found. '
            f'Available: {list(POLICY_REGISTRY.keys())}'
        )

    policy_cls = POLICY_REGISTRY[policy_class_name]
    policy = policy_cls(policy_config)

    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in policy.parameters())
    print('number of parameters: %.2fM' % (n_parameters / 1e6,))
    print('total parameterss: %.2fM' % (total_parameters / 1e6,))

    return policy
