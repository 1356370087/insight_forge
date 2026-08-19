"""Model identity, resolution, recovery, and circuit governance.

Submodules:

- ``resolution``: provider inference, API keys, and model config assembly.
- ``errors``: provider token-limit error detection.
- ``fallback``: candidate-chain routing and error classification.
- ``circuit``: process-local three-state circuit breakers.
- ``capabilities``: declarative model capability metadata.
- ``limits``: MODEL_TOKEN_LIMITS registry and lookups.

The package intentionally lives below ``agents`` and ``tools`` so both layers
can reuse it without reversing their dependency direction.
"""
