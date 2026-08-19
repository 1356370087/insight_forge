"""Quality gates, coverage contracts, and evaluation policy.

Submodules:

- ``gate``: tool-batch and handoff evaluation combining deterministic
  gates with Judge semantic scoring.
- ``contract``: coverage-contract compilation and source-scope rules.
- ``policy``: quality policy versioning and rigor resolution.

``__init__`` deliberately avoids eager re-exports: ``configuration``
imports ``quality.policy`` at module load, so importing ``gate`` from
here would create an import cycle.
"""
