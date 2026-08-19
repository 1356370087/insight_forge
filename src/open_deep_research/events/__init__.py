"""Public run-event and task-activity streams surfaced over SSE.

Submodules:

- ``public``: run-level public events (store, publisher, payload allowlists).
- ``task_activity``: per-task fine-grained activity journals bridging into
  the run stream.

``__init__`` deliberately avoids eager re-exports because both submodules
import ``configuration`` at module load.
"""
