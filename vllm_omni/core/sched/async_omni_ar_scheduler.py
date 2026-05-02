"""
AsyncOmniARScheduler: ``OmniARScheduler`` variant that activates the
async-scheduling bookkeeping from ``vllm.v1.core.sched.async_scheduler``.

When ``async_scheduling=True``, vLLM's ``EngineCoreProc`` drives
``step_with_batch_queue``, which speculatively schedules the *next* batch
while the current one is still on the GPU. For that to actually keep the
queue full, the scheduler must increment ``request.num_output_placeholders``
after every scheduled step (so the next call to ``schedule()`` knows it can
launch one more decode token even though the previous step's output has
not been merged in yet) and decrement it again when the output arrives.

The base ``OmniARScheduler`` inherits from ``vllm.v1.core.sched.Scheduler``
and does **not** do this bookkeeping. As a result, with
``async_scheduling=True`` the scheduler returns 0 scheduled tokens on every
other engine step. The engine then sees ``model_executed=False`` and falls
through ``EngineCoreProc._process_engine_step`` 's
``time.sleep(0.001)`` guard, producing the alternating empty-step pattern
seen in profiles.

``AsyncOmniARScheduler`` injects ``AsyncScheduler`` into the MRO so the
``_update_after_schedule`` / ``_update_request_with_output`` overrides take
effect, while keeping every Omni-specific behaviour (OmniNewRequestData
wrapping, KV-transfer metadata, chunk-transfer adapter, streaming-session
hooks, etc.) provided by ``OmniARScheduler`` and ``OmniSchedulerMixin``.

Use this class instead of ``OmniARScheduler`` for any stage that sets
``async_scheduling: true``. Stages with ``async_scheduling: false`` should
keep using ``OmniARScheduler`` to preserve their current behaviour.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

logger = init_logger(__name__)


class AsyncOmniARScheduler(OmniARScheduler, AsyncScheduler):
    """OmniARScheduler with async-scheduling placeholder bookkeeping.

    MRO:
        AsyncOmniARScheduler
        -> OmniARScheduler
        -> OmniSchedulerMixin
        -> AsyncScheduler
        -> vllm.v1.core.sched.Scheduler
        -> object

    With this MRO:

    * ``OmniARScheduler.schedule()`` runs as before (calls
      ``super().schedule()`` which ends in ``Scheduler.schedule()``); inside,
      ``self._update_after_schedule(...)`` resolves to
      ``AsyncScheduler._update_after_schedule``, which calls back into
      ``Scheduler._update_after_schedule`` via ``super()`` and then bumps
      ``num_output_placeholders``.
    * ``OmniARScheduler.update_from_output()`` runs as before (Omni-specific
      KV-transfer / OmniModelRunnerOutput handling); inside, the call to
      ``self._update_request_with_output(...)`` resolves to
      ``AsyncScheduler._update_request_with_output``, which decrements
      ``num_output_placeholders`` and caches blocks at the correct offset
      for prefix caching.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.scheduler_config.async_scheduling:
            logger.warning(
                "AsyncOmniARScheduler is in use but async_scheduling=False. "
                "Use OmniARScheduler for sync stages; AsyncOmniARScheduler "
                "is only meaningful when async_scheduling=True."
            )
