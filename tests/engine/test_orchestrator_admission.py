# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from vllm_omni.distributed.omni_coordinator.load_balancer import RoundRobinBalancer
from vllm_omni.distributed.omni_coordinator.messages import ReplicaInfo, ReplicaList, ReplicaStatus
from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AbortResultMessage,
    AddCompanionRequestMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    ErrorMessage,
    InteractionMessage,
    RegisterRemoteReplicaMessage,
    ShutdownRequestMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool

from .test_orchestrator import (
    FakeOutputProcessor,
    FakePromptRequest,
    FakeRunningCounter,
    FakeStageClient,
    _build_stage_pools,
    _sampling_params,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class AdmissionStageClient(FakeStageClient):
    def __init__(self, input_address: str) -> None:
        super().__init__(final_output=True)
        self.client_addresses = {"input_address": input_address}
        self.submissions: asyncio.Queue[FakePromptRequest] = asyncio.Queue()
        self.submission_started: asyncio.Queue[FakePromptRequest] = asyncio.Queue()
        self.request_gates: dict[str, asyncio.Event] = {}
        self.submit_gate = asyncio.Event()
        self.submit_gate.set()
        self.rpc_started = asyncio.Event()
        self.rpc_gate = asyncio.Event()
        self.rpc_gate.set()

    async def add_request_async(self, request, **kwargs) -> None:
        self.submission_started.put_nowait(request)
        await self.request_gates.get(request.request_id, self.submit_gate).wait()
        await super().add_request_async(request, **kwargs)
        self.submissions.put_nowait(request)

    async def collective_rpc_async(self, **kwargs):
        self.rpc_started.set()
        await self.rpc_gate.wait()
        return await super().collective_rpc_async(**kwargs)


class AdmissionOutputProcessor(FakeOutputProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.requests: dict[str, FakePromptRequest] = {}

    def add_request(self, *args, **kwargs) -> None:
        super().add_request(*args, **kwargs)
        request = kwargs["request"]
        self.requests[request.request_id] = request

    def commit_aborted_request_state(self, request_ids, *, internal: bool = False) -> None:
        for request_id in request_ids:
            self.requests.pop(request_id, None)

    def remove_request(self, request_id: str) -> None:
        self.requests.pop(request_id, None)


class AdmissionHub:
    def __init__(self, replacement: AdmissionStageClient, stage_id: int = 0) -> None:
        # The hub advertises an UP replacement before its client is attached.
        # An older attached slot keeps stage 0's empty-pool guard from firing.
        self.snapshot = ReplicaList(
            replicas=[
                ReplicaInfo(
                    input_addr=replacement.client_addresses["input_address"],
                    output_addr="tcp://replacement-output",
                    stage_id=stage_id,
                    status=ReplicaStatus.UP,
                    queue_length=0,
                    last_heartbeat=0,
                    registered_at=0,
                )
            ],
            timestamp=0,
        )
        self.queried = asyncio.Event()

    def get_replicas_for_stage(self, stage_id: int) -> ReplicaList:
        assert stage_id == self.snapshot.replicas[0].stage_id
        self.queried.set()
        return self.snapshot


class AdmissionMembership:
    def __init__(self, pool: StagePool, replacement: AdmissionStageClient) -> None:
        self.pool = pool
        self.replacement = replacement
        self.registered = asyncio.Event()

    async def handle_register(self, stage_id: int, replica_id: int) -> None:
        assert stage_id == self.pool.stage_id
        self.pool.add_client(
            self.replacement.client_addresses["input_address"], self.replacement, replica_id=replica_id
        )
        self.registered.set()


@dataclass
class AdmissionHarness:
    orchestrator: Orchestrator
    pool: StagePool
    hub: AdmissionHub
    membership: AdmissionMembership
    original: AdmissionStageClient
    replacement: AdmissionStageClient
    running: FakeRunningCounter


@pytest.fixture
def admission(monkeypatch: pytest.MonkeyPatch) -> AdmissionHarness:
    original = AdmissionStageClient("tcp://original-input")
    replacement = AdmissionStageClient("tcp://replacement-input")
    pool = StagePool(0, [original], output_processor=AdmissionOutputProcessor())
    hub = AdmissionHub(replacement)
    pool.attach_hub(hub)
    pool.attach_load_balancer(RoundRobinBalancer())
    # A failing control wait must expire well before production dispatch does.
    monkeypatch.setattr(pool, "DISPATCH_WAIT_TIMEOUT_S", 30.0)
    monkeypatch.setattr(pool, "DISPATCH_RETRY_INTERVAL_S", 0.005)
    membership = AdmissionMembership(pool, replacement)
    running = FakeRunningCounter()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=[pool],
        membership_controller=membership,
        running_counter=running,
    )
    hub.queried.clear()
    return AdmissionHarness(orchestrator, pool, hub, membership, original, replacement, running)


def submission(request_id: str, tokens: list[int], *, update: bool = False) -> StageSubmissionMessage:
    prompt = FakePromptRequest(request_id, tokens)
    return StageSubmissionMessage(
        type="streaming_update" if update else "add_request",
        request_id=request_id,
        prompt=prompt,
        original_prompt=prompt,
        output_prompt_text=None,
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
        preprocess_ms=0,
        request_timestamp=0,
        enqueue_ts=0,
    )


@asynccontextmanager
async def request_handlers(orchestrator: Orchestrator) -> AsyncIterator[asyncio.Task]:
    # run() owns its event loop and cancels all remaining tasks on teardown;
    # exercise its request/admission consumers without cancelling pytest.
    request_task = asyncio.create_task(orchestrator._request_handler())
    tasks = [request_task]
    admission_handler = getattr(orchestrator, "_admission_handler", None)
    if admission_handler is not None:
        tasks.append(asyncio.create_task(admission_handler()))
    try:
        yield request_task
    finally:
        orchestrator._shutdown_event.set()
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result


async def start_waiting_request(admission: AdmissionHarness, request_id: str = "waiting") -> None:
    admission.orchestrator.request_async_queue.put_nowait(submission(request_id, [1]))
    await asyncio.wait_for(admission.hub.queried.wait(), timeout=1)
    assert admission.pool.live_num_replicas == 1
    assert admission.original.add_request_calls == []
    assert admission.replacement.add_request_calls == []
    assert request_id in admission.orchestrator.request_states


@pytest.mark.asyncio
async def test_registration_recovers_waiting_stage_admission(admission: AdmissionHarness) -> None:
    async with request_handlers(admission.orchestrator):
        await start_waiting_request(admission)
        admission.orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))

        await asyncio.wait_for(admission.membership.registered.wait(), timeout=1)
        request = await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)

        assert request.request_id == "waiting"
        assert admission.pool.get_bound_replica_id("waiting") == 1
        assert admission.original.add_request_calls == []
        assert admission.running.value == 1
        assert admission.orchestrator.output_async_queue.empty()


@pytest.mark.asyncio
async def test_abort_acknowledges_waiting_admission_without_late_submission(admission: AdmissionHarness) -> None:
    async with request_handlers(admission.orchestrator):
        await start_waiting_request(admission)
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["waiting"], rpc_id="cancel-waiting")
        )

        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.rpc_id == "cancel-waiting"
        assert result.success
        assert "waiting" not in admission.orchestrator.request_states
        assert admission.running.value == 0
        assert admission.pool.get_bound_replica_id("waiting") is None

        admission.orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))
        admission.orchestrator.request_async_queue.put_nowait(submission("survivor", [2]))
        request = await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)

        assert request.request_id == "survivor"
        assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["survivor"]
        assert admission.original.add_request_calls == []
        assert "waiting" not in admission.orchestrator.request_states
        assert admission.pool.get_bound_replica_id("waiting") is None
        assert admission.orchestrator.output_async_queue.empty()


@pytest.mark.asyncio
async def test_shutdown_is_received_while_stage_admission_waits(admission: AdmissionHarness) -> None:
    async with request_handlers(admission.orchestrator) as request_task:
        await start_waiting_request(admission)
        admission.orchestrator.request_async_queue.put_nowait(ShutdownRequestMessage())

        await asyncio.wait_for(asyncio.shield(request_task), timeout=1)

        assert admission.orchestrator._shutdown_event.is_set()
        assert admission.original.add_request_calls == []
        assert admission.replacement.add_request_calls == []


@pytest.mark.asyncio
async def test_streaming_update_waits_for_initial_submission(admission: AdmissionHarness) -> None:
    await admission.membership.handle_register(0, 1)
    admission.replacement.submit_gate.clear()
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("stream", [1]))
        first = await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
        assert first.prompt_token_ids == [1]
        admission.orchestrator.request_async_queue.put_nowait(submission("stream", [2, 3], update=True))
        # This acknowledgment proves the consumer read past the update while
        # the initial client submission remains blocked.
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["unrelated"], rpc_id="control")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.success
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=0.05)

        admission.replacement.submit_gate.set()
        requests = [await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1) for _ in range(2)]
        assert [request.prompt_token_ids for request in requests] == [[1], [2, 3]]
        assert admission.running.value == 1
        assert admission.orchestrator.request_states["stream"].streaming.enabled


def companion(parent_id: str = "parent", companion_id: str = "negative") -> AddCompanionRequestMessage:
    return AddCompanionRequestMessage(
        parent_id=parent_id,
        companion_id=companion_id,
        role="negative",
        prompt=FakePromptRequest(companion_id, [2]),
        companion_prompt_text=None,
        sampling_params_list=[_sampling_params()],
    )


@pytest.mark.asyncio
async def test_cfg_companion_waits_for_parent_submission(admission: AdmissionHarness) -> None:
    await admission.membership.handle_register(0, 1)
    admission.replacement.submit_gate.clear()
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("parent", [1]))
        await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
        admission.orchestrator.request_async_queue.put_nowait(companion())
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["unrelated"], rpc_id="control")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.success
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=0.05)

        admission.replacement.submit_gate.set()
        requests = [await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1) for _ in range(2)]
        assert [request.request_id for request in requests] == ["parent", "negative"]
        assert admission.pool.get_bound_replica_id("negative") == admission.pool.get_bound_replica_id("parent")
        assert admission.orchestrator._cfg_tracker.get_parent_id("negative") == "parent"


@pytest.mark.asyncio
async def test_update_for_queued_companion_waits_until_companion_is_registered(admission: AdmissionHarness) -> None:
    await admission.membership.handle_register(0, 1)
    admission.replacement.request_gates["parent"] = asyncio.Event()
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("parent", [1]))
        await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
        admission.orchestrator.request_async_queue.put_nowait(companion())
        admission.orchestrator.request_async_queue.put_nowait(submission("negative", [3, 4], update=True))
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["unrelated"], rpc_id="update-consumed")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.rpc_id == "update-consumed"
        assert result.success
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=0.05)
        assert "negative" not in admission.orchestrator.request_states

        admission.replacement.request_gates["parent"].set()
        requests = [await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1) for _ in range(3)]
        assert [(request.request_id, request.prompt_token_ids) for request in requests] == [
            ("parent", [1]),
            ("negative", [2]),
            ("negative", [3, 4]),
        ]
        assert admission.orchestrator._cfg_tracker.get_parent_id("negative") == "parent"
        assert admission.pool.get_bound_replica_id("negative") == admission.pool.get_bound_replica_id("parent")
        assert admission.orchestrator.output_async_queue.empty()


@pytest.mark.asyncio
async def test_update_for_registered_companion_waits_for_its_initial_submission(admission: AdmissionHarness) -> None:
    await admission.membership.handle_register(0, 1)
    admission.replacement.request_gates["negative"] = asyncio.Event()
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("parent", [1]))
        await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
        admission.replacement.submission_started.get_nowait()
        admission.orchestrator.request_async_queue.put_nowait(companion())
        started = await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
        assert started.request_id == "negative"
        assert admission.orchestrator._cfg_tracker.get_parent_id("negative") == "parent"
        admission.orchestrator.request_async_queue.put_nowait(submission("negative", [3, 4], update=True))
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["unrelated"], rpc_id="update-consumed")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.rpc_id == "update-consumed"
        assert result.success
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=0.05)

        admission.replacement.request_gates["negative"].set()
        requests = [await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1) for _ in range(2)]
        assert [(request.request_id, request.prompt_token_ids) for request in requests] == [
            ("negative", [2]),
            ("negative", [3, 4]),
        ]
        assert admission.pool.output_processor.requests["negative"] is requests[1]
        assert admission.orchestrator.output_async_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_id", ["parent", "negative"], ids=["queued-companion", "registered-companion"])
async def test_companion_interaction_waits_for_initial_submission(
    admission: AdmissionHarness, monkeypatch: pytest.MonkeyPatch, blocked_id: str
) -> None:
    await admission.membership.handle_register(0, 1)
    admission.replacement.request_gates[blocked_id] = asyncio.Event()
    interactions: asyncio.Queue[tuple[str, dict, int]] = asyncio.Queue()

    async def submit_interaction(request_id: str, interaction: dict) -> None:
        interactions.put_nowait((request_id, interaction, len(admission.replacement.add_request_calls)))

    monkeypatch.setattr(admission.replacement, "submit_interaction_async", submit_interaction, raising=False)
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("parent", [1]))
        await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
        admission.orchestrator.request_async_queue.put_nowait(companion())
        if blocked_id == "negative":
            started = await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
            assert started.request_id == "negative"
        admission.orchestrator.request_async_queue.put_nowait(
            InteractionMessage(
                request_id="negative",
                interaction={"event_id": "change-negative", "event": {"prompt": "new conditioning"}},
            )
        )
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["unrelated"], rpc_id="interaction-consumed")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.rpc_id == "interaction-consumed"
        assert result.success
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(interactions.get(), timeout=0.05)
        assert admission.orchestrator.output_async_queue.empty()

        admission.replacement.request_gates[blocked_id].set()
        request_id, interaction, preceding_submissions = await asyncio.wait_for(interactions.get(), timeout=1)
        assert request_id == "negative"
        assert interaction["event_id"] == "change-negative"
        assert preceding_submissions == 2
        assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["parent", "negative"]
        assert admission.orchestrator.output_async_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("aborted_id", ["parent", "negative"])
async def test_abort_removes_queued_cfg_group(admission: AdmissionHarness, aborted_id: str) -> None:
    async with request_handlers(admission.orchestrator):
        await start_waiting_request(admission, "parent")
        admission.orchestrator.request_async_queue.put_nowait(companion())
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=[aborted_id], rpc_id="cancel-cfg")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.success
        assert not admission.orchestrator.request_states
        assert admission.running.value == 0

        admission.orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))
        admission.orchestrator.request_async_queue.put_nowait(submission("survivor", [3]))
        request = await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
        assert request.request_id == "survivor"
        assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["survivor"]
        assert admission.pool.get_bound_replica_id("parent") is None
        assert admission.pool.get_bound_replica_id("negative") is None
        assert not admission.orchestrator._cfg_tracker.has_companions("parent")
        if aborted_id == "negative":
            error = admission.orchestrator.output_async_queue.get_nowait()
            assert isinstance(error, ErrorMessage)
            assert error.request_id == "parent"
            assert "negative" in error.error
        assert admission.orchestrator.output_async_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["pause_scheduler", "sleep"])
async def test_collective_rpc_fences_admissions_but_not_abort(admission: AdmissionHarness, method: str) -> None:
    admission.original.rpc_gate.clear()
    async with request_handlers(admission.orchestrator):
        await start_waiting_request(admission)
        admission.orchestrator.request_async_queue.put_nowait(
            CollectiveRPCRequestMessage(rpc_id="fence", method=method, args=(), kwargs={}, stage_ids=None)
        )
        admission.orchestrator.request_async_queue.put_nowait(submission("later", [2]))
        admission.orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))
        await asyncio.wait_for(admission.original.rpc_started.wait(), timeout=1)

        assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["waiting"]
        assert "later" not in admission.orchestrator.request_states
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["waiting"], rpc_id="control")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.rpc_id == "control"
        assert result.success
        assert "later" not in admission.orchestrator.request_states

        admission.original.rpc_gate.set()
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, CollectiveRPCResultMessage)
        assert result.rpc_id == "fence"
        requests = [await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1) for _ in range(2)]
        assert [request.request_id for request in requests] == ["waiting", "later"]
        assert admission.replacement.collective_rpc_calls[0][0] == method


@pytest.mark.asyncio
async def test_admission_overflow_preserves_control_and_other_requests(
    admission: AdmissionHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vllm_omni.engine.orchestrator._MAX_CONCURRENT_ADMISSIONS", 1)
    monkeypatch.setattr("vllm_omni.engine.orchestrator._MAX_PENDING_ADMISSIONS", 2)
    async with request_handlers(admission.orchestrator):
        await start_waiting_request(admission)
        for request_id in ("queued-1", "queued-2", "overflow"):
            admission.orchestrator.request_async_queue.put_nowait(submission(request_id, [2]))
        error = await asyncio.wait_for(admission.orchestrator.output_async_queue.get(), timeout=1)
        assert isinstance(error, ErrorMessage)
        assert error.request_id == "overflow"
        assert error.status_code == 429
        assert "overflow" not in admission.orchestrator.request_states

        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["waiting", "queued-1"], rpc_id="free-capacity")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.success
        admission.orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))
        request = await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
        assert request.request_id == "queued-2"
        assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["queued-2"]
        assert set(admission.orchestrator.request_states) == {"queued-2"}
        assert admission.running.value == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True], ids=["register", "abort"])
async def test_control_progress_during_downstream_async_chunk_prewarm(
    monkeypatch: pytest.MonkeyPatch, abort: bool
) -> None:
    stage0 = AdmissionStageClient("tcp://stage0-input")
    stage0.final_output = False
    old_stage1 = AdmissionStageClient("tcp://stage1-old-input")
    replacement = AdmissionStageClient("tcp://stage1-new-input")
    pools = _build_stage_pools([[stage0], [old_stage1]])
    hub = AdmissionHub(replacement, stage_id=1)
    pools[1].attach_hub(hub)
    pools[1].attach_load_balancer(RoundRobinBalancer())
    monkeypatch.setattr(pools[1], "DISPATCH_WAIT_TIMEOUT_S", 30.0)
    monkeypatch.setattr(pools[1], "DISPATCH_RETRY_INTERVAL_S", 0.005)
    membership = AdmissionMembership(pools[1], replacement)
    running = FakeRunningCounter()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=pools,
        membership_controller=membership,
        running_counter=running,
        async_chunk=True,
    )
    request = submission("prewarm", [1, 2])
    request.final_stage_id = 1
    request.sampling_params_list.append(_sampling_params())
    hub.queried.clear()
    async with request_handlers(orchestrator):
        orchestrator.request_async_queue.put_nowait(request)
        await asyncio.wait_for(hub.queried.wait(), timeout=1)
        assert [args[0].request_id for args in stage0.add_request_calls] == ["prewarm"]
        assert replacement.add_request_calls == []

        if abort:
            orchestrator.request_async_queue.put_nowait(
                AbortRequestMessage(request_ids=["prewarm"], rpc_id="cancel-prewarm")
            )
            result = await asyncio.wait_for(orchestrator.rpc_async_queue.get(), timeout=1)
            assert isinstance(result, AbortResultMessage)
            assert result.success
            assert stage0.abort_calls == [["prewarm"]]
            assert not orchestrator.request_states
            assert running.value == 0
        orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=1, replica_id=1))
        if abort:
            survivor = submission("survivor", [3])
            survivor.final_stage_id = 1
            survivor.sampling_params_list.append(_sampling_params())
            orchestrator.request_async_queue.put_nowait(survivor)
        submitted = await asyncio.wait_for(replacement.submissions.get(), timeout=1)
        assert submitted.request_id == ("survivor" if abort else "prewarm")
        assert submitted.prompt_token_ids
        assert all(token_id == 0 for token_id in submitted.prompt_token_ids)
        assert len(replacement.add_request_calls) == 1
        assert running.value == 1
        assert orchestrator.output_async_queue.empty()


@pytest.mark.asyncio
async def test_abort_cleans_up_admission_already_inside_stage_client(admission: AdmissionHarness) -> None:
    await admission.membership.handle_register(0, 1)
    gate = asyncio.Event()
    admission.replacement.request_gates["waiting"] = gate
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("waiting", [1]))
        await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
        assert set(admission.pool.output_processor.requests) == {"waiting"}
        assert admission.pool.get_bound_replica_id("waiting") == 1
        assert admission.running.value == 1

        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["waiting"], rpc_id="cancel-client-submit")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.success
        assert admission.replacement.abort_calls == [["waiting"]]
        assert not admission.pool.output_processor.requests
        assert admission.pool.get_bound_replica_id("waiting") is None
        assert not admission.orchestrator.request_states
        assert admission.running.value == 0

        gate.set()
        admission.orchestrator.request_async_queue.put_nowait(submission("survivor", [2]))
        request = await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
        assert request.request_id == "survivor"
        assert set(admission.pool.output_processor.requests) == {"survivor"}
        assert admission.running.value == 1
        assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["survivor"]


@pytest.mark.asyncio
async def test_cancelled_dispatch_stays_cancelled_after_replica_recovery(admission: AdmissionHarness) -> None:
    async with request_handlers(admission.orchestrator):
        await start_waiting_request(admission)
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["waiting"], rpc_id="cancel-waiting")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.success
        admission.orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))
        await asyncio.wait_for(admission.membership.registered.wait(), timeout=1)
        # Allow several real pick() retry intervals after recovery. Checking
        # immediately can miss a cancelled request's late physical submission.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(admission.replacement.submissions.get(), timeout=0.05)
        assert not admission.pool.output_processor.requests
        assert not admission.orchestrator.request_states
        assert admission.running.value == 0
        assert admission.pool.get_bound_replica_id("waiting") is None


@pytest.mark.asyncio
async def test_independent_request_progresses_while_another_client_submission_waits(
    admission: AdmissionHarness,
) -> None:
    await admission.membership.handle_register(0, 1)
    admission.replacement.request_gates["slow"] = asyncio.Event()
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("slow", [1]))
        await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
        admission.orchestrator.request_async_queue.put_nowait(submission("fast", [2]))
        request = await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
        assert request.request_id == "fast"
        assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["fast"]
        assert set(admission.orchestrator.request_states) == {"slow", "fast"}
        assert admission.running.value == 2


@pytest.mark.asyncio
async def test_active_admission_limit_keeps_registration_and_abort_responsive(
    admission: AdmissionHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vllm_omni.engine.orchestrator._MAX_CONCURRENT_ADMISSIONS", 1)
    await admission.membership.handle_register(0, 1)
    admission.replacement.request_gates["slow"] = asyncio.Event()
    admission.membership.registered.clear()
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("slow", [1]))
        await asyncio.wait_for(admission.replacement.submission_started.get(), timeout=1)
        admission.orchestrator.request_async_queue.put_nowait(submission("queued", [2]))
        admission.orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))
        await asyncio.wait_for(admission.membership.registered.wait(), timeout=1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(admission.replacement.submissions.get(), timeout=0.05)
        assert "queued" not in admission.orchestrator.request_states

        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["slow"], rpc_id="free-active-slot")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.success
        request = await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
        assert request.request_id == "queued"
        assert set(admission.pool.output_processor.requests) == {"queued"}
        assert admission.running.value == 1


@pytest.mark.asyncio
async def test_admission_failure_propagates_and_stops_its_tasks(
    admission: AdmissionHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await admission.membership.handle_register(0, 1)

    async def fail_submission(*args, **kwargs) -> None:
        raise RuntimeError("client submission failed")

    monkeypatch.setattr(admission.replacement, "add_request_async", fail_submission)
    existing_tasks = set(asyncio.all_tasks())
    request_task = asyncio.create_task(admission.orchestrator._request_handler())
    admission_task = asyncio.create_task(admission.orchestrator._admission_handler())
    try:
        admission.orchestrator.request_async_queue.put_nowait(submission("failed", [1]))
        with pytest.raises(RuntimeError, match="client submission failed"):
            await asyncio.wait_for(admission_task, timeout=1)
    finally:
        request_task.cancel()
        admission_task.cancel()
        await asyncio.gather(request_task, admission_task, return_exceptions=True)
    assert admission.pool.get_bound_replica_id("failed") is None
    assert not admission.pool.output_processor.requests
    assert not set(asyncio.all_tasks()).difference(existing_tasks)


@pytest.mark.asyncio
async def test_shutdown_cancels_active_and_queued_admissions(admission: AdmissionHarness) -> None:
    existing_tasks = set(asyncio.all_tasks())
    request_task = asyncio.create_task(admission.orchestrator._request_handler())
    admission_task = asyncio.create_task(admission.orchestrator._admission_handler())
    try:
        await start_waiting_request(admission)
        admission.orchestrator.request_async_queue.put_nowait(submission("waiting", [2], update=True))
        admission.orchestrator.request_async_queue.put_nowait(ShutdownRequestMessage())
        await asyncio.wait_for(asyncio.gather(request_task, admission_task), timeout=1)
        await admission.membership.handle_register(0, 1)
        assert admission.replacement.add_request_calls == []
        assert not set(asyncio.all_tasks()).difference(existing_tasks)
    finally:
        request_task.cancel()
        admission_task.cancel()
        await asyncio.gather(request_task, admission_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["sleep", "wake_up"])
async def test_full_admission_queue_preserves_administrative_rpc_acknowledgment(
    admission: AdmissionHarness, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    monkeypatch.setattr("vllm_omni.engine.orchestrator._MAX_CONCURRENT_ADMISSIONS", 1)
    monkeypatch.setattr("vllm_omni.engine.orchestrator._MAX_PENDING_ADMISSIONS", 1)
    control_started = asyncio.Event()
    control_release = asyncio.Event()
    control_calls: list[str] = []

    async def engine_control() -> dict[str, str]:
        control_calls.append(method)
        control_started.set()
        await control_release.wait()
        return {"completed": method}

    for client in (admission.original, admission.replacement):
        monkeypatch.setattr(client, f"{method}_async", engine_control, raising=False)
    async with request_handlers(admission.orchestrator):
        await start_waiting_request(admission)
        admission.orchestrator.request_async_queue.put_nowait(submission("queued", [2]))
        admission.orchestrator.request_async_queue.put_nowait(
            CollectiveRPCRequestMessage(rpc_id="full-queue-rpc", method=method, args=(), kwargs={}, stage_ids=None)
        )
        admission.orchestrator.request_async_queue.put_nowait(
            AbortRequestMessage(request_ids=["unrelated"], rpc_id="queue-consumed")
        )
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, AbortResultMessage)
        assert result.rpc_id == "queue-consumed"
        assert result.success
        assert not control_calls

        admission.orchestrator.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))
        await asyncio.wait_for(control_started.wait(), timeout=1)
        assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["waiting", "queued"]
        assert control_calls == [method]
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=0.05)

        control_release.set()
        result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
        assert isinstance(result, CollectiveRPCResultMessage)
        assert result.rpc_id == "full-queue-rpc"
        assert result.results == [{"completed": method}, {"completed": method}]
        assert control_calls == [method, method]
        assert admission.orchestrator.output_async_queue.empty()


def block_stage_aborts(client: AdmissionStageClient, monkeypatch: pytest.MonkeyPatch) -> asyncio.Queue[asyncio.Event]:
    gates: asyncio.Queue[asyncio.Event] = asyncio.Queue()

    async def abort(request_ids: list[str]) -> None:
        await FakeStageClient.abort_requests_async(client, request_ids)
        gate = asyncio.Event()
        gates.put_nowait(gate)
        await gate.wait()

    monkeypatch.setattr(client, "abort_requests_async", abort)
    return gates


@pytest.mark.asyncio
async def test_external_cleanup_blocks_late_cfg_companion(admission: AdmissionHarness, monkeypatch: pytest.MonkeyPatch):
    await admission.membership.handle_register(0, 1)
    abort_gates = block_stage_aborts(admission.replacement, monkeypatch)
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("parent", [1]))
        await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
        cleanup = asyncio.create_task(admission.orchestrator._cleanup_request_ids(["parent"], abort=True))
        try:
            release_abort = await asyncio.wait_for(abort_gates.get(), timeout=1)
            admission.orchestrator.request_async_queue.put_nowait(companion())
            admission.orchestrator.request_async_queue.put_nowait(
                AbortRequestMessage(request_ids=["unrelated"], rpc_id="companion-consumed")
            )
            result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
            assert isinstance(result, AbortResultMessage)
            assert result.rpc_id == "companion-consumed"
            assert result.success
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(admission.replacement.submissions.get(), timeout=0.05)

            release_abort.set()
            await asyncio.wait_for(cleanup, timeout=1)
            admission.orchestrator.request_async_queue.put_nowait(
                CollectiveRPCRequestMessage(rpc_id="drained", method="ping", args=(), kwargs={}, stage_ids=None)
            )
            result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
            assert isinstance(result, CollectiveRPCResultMessage)
            assert result.rpc_id == "drained"
            assert [args[0].request_id for args in admission.replacement.add_request_calls] == ["parent"]
            assert not admission.orchestrator.request_states
            assert not admission.pool.output_processor.requests
            assert admission.pool.get_bound_replica_id("parent") is None
            assert admission.pool.get_bound_replica_id("negative") is None
            assert not admission.orchestrator._cfg_tracker.has_companions("parent")
            assert admission.orchestrator._cfg_tracker.get_parent_id("negative") is None
            assert admission.running.value == 0
        finally:
            cleanup.cancel()
            await asyncio.gather(cleanup, return_exceptions=True)


@pytest.mark.asyncio
async def test_parallel_cleanup_keeps_group_closed_until_both_aborts_finish(
    admission: AdmissionHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await admission.membership.handle_register(0, 1)
    abort_gates = block_stage_aborts(admission.replacement, monkeypatch)
    cleanups: list[asyncio.Task] = []
    async with request_handlers(admission.orchestrator):
        admission.orchestrator.request_async_queue.put_nowait(submission("parent", [1]))
        await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
        try:
            cleanups.append(asyncio.create_task(admission.orchestrator._cleanup_request_ids(["parent"], abort=True)))
            first_abort = await asyncio.wait_for(abort_gates.get(), timeout=1)
            cleanups.append(asyncio.create_task(admission.orchestrator._cleanup_request_ids(["parent"], abort=True)))
            second_abort = await asyncio.wait_for(abort_gates.get(), timeout=1)
            first_abort.set()
            await asyncio.wait_for(cleanups[0], timeout=1)
            assert not cleanups[1].done()

            # The first cleanup removed the old state. A replacement add with
            # the same id must wait, or the second cleanup can erase its state.
            admission.orchestrator.request_async_queue.put_nowait(submission("parent", [9]))
            admission.orchestrator.request_async_queue.put_nowait(
                AbortRequestMessage(request_ids=["unrelated"], rpc_id="replacement-consumed")
            )
            result = await asyncio.wait_for(admission.orchestrator.rpc_async_queue.get(), timeout=1)
            assert isinstance(result, AbortResultMessage)
            assert result.rpc_id == "replacement-consumed"
            assert result.success
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(admission.replacement.submissions.get(), timeout=0.05)
            assert "parent" not in admission.orchestrator.request_states
            assert admission.running.value == 0

            second_abort.set()
            await asyncio.wait_for(cleanups[1], timeout=1)
            request = await asyncio.wait_for(admission.replacement.submissions.get(), timeout=1)
            assert request.request_id == "parent"
            assert request.prompt_token_ids == [9]
            assert admission.pool.output_processor.requests["parent"] is request
            assert admission.pool.get_bound_replica_id("parent") == 1
            assert set(admission.orchestrator.request_states) == {"parent"}
            assert admission.running.value == 1
        finally:
            for cleanup in cleanups:
                cleanup.cancel()
            await asyncio.gather(*cleanups, return_exceptions=True)
