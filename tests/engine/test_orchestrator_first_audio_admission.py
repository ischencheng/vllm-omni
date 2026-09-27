# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""First-audio routing through real stage admission and request cleanup."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.distributed.omni_coordinator.load_balancer import RoundRobinBalancer
from vllm_omni.distributed.omni_coordinator.messages import ReplicaInfo, ReplicaList, ReplicaStatus
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs
from vllm_omni.engine.membership_controller import MembershipController
from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AbortResultMessage,
    ErrorMessage,
    OutputMessage,
    RegisterRemoteReplicaMessage,
    UnregisterRemoteReplicaMessage,
)
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor

from .test_orchestrator import FakeRunningCounter, _build_stage_pools, _sampling_params
from .test_orchestrator_admission import (
    AdmissionOutputProcessor,
    AdmissionStageClient,
    request_handlers,
    submission,
)
from .test_orchestrator_first_audio import _audio, _messages, _raw

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class AudioHub:
    def __init__(self, clients):
        self.clients = clients
        self.codec_queried = asyncio.Event()

    def get_replicas_for_stage(self, stage_id):
        if stage_id == 1:
            self.codec_queried.set()
        return ReplicaList(
            replicas=[
                ReplicaInfo(
                    input_addr=client.client_addresses["input_address"],
                    output_addr=f"tcp://stage-{stage_id}-output",
                    stage_id=stage_id,
                    status=ReplicaStatus.UP,
                    queue_length=0,
                    last_heartbeat=0,
                    registered_at=0,
                )
                for client in self.clients[stage_id]
            ],
            timestamp=0,
        )

    def close(self):
        pass


@dataclass
class AudioAdmissionHarness:
    obj: Orchestrator
    pools: list[StagePool]
    source: AdmissionStageClient
    spare_source: AdmissionStageClient
    codec: AdmissionStageClient
    later: AdmissionStageClient
    hub: AudioHub
    membership: MembershipController
    running: FakeRunningCounter
    processor: MultimodalOutputProcessor


@pytest.fixture
def audio_admission(monkeypatch: pytest.MonkeyPatch) -> AudioAdmissionHarness:
    source = AdmissionStageClient("tcp://source-input")
    spare_source = AdmissionStageClient("tcp://spare-source-input")
    source.final_output = spare_source.final_output = False
    old_codec = AdmissionStageClient("tcp://old-codec-input")
    codec = AdmissionStageClient("tcp://codec-input")
    old_codec.final_output_type = codec.final_output_type = "audio"
    later = AdmissionStageClient("tcp://later-input")
    later.final_output = False
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False, engine_core_output_type="audio")
    pools = _build_stage_pools(
        [[source, spare_source], [old_codec], [later]],
        output_processors=[AdmissionOutputProcessor(), processor, AdmissionOutputProcessor()],
    )
    # The codec is UP at the coordinator before its head-side client attaches.
    # Keeping the old slot preserves stage metadata while the real pick waits.
    hub = AudioHub([[source], [codec], [later]])
    monkeypatch.setattr("vllm_omni.engine.membership_controller.OmniCoordClientForHub", lambda _address: hub)

    def remote_factory(stage_id, replica_id):
        assert (stage_id, replica_id) == (1, 2)
        return codec

    membership = MembershipController(pools, "tcp://test-hub", RoundRobinBalancer, remote_factory)
    monkeypatch.setattr(pools[1], "DISPATCH_WAIT_TIMEOUT_S", 30.0)
    monkeypatch.setattr(pools[1], "DISPATCH_RETRY_INTERVAL_S", 0.005)
    monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", "0")
    running = FakeRunningCounter()
    obj = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=pools,
        membership_controller=membership,
        running_counter=running,
        async_chunk=True,
    )
    membership.install_unregister_handlers(
        output_queue=obj.output_async_queue,
        cleanup_callback=lambda ids: obj._cleanup_request_ids(ids, abort=True),
    )
    return AudioAdmissionHarness(
        obj=obj,
        pools=pools,
        source=source,
        spare_source=spare_source,
        codec=codec,
        later=later,
        hub=hub,
        membership=membership,
        running=running,
        processor=processor,
    )


def audio_request(request_id="r", *, kind=RequestOutputKind.DELTA, later_stage=False, tokens=None):
    message = submission(request_id, tokens or [1, 2])
    message.prompt.resumable = False
    message.final_stage_id = 2 if later_stage else 1
    message.final_output_stage_ids = [1]
    message.sampling_params_list.extend(_sampling_params() for _ in range(message.final_stage_id))
    message.sampling_params_list[1].output_kind = kind
    return message


def audio_raw(request_id="r", **kwargs):
    raw = _raw(**kwargs)
    raw.outputs[0].request_id = request_id
    return raw


def source_terminal(request_id="r"):
    return OmniEngineCoreOutputs(
        outputs=[OmniEngineCoreOutput(request_id=request_id, new_token_ids=[], finish_reason=FinishReason.STOP)]
    )


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=2)


@asynccontextmanager
async def audio_handlers(harness):
    async with request_handlers(harness.obj):
        output_task = asyncio.create_task(harness.obj._orchestration_output_handler())
        try:
            yield
        finally:
            output_task.cancel()
            await asyncio.gather(output_task)
            await harness.membership.drain_tasks(timeout=1)
            harness.membership.shutdown()


async def pending_first_audio(harness, message):
    harness.obj.request_async_queue.put_nowait(message)
    await asyncio.wait_for(harness.hub.codec_queried.wait(), timeout=2)
    state = harness.obj.request_states[message.request_id]
    assert harness.pools[1].get_bound_replica_id(message.request_id) is None
    harness.source.push_engine_core_outputs(audio_raw(message.request_id, first=True, samples=(10, 11)))
    await wait_until(lambda: state.pending_upstream_first_audio is not None)
    assert harness.obj.output_async_queue.empty()
    return state


async def attach_with_overtaking_terminal(harness, state):
    harness.codec.submit_gate.clear()
    harness.obj.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=1, replica_id=2))
    started = await asyncio.wait_for(harness.codec.submission_started.get(), timeout=2)
    assert started.request_id == state.request_id
    assert harness.pools[1].get_bound_replica_id(state.request_id) == 2
    assert state.request_id in harness.processor.request_states
    # The decoder can return chunks while add_request_async has not yet returned
    # to prewarm. Its terminal must stay behind the pending source PCM.
    harness.codec.push_engine_core_outputs(audio_raw(state.request_id, required=True, samples=(20, 21)))
    harness.codec.push_engine_core_outputs(audio_raw(state.request_id, terminal=True, samples=(30, 31)))
    harness.source.push_engine_core_outputs(source_terminal(state.request_id))
    await wait_until(lambda: len(state.pending_first_audio_outputs) == 2 and 0 in state.finished_stage_ids)
    assert harness.obj.output_async_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(RequestOutputKind))
async def test_registration_flush_preserves_terminal_order_output_kind_and_codec_owner(audio_admission, kind):
    h = audio_admission
    async with audio_handlers(h):
        state = await pending_first_audio(h, audio_request(kind=kind))
        await attach_with_overtaking_terminal(h, state)
        h.codec.submit_gate.set()
        await wait_until(lambda: "r" not in h.obj.request_states and not h.obj._active_admissions)
        messages = _messages(h.obj)
        assert all(isinstance(message, OutputMessage) for message in messages)
        assert all((message.stage_id, message.replica_id) == (1, 2) for message in messages)
        assert [_audio(message.engine_outputs).tolist() for message in messages] == {
            RequestOutputKind.DELTA: [[10, 11], [20, 21], [30, 31]],
            RequestOutputKind.CUMULATIVE: [[10, 11], [10, 11, 20, 21], [10, 11, 20, 21, 30, 31]],
            RequestOutputKind.FINAL_ONLY: [[10, 11, 20, 21, 30, 31]],
        }[kind]
        assert [message.finished for message in messages] == [False] * (len(messages) - 1) + [True]
        assert h.running.value == 0
        assert not h.processor.request_states
        assert not state.pending_first_audio_outputs
        assert state.pending_upstream_first_audio is None
        assert all(pool.get_bound_replica_id("r") is None for pool in h.pools)
        assert not h.obj._pending_admissions
        assert not h.obj._admission_cleanup_counts


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["abort", "unregister"])
async def test_pending_audio_cleanup_during_pick_does_not_block_healthy_request(audio_admission, cleanup):
    h = audio_admission
    async with audio_handlers(h):
        state = await pending_first_audio(h, audio_request())
        if cleanup == "abort":
            h.obj.request_async_queue.put_nowait(AbortRequestMessage(request_ids=["r"], rpc_id="cancel"))
            ack = await asyncio.wait_for(h.obj.rpc_async_queue.get(), timeout=2)
            assert isinstance(ack, AbortResultMessage) and ack.success
            assert ack.rpc_id == "cancel"
        else:
            h.hub.clients[0] = [h.spare_source]
            h.obj.request_async_queue.put_nowait(
                UnregisterRemoteReplicaMessage(stage_id=0, input_addr=h.source.client_addresses["input_address"])
            )
            error = await asyncio.wait_for(h.obj.output_async_queue.get(), timeout=2)
            assert isinstance(error, ErrorMessage) and error.request_id == "r"
            assert h.source.shutdown_calls == 1
        assert "r" not in h.obj.request_states
        assert h.running.value == 0
        assert all(pool.get_bound_replica_id("r") is None for pool in h.pools)
        assert h.codec.add_request_calls == []
        assert not h.processor.request_states
        # Releasing the old pending state after cancellation must remain a no-op.
        await h.obj._flush_upstream_first_audio(state)
        await h.obj._route_upstream_first_audio(0, 0, audio_raw(first=True, samples=(90, 91)))
        assert h.obj.output_async_queue.empty()

        h.obj.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=1, replica_id=2))
        h.obj.request_async_queue.put_nowait(audio_request("r"))
        submitted = await asyncio.wait_for(h.codec.submissions.get(), timeout=2)
        assert submitted.request_id == "r"
        replacement_state = h.obj.request_states["r"]
        processor_state = h.processor.request_states["r"]
        assert replacement_state is not state
        assert state.pending_upstream_first_audio is not None
        assert processor_state.mm_accumulated.is_empty
        # The old PCM is still retained on its original state object. Reusing
        # the ID must not let that stale flush write into the new processor.
        await h.obj._flush_upstream_first_audio(state)
        assert h.obj.request_states["r"] is replacement_state
        assert h.processor.request_states["r"] is processor_state
        assert processor_state.mm_accumulated.is_empty
        assert h.obj.output_async_queue.empty()
        source = h.spare_source if cleanup == "unregister" else h.source
        source.push_engine_core_outputs(audio_raw("r", first=True, samples=(40, 41)))
        source.push_engine_core_outputs(source_terminal("r"))
        h.codec.push_engine_core_outputs(audio_raw("r", required=True, terminal=True, samples=(50, 51)))
        await wait_until(lambda: "r" not in h.obj.request_states and not h.obj._active_admissions)
        messages = _messages(h.obj)
        assert [message.request_id for message in messages] == ["r", "r"]
        assert [_audio(message.engine_outputs).tolist() for message in messages] == [[40, 41], [50, 51]]
        assert messages[-1].finished
        assert [args[0].request_id for args in h.codec.add_request_calls] == ["r"]
        assert h.running.value == 0
        assert not h.processor.request_states
        assert not h.obj._pending_admissions
        assert not h.obj._admission_cleanup_counts


@pytest.mark.asyncio
async def test_prewarm_terminal_self_cleanup_allows_same_id_replacement(audio_admission):
    h = audio_admission
    async with audio_handlers(h):
        old_state = await pending_first_audio(h, audio_request(later_stage=True))
        await attach_with_overtaking_terminal(h, old_state)
        # Terminal routing cleans up from inside the admission task. It must
        # neither await itself nor submit the now-unneeded third stage.
        h.codec.submit_gate.set()
        await asyncio.wait_for(h.codec.submissions.get(), timeout=2)
        await wait_until(lambda: not h.obj.request_states and not h.obj._active_admissions)
        assert h.later.add_request_calls == []
        assert h.running.value == 0
        assert not h.processor.request_states
        assert not h.obj._admission_cleanup_counts
        assert not old_state.pending_first_audio_outputs
        old_messages = _messages(h.obj)
        assert [_audio(message.engine_outputs).tolist() for message in old_messages] == [[10, 11], [20, 21], [30, 31]]
        assert old_messages[-1].finished

        # Reuse the ID only after cleanup; queued old-group messages are
        # intentionally cancelled by cleanup, not treated as replacements.
        h.obj.request_async_queue.put_nowait(audio_request(tokens=[9]))
        await asyncio.wait_for(h.codec.submissions.get(), timeout=2)
        new_state = h.obj.request_states["r"]
        assert new_state is not old_state
        assert new_state.prompt.prompt_token_ids == [9]
        assert h.running.value == 1
        await h.obj._flush_upstream_first_audio(old_state)
        assert h.obj.output_async_queue.empty()
        assert h.obj.request_states["r"] is new_state

        h.source.push_engine_core_outputs(audio_raw(first=True, samples=(40, 41)))
        h.source.push_engine_core_outputs(source_terminal())
        h.codec.push_engine_core_outputs(audio_raw(required=True, terminal=True, samples=(50, 51)))
        await wait_until(lambda: not h.obj.request_states and not h.obj._active_admissions)
        messages = _messages(h.obj)
        assert [_audio(message.engine_outputs).tolist() for message in messages] == [[40, 41], [50, 51]]
        assert messages[-1].finished
        assert h.running.value == 0
        assert not h.processor.request_states
        assert all(pool.get_bound_replica_id("r") is None for pool in h.pools)
        assert not h.obj._pending_admissions
        assert not h.obj._admission_cleanup_counts
