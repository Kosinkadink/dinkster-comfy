from unittest.mock import Mock, call

import pytest

import dinkster_comfy.model_management as model_management


class NPUDevice:
    type = "npu"


class FakeStream:
    def __init__(self):
        self.waited_for = None

    def wait_stream(self, stream):
        self.waited_for = stream


def test_npu_current_stream(monkeypatch):
    device = NPUDevice()
    current_stream = object()
    npu = Mock()
    npu.current_stream.return_value = current_stream
    monkeypatch.setattr(model_management.torch, "npu", npu, raising=False)

    assert model_management.is_device_npu(device)
    assert model_management.current_stream(device) is current_stream
    npu.current_stream.assert_called_once_with(device)


def test_npu_offload_streams(monkeypatch):
    device = NPUDevice()
    current_stream = object()
    first_stream = FakeStream()
    second_stream = FakeStream()
    stream_context = object()
    npu = Mock()
    npu.current_stream.return_value = current_stream
    npu.Stream.side_effect = [first_stream, second_stream]
    npu.stream = stream_context

    monkeypatch.setattr(model_management.torch, "npu", npu, raising=False)
    monkeypatch.setattr(model_management.torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(model_management, "NUM_STREAMS", 2)
    monkeypatch.setattr(model_management, "STREAMS", {})
    monkeypatch.setattr(model_management, "stream_counters", {})

    assert model_management.get_offload_stream(device) is first_stream
    assert model_management.get_offload_stream(device) is second_stream
    assert model_management.get_offload_stream(device) is first_stream
    assert model_management.STREAMS[device] == [first_stream, second_stream]
    assert first_stream.as_context is stream_context
    assert second_stream.as_context is stream_context
    assert first_stream.waited_for is current_stream
    assert second_stream.waited_for is current_stream
    assert model_management.stream_counters[device] == 0
    assert npu.Stream.call_count == 2
    assert npu.Stream.call_args_list == [
        call(device=device, priority=0),
        call(device=device, priority=0),
    ]


def test_npu_offload_streams_disabled(monkeypatch):
    npu = Mock()
    monkeypatch.setattr(model_management.torch, "npu", npu, raising=False)
    monkeypatch.setattr(model_management, "NUM_STREAMS", 0)

    assert model_management.get_offload_stream(NPUDevice()) is None
    npu.Stream.assert_not_called()


def test_npu_synchronize(monkeypatch):
    npu = Mock()
    monkeypatch.setattr(model_management.torch, "npu", npu, raising=False)
    monkeypatch.setattr(model_management, "cpu_mode", lambda: False)
    monkeypatch.setattr(model_management, "is_intel_xpu", lambda: False)
    monkeypatch.setattr(model_management, "is_ascend_npu", lambda: True)

    model_management.synchronize()

    npu.synchronize.assert_called_once_with()


class FakeManagedModel:
    def __init__(self, name):
        self.model = type(name, (), {})()

    def is_dynamic(self):
        return False


class FakeLoadedModel:
    def __init__(self, name, device):
        self.model = FakeManagedModel(name)
        self.device = device
        self.currently_used = True
        self.unloaded = False

    def is_dead(self):
        return False

    def model_offloaded_memory(self):
        return 0

    def model_memory(self):
        return 4

    def model_unload(self, memory_to_free):
        self.unloaded = True
        return True


def test_model_manager_owns_loaded_models_and_unload_decisions(monkeypatch):
    first = model_management.ModelManager()
    second = model_management.ModelManager()
    loaded = FakeLoadedModel("FirstModel", "cpu")
    first._loaded_models.append(loaded)
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(model_management, "get_free_memory", lambda device: 0)

    unloaded = first.free_memory(1, "cpu")

    assert unloaded == [loaded]
    assert loaded.unloaded
    assert first.loaded_models() == []
    assert second.loaded_models() == []


def test_module_load_calls_delegate_to_process_model_manager(monkeypatch):
    class RecordingManager(model_management.ModelManager):
        def load_models_gpu(self, *args, **kwargs):
            self.call = args, kwargs
            return "loaded"

    manager = RecordingManager()
    monkeypatch.setattr(model_management, "_model_manager", manager)

    result = model_management.load_models_gpu(
        ["model"], memory_required=12, force_full_load=True
    )

    assert result == "loaded"
    assert manager.call == ((["model"], 12, False, None, True), {})


def test_model_manager_cannot_be_replaced_while_models_are_loaded(monkeypatch):
    manager = model_management.ModelManager()
    manager._loaded_models.append(FakeLoadedModel("LoadedModel", "cpu"))
    monkeypatch.setattr(model_management, "_model_manager", manager)

    with pytest.raises(RuntimeError, match="while models are loaded"):
        model_management.set_model_manager(model_management.ModelManager())
