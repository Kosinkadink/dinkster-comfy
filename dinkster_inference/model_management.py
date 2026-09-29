"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import psutil
import logging
from enum import Enum
from dinkster_inference.cli_args import args, PerformanceFeature
import threading
import torch
import sys
import platform
import weakref
import gc
import os
from contextlib import contextmanager, nullcontext
import dinkster_inference.memory_management
import dinkster_inference.system_memory
import dinkster_inference.utils
import dinkster_inference.quant_ops
import comfy_aimdo.host_buffer
import comfy_aimdo.vram_buffer
from dinkster_inference.internal_logging import detail

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dinkster_inference.model_patcher import ModelPatcher


class VRAMState(Enum):
    DISABLED = 0    #No vram present: no need to move models to vram
    NO_VRAM = 1     #Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      #No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0


# Training Related State
in_training = False
training_fp8_bwd = False


def get_supported_float8_types():
    float8_types = []
    try:
        float8_types.append(torch.float8_e4m3fn)
    except:
        pass
    try:
        float8_types.append(torch.float8_e4m3fnuz)
    except:
        pass
    try:
        float8_types.append(torch.float8_e5m2)
    except:
        pass
    try:
        float8_types.append(torch.float8_e5m2fnuz)
    except:
        pass
    try:
        float8_types.append(torch.float8_e8m0fnu)
    except:
        pass
    return float8_types

FLOAT8_TYPES = get_supported_float8_types()

xpu_available = False
torch_version = ""
try:
    torch_version = torch.version.__version__
    temp = torch_version.split(".")
    torch_version_numeric = (int(temp[0]), int(temp[1]))
except:
    pass

lowvram_available = True
if args.deterministic:
    logging.info("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
    logging.warning("WARNING: torch-directml barely works, is very slow, has not been updated in over 1 year and might be removed soon, please don't use it, there are better options.")
    import torch_directml
    directml_enabled = True
    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    logging.info("Using directml with device: {}".format(torch_directml.device_name(device_index)))
    # torch_directml.disable_tiled_resources(True)
    lowvram_available = False #TODO: need to find a way to get free memory in directml before this can be enabled by default.


try:
    _ = torch.xpu.device_count()
    xpu_available = torch.xpu.is_available()
except:
    xpu_available = False

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

try:
    import torch_npu  # noqa: F401
    _ = torch.npu.device_count()
    npu_available = torch.npu.is_available()
except:
    npu_available = False

try:
    import torch_mlu  # noqa: F401
    _ = torch.mlu.device_count()
    mlu_available = torch.mlu.is_available()
except:
    mlu_available = False

try:
    ixuca_available = hasattr(torch, "corex")
except:
    ixuca_available = False

if args.cpu or (
    cpu_state == CPUState.GPU
    and not directml_enabled
    and not torch.cuda.is_available()
    and not xpu_available
    and not npu_available
    and not mlu_available
    and not ixuca_available
):
    cpu_state = CPUState.CPU

def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False

def is_ascend_npu():
    global npu_available
    if npu_available:
        return True
    return False

def is_mlu():
    global mlu_available
    if mlu_available:
        return True
    return False

def is_ixuca():
    global ixuca_available
    if ixuca_available:
        return True
    return False

def is_wsl():
    version = platform.uname().release
    if version.endswith("-Microsoft"):
        return True
    elif version.endswith("microsoft-standard-WSL2"):
        return True
    return False

def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        elif is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        elif is_mlu():
            return torch.device("mlu", torch.mlu.current_device())
        else:
            return torch.device(torch.cuda.current_device())

def get_all_torch_devices(exclude_current=False):
    global cpu_state
    devices = []
    if cpu_state == CPUState.GPU:
        # NVIDIA + AMD/ROCm both expose their GPUs through torch.cuda.*;
        # without the AMD arm, single-GPU ROCm users get an empty list
        # which silently turns unload_all_models() into a no-op.
        if is_nvidia() or is_amd():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device("cuda", i))
        elif is_intel_xpu():
            for i in range(torch.xpu.device_count()):
                devices.append(torch.device("xpu", i))
        elif is_ascend_npu():
            for i in range(torch.npu.device_count()):
                devices.append(torch.device("npu", i))
        elif is_mlu():
            for i in range(torch.mlu.device_count()):
                devices.append(torch.device("mlu", i))
        else:
            # Fallback for unhandled GPU backends (e.g. DirectML): at least
            # report the current device so callers like unload_all_models()
            # do not silently no-op.
            devices.append(get_torch_device())
    else:
        devices.append(get_torch_device())
    if exclude_current:
        current = get_torch_device()
        if current in devices:
            devices.remove(current)
    return devices

def get_gpu_device_options():
    """Return list of device option strings for node widgets.

    Always includes "default" and "cpu". When multiple GPUs are present,
    adds "gpu:0", "gpu:1", etc. (vendor-agnostic labels).
    """
    options = ["default", "cpu"]
    devices = get_all_torch_devices()
    if len(devices) > 1:
        for i in range(len(devices)):
            options.append(f"gpu:{i}")
    return options

def get_gpu_device_options_no_cpu():
    """Variant of get_gpu_device_options that omits "cpu".

    Intended for components like the VAE selector where running on CPU
    is impractical and should not be offered as a choice.
    """
    return [o for o in get_gpu_device_options() if o != "cpu"]

def resolve_gpu_device_option(option: str):
    """Resolve a device option string to a torch.device.

    Returns None for "default" (let the caller use its normal default).
    Returns torch.device("cpu") for "cpu".
    For "gpu:N", returns the Nth torch device. Returns None if the
    index is out of range, the option string is malformed, or
    unrecognized (callers are expected to log their own context-rich
    message before falling back to the default device).
    """
    if option is None or option == "default":
        return None
    if option == "cpu":
        return torch.device("cpu")
    if option.startswith("gpu:"):
        try:
            idx = int(option[4:])
        except ValueError:
            return None
        devices = get_all_torch_devices()
        if 0 <= idx < len(devices):
            return devices[idx]
    return None

@contextmanager
def cuda_device_context(device):
    """Context manager that sets torch.cuda.current_device to match *device*.

    Used when running operations on a non-default CUDA device so that custom
    CUDA kernels (e.g. comfy_kitchen fp8 quantization) pick up the correct
    device index.  The previous device is restored on exit.

    No-op when *device* is not CUDA, has no explicit index, or already matches
    the current device.
    """
    prev = None
    if device.type == "cuda" and device.index is not None:
        prev = torch.cuda.current_device()
        if prev != device.index:
            torch.cuda.set_device(device)
        else:
            prev = None
    try:
        yield
    finally:
        if prev is not None:
            torch.cuda.set_device(prev)

def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = dinkster_inference.system_memory.virtual_memory_total()
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024 #TODO
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_reserved
            mem_total = mem_total_xpu
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_npu = torch.npu.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_npu
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_mlu = torch.mlu.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_mlu
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total

def mac_version():
    try:
        return tuple(int(n) for n in platform.mac_ver()[0].split("."))
    except:
        return None

total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = dinkster_inference.system_memory.virtual_memory_total() / (1024 * 1024)
logging.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))
cgroup_ram_limit = dinkster_inference.system_memory.cgroup_memory_limit()
if cgroup_ram_limit is not None:
    logging.info("RAM limited by cgroup to {:0.0f} MB (host has {:0.0f} MB)".format(cgroup_ram_limit / (1024 * 1024), psutil.virtual_memory().total / (1024 * 1024)))

try:
    logging.info("pytorch version: {}".format(torch_version))
    mac_ver = mac_version()
    if mac_ver is not None:
        logging.info("Mac Version {}".format(mac_ver))
except:
    pass

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

try:
    ACCELERATOR_ERROR = torch.AcceleratorError
except AttributeError:
    ACCELERATOR_ERROR = RuntimeError

def is_oom(e):
    if isinstance(e, OOM_EXCEPTION):
        return True
    if isinstance(e, ACCELERATOR_ERROR) and (getattr(e, 'error_code', None) == 2 or "out of memory" in str(e).lower()):
        discard_cuda_async_error()
        return True
    return False

def raise_non_oom(e):
    if not is_oom(e):
        raise e

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops
        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
            logging.info("xformers version: {}".format(XFORMERS_VERSION))
            if XFORMERS_VERSION.startswith("0.0.18"):
                logging.warning("\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
                logging.warning("Please downgrade or upgrade xformers to a different version.\n")
                XFORMERS_ENABLED_VAE = False
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False

def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False

def is_amd():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.hip:
            return True
    return False

def amd_min_version(device=None, min_rdna_version=0):
    if not is_amd():
        return False

    if is_device_cpu(device):
        return False

    arch = torch.cuda.get_device_properties(device).gcnArchName
    if arch.startswith('gfx') and len(arch) == 7:
        try:
            cmp_rdna_version = int(arch[4]) + 2
        except:
            cmp_rdna_version = 0
        if cmp_rdna_version >= min_rdna_version:
            return True

    return False

MIN_WEIGHT_MEMORY_RATIO = 0.4
if is_nvidia():
    MIN_WEIGHT_MEMORY_RATIO = 0.0

ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

try:
    if is_nvidia():
        if torch_version_numeric[0] >= 2:
            if ENABLE_PYTORCH_ATTENTION == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
                ENABLE_PYTORCH_ATTENTION = True
    if is_intel_xpu() or is_ascend_npu() or is_mlu() or is_ixuca():
        if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
            ENABLE_PYTORCH_ATTENTION = True
except:
    pass


SUPPORT_FP8_OPS = args.supports_fp8_compute

AMD_RDNA2_AND_OLDER_ARCH = ["gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036", "gfx1010", "gfx1011", "gfx1012", "gfx906", "gfx900", "gfx803"]
AMD_ENABLE_MIOPEN_ENV = 'COMFYUI_ENABLE_MIOPEN'

try:
    if is_amd():
        arch = torch.cuda.get_device_properties(get_torch_device()).gcnArchName.split(':')[0]
        if not (any((a in arch) for a in AMD_RDNA2_AND_OLDER_ARCH)):
            if os.getenv(AMD_ENABLE_MIOPEN_ENV) != '1':
                torch.backends.cudnn.enabled = False  # Seems to improve things a lot on AMD
                logging.info("Set: torch.backends.cudnn.enabled = False for better AMD performance.")

        try:
            rocm_version = tuple(map(int, str(torch.version.hip).split(".")[:2]))
        except:
            rocm_version = (6, -1)

        def aotriton_supported():
            """Whether pytorch reports flash attention as usable on this gpu.

            can_use_flash_attention() evaluates runtime eligibility for the given
            parameters; on a ROCm build that includes checking the gpu arch against the
            arches AOTriton was built for. Querying it avoids assuming where the kernel
            images live inside the torch install. The probe tensor is shaped and
            typed to pass the unrelated SDPA checks, so False means no hardware support
            rather than a rejected shape.

            It answers True on a supported arch whose kernel image was never shipped,
            and that only fails at launch, without raising. So run one attention
            through the flash backend and force the pending error check.
            """
            try:
                device = get_torch_device()
                if not torch.backends.cuda.is_flash_attention_available():  # not built with flash attention
                    return False
                q = torch.zeros((1, 1, 8, 64), dtype=torch.float16, device=device)
                params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, False, False)
                if not torch.backends.cuda.can_use_flash_attention(params, False):
                    return False
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
                torch.zeros(1, device=device).add_(1).item()  # raises if the launch above failed
                return True
            except Exception as e:
                logging.warning("Could not run flash attention, disabling it: {}".format(e))
                return False

        logging.info("AMD arch: {}".format(arch))
        logging.info("ROCm version: {}".format(rocm_version))
        if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
            if aotriton_supported():  # AMD efficient attention implementation depends on aotriton.
                if torch_version_numeric >= (2, 7):  # works on 2.6 but doesn't actually seem to improve much
                    if any((a in arch) for a in ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1170", "gfx1171"]):  # TODO: more arches, TODO: gfx950
                        ENABLE_PYTORCH_ATTENTION = True
                if rocm_version >= (7, 0):
                    if any((a in arch) for a in ["gfx1200", "gfx1201"]):
                        ENABLE_PYTORCH_ATTENTION = True
        if torch_version_numeric >= (2, 7) and rocm_version >= (6, 4):
            if any((a in arch) for a in ["gfx1200", "gfx1201", "gfx950", "gfx1170", "gfx1171"]):  # TODO: more arches, "gfx942" gives error on pytorch nightly 2.10 1013 rocm7.0
                SUPPORT_FP8_OPS = True

except:
    pass


if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


PRIORITIZE_FP16 = False  # TODO: remove and replace with something that shows exactly which dtype is faster than the other
try:
    if (is_nvidia() or is_amd()) and PerformanceFeature.Fp16Accumulation in args.fast:
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        PRIORITIZE_FP16 = True  # TODO: limit to cards where it actually boosts performance
        logging.info("Enabled fp16 accumulation.")
except:
    pass


def set_cudnn_benchmark():
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = PerformanceFeature.AutoTune in args.fast

try:
    if torch_version_numeric >= (2, 5):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
except:
    logging.warning("Warning, could not set allow_fp16_bf16_reduction_math_sdp")

if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
if args.force_fp32:
    logging.info("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to


if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")

def get_torch_device_name(device):
    if hasattr(device, 'type'):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
        elif device.type == "xpu":
            return "{} {}".format(device, torch.xpu.get_device_name(device))
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    elif is_ascend_npu():
        return "{} {}".format(device, torch.npu.get_device_name(device))
    elif is_mlu():
        return "{} {}".format(device, torch.mlu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))

try:
    logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logging.warning("Could not pick default device.")
try:
    for device in get_all_torch_devices(exclude_current=True):
        logging.info("Device: {}".format(get_torch_device_name(device)))
except:
    pass

PIN_PRESSURE_HYSTERESIS = 256 * 1024 * 1024

#Freeing registerables on pressure does imply a GPU sync, so go big on
#the hysteresis so each expensive sync gives us back a good chunk.
REGISTERABLE_PIN_HYSTERESIS = 2048 * 1024 * 1024
WINDOWS_PIN_EVICTION_SWAP_PERCENT = 5.0
WINDOWS_PIN_EVICTION_EMERGENCY_AVAILABLE = 512 * 1024 ** 2

def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nbytes
    return module_mem

def mark_mmap_dirty(storage):
    mmap_refs = getattr(storage, "_comfy_tensor_mmap_refs", None)
    if mmap_refs is not None:
        get_model_manager().dirty_mmaps.add(mmap_refs[0])

PIN_SUBSETS = [ "weights", "patches" ]
LOADED_PIN_SUBSETS = [ "weights-loaded", "patches-loaded" ]
FAST_PIN_SUBSETS = [ "weights-fast", "patches-fast" ]

def models_for_pin_eviction(active, current_prompt=None):
    return get_model_manager().models_for_pin_eviction(active, current_prompt)


def _models_for_pin_eviction(manager, active, current_prompt=None):
    for loaded_model in manager.loaded_model_records():
        model = loaded_model.model
        if model is None or not model.is_dynamic():
            continue
        pin_state = model.model.dynamic_pins[model.load_device]
        if ((active is None or pin_state["active"] == active) and
            (current_prompt is None or pin_state["current_prompt"] == current_prompt)):
            yield model

def free_model_pins(size, subsets, current_prompt, active, registrations=False):
    return get_model_manager().free_model_pins(
        size, subsets, current_prompt, active, registrations
    )


def _free_model_pins(manager, size, subsets, current_prompt, active, registrations=False):
    freed_total = 0
    for model in manager.models_for_pin_eviction(
        active, current_prompt=current_prompt
    ):
        if size <= 0:
            return freed_total
        if registrations:
            freed = model.unregister_inactive_pins(size, subsets=subsets)
        else:
            freed = model.partially_unload_ram(size, subsets=subsets)
        if freed > 0:
            detail(
                "Pin eviction: model=%s subsets=%s workflow=%s active=%s action=%s freed_mb=%.1f",
                model.model.__class__.__name__, subsets, current_prompt, active,
                "unregister" if registrations else "destroy", freed / (1024 ** 2),
            )
        freed_total += freed
        size -= freed
    return freed_total

def pin_eviction_tiers(loaded, evict_active):
    tiers = [
        (FAST_PIN_SUBSETS, False, False),
        (PIN_SUBSETS, False, None),
        (LOADED_PIN_SUBSETS, False, None),
        (FAST_PIN_SUBSETS, True, False),
        (LOADED_PIN_SUBSETS, True, None),
    ]
    if not loaded:
        tiers.append((PIN_SUBSETS, True, False))
        if evict_active:
            tiers.extend([
                (FAST_PIN_SUBSETS, False, True),
                (FAST_PIN_SUBSETS, True, True),
                (PIN_SUBSETS, True, True),
            ])
    return tiers

def registration_eviction_tiers(evict_active):
    subsets = PIN_SUBSETS + LOADED_PIN_SUBSETS
    tiers = [
        (FAST_PIN_SUBSETS, False, False, False),
        (subsets, False, False, True),
        (FAST_PIN_SUBSETS, True, False, False),
        (subsets, True, False, True),
    ]
    if evict_active:
        tiers.extend([
            (FAST_PIN_SUBSETS, False, True, False),
            (subsets, False, True, True),
            (FAST_PIN_SUBSETS, True, True, False),
            (subsets, True, True, True),
        ])
    return tiers

def free_pins(size, evict_active=False, loaded=False):
    return get_model_manager().free_pins(size, evict_active, loaded)


def _free_pins(manager, size, evict_active=False, loaded=False):
    freed = 0
    for subsets, current_prompt, active in pin_eviction_tiers(loaded, evict_active):
        freed += manager.free_model_pins(
            size - freed, subsets, current_prompt, active
        )
    return freed

def should_free_pins_for_ram_pressure(shortfall):
    if shortfall <= 0:
        return False
    if not WINDOWS:
        return True
    if dinkster_inference.system_memory.virtual_memory_available() < WINDOWS_PIN_EVICTION_EMERGENCY_AVAILABLE:
        return True
    try:
        return psutil.swap_memory().percent >= WINDOWS_PIN_EVICTION_SWAP_PERCENT
    except RuntimeError as err:
        logging.warning("Could not read Windows swap usage; falling back to RAM-pressure pin eviction: %s", err)
        return True

def ensure_pin_budget(size, evict_active=False, loaded=False):
    return get_model_manager().ensure_pin_budget(size, evict_active, loaded)


def _ensure_pin_budget(manager, size, evict_active=False, loaded=False):
    if args.high_ram:
        return True
    shortfall = size + max(dinkster_inference.memory_management.RAM_CACHE_HEADROOM / 2, 2048 * 1024 ** 2) - dinkster_inference.system_memory.virtual_memory_available()
    if shortfall <= 0:
        return True

    to_free = shortfall + PIN_PRESSURE_HYSTERESIS
    return manager.free_pins(
        to_free, evict_active=evict_active, loaded=loaded
    ) >= shortfall

def free_registrations(shortfall, evict_active=True):
    return get_model_manager().free_registrations(shortfall, evict_active)


def _free_registrations(manager, shortfall, evict_active=True):
    if manager.max_pinned_memory <= 0:
        return False
    if shortfall <= 0:
        return True

    shortfall += REGISTERABLE_PIN_HYSTERESIS
    for subsets, current_prompt, active, registrations in registration_eviction_tiers(evict_active):
        shortfall -= manager.free_model_pins(
            shortfall,
            subsets,
            current_prompt,
            active,
            registrations=registrations,
        )
    return shortfall <= REGISTERABLE_PIN_HYSTERESIS

def ensure_pin_registerable(size, evict_active=True):
    return get_model_manager().ensure_pin_registerable(size, evict_active)


def _ensure_pin_registerable(manager, size, evict_active=True):
    return manager.free_registrations(
        manager.total_pinned_memory + size - manager.max_pinned_memory,
        evict_active=evict_active,
    )

class LoadedModel:
    def __init__(self, model: ModelPatcher, manager=None):
        self.manager = manager if manager is not None else get_model_manager()
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model: ModelPatcher):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)
            self._patcher_finalizer.atexit = False

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)
            self.device = model.load_device

    @property
    def model(self):
        return self._model()

    def model_memory(self):
        return self.model.model_size()

    def model_loaded_memory(self):
        return self.model.loaded_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        # if self.model.loaded_size() > 0:
        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)

        real_model = self.model.model


        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, self.manager.cleanup_models)
        self.model_finalizer.atexit = False
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model() is not None and self.model is None


class ModelManager:
    def __init__(self):
        self._loaded_models: list[LoadedModel] = []
        self.dirty_mmaps = set()
        self.streams = {}
        self.stream_counters = {}
        self.stream_cast_buffers = {}
        self.largest_casted_weight = (None, 0)
        self.stream_aimdo_cast_buffers = {}
        self.largest_aimdo_casted_weight = (None, 0)
        self.cross_step_state = weakref.WeakSet()
        self.pinned_memory = {}
        self.total_pinned_memory = 0
        self.max_pinned_memory = -1
        self.lora_compute_dtypes = {}
        self.interrupt_processing_mutex = threading.RLock()
        self.interrupt_processing = False

    def loaded_model_records(self):
        return tuple(self._loaded_models)

    def models_for_pin_eviction(self, active, current_prompt=None):
        return _models_for_pin_eviction(self, active, current_prompt)

    def free_model_pins(
        self, size, subsets, current_prompt, active, registrations=False
    ):
        return _free_model_pins(
            self, size, subsets, current_prompt, active, registrations
        )

    def free_pins(self, size, evict_active=False, loaded=False):
        return _free_pins(self, size, evict_active, loaded)

    def ensure_pin_budget(self, size, evict_active=False, loaded=False):
        return _ensure_pin_budget(self, size, evict_active, loaded)

    def free_registrations(self, shortfall, evict_active=True):
        return _free_registrations(self, shortfall, evict_active)

    def ensure_pin_registerable(self, size, evict_active=True):
        return _ensure_pin_registerable(self, size, evict_active)

    def add_pinned_memory(self, size):
        self.total_pinned_memory += size

    def release_pinned_memory(self, size):
        self.total_pinned_memory = max(0, self.total_pinned_memory - size)

    def free_memory(
        self,
        memory_required,
        device,
        keep_loaded=(),
        for_dynamic=False,
        pins_required=0,
        ram_required=0,
    ):
        return _free_memory(
            self,
            memory_required,
            device,
            keep_loaded,
            for_dynamic,
            pins_required,
            ram_required,
        )

    def load_models_gpu(
        self,
        models,
        memory_required=0,
        force_patch_weights=False,
        minimum_memory_required=None,
        force_full_load=False,
    ):
        return _load_models_gpu(
            self,
            models,
            memory_required,
            force_patch_weights,
            minimum_memory_required,
            force_full_load,
        )

    def loaded_models(self, only_currently_used=False):
        return _loaded_models(self, only_currently_used)

    def cleanup_models_gc(self):
        return _cleanup_models_gc(self)

    def cleanup_models(self):
        return _cleanup_models(self)

    def unload_all_models(self):
        for device in get_all_torch_devices():
            self.free_memory(1e30, device)

    def unload_model_and_clones(
        self, model, unload_additional_models=True, all_devices=False
    ):
        return _unload_model_and_clones(
            self, model, unload_additional_models, all_devices
        )


_model_manager = ModelManager()


def get_model_manager():
    return _model_manager


def set_model_manager(manager):
    global _model_manager
    if not isinstance(manager, ModelManager):
        raise TypeError("model manager must be a ModelManager")
    if _model_manager.loaded_model_records():
        raise RuntimeError("cannot replace the model manager while models are loaded")
    manager.max_pinned_memory = _model_manager.max_pinned_memory
    _model_manager = manager


def use_more_memory(extra_memory, loaded_models, device):
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_vram(extra_memory)
            if extra_memory <= 0:
                break

def offloaded_memory(loaded_models, device):
    offloaded_mem = 0
    for m in loaded_models:
        if m.device == device:
            offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem

WINDOWS = any(platform.win32_ver())

EXTRA_RESERVED_VRAM = 400 * 1024 * 1024
if WINDOWS:
    EXTRA_RESERVED_VRAM = 600 * 1024 * 1024 #Windows is higher because of the shared vram issue
    if total_vram > (15 * 1024):  # more extra reserved vram on 16GB+ cards
        EXTRA_RESERVED_VRAM += 100 * 1024 * 1024

if args.reserve_vram is not None:
    EXTRA_RESERVED_VRAM = args.reserve_vram * 1024 * 1024 * 1024
    logging.debug("Reserving {}MB vram for other applications.".format(EXTRA_RESERVED_VRAM / (1024 * 1024)))

def extra_reserved_memory():
    return EXTRA_RESERVED_VRAM

def minimum_inference_memory():
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()

def free_memory(memory_required, device, keep_loaded=(), for_dynamic=False, pins_required=0, ram_required=0):
    return get_model_manager().free_memory(
        memory_required,
        device,
        keep_loaded,
        for_dynamic,
        pins_required,
        ram_required,
    )


def _free_memory(manager, memory_required, device, keep_loaded=(), for_dynamic=False, pins_required=0, ram_required=0):
    manager.cleanup_models_gc()
    if not for_dynamic:
        detail("Non dynamic memory free called! memory_required=%s pins_required=%s ram_required=%s", memory_required, pins_required, ram_required)
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    loaded_models = manager._loaded_models
    for i in range(len(loaded_models) -1, -1, -1):
        shift_model = loaded_models[i]
        if device is None or shift_model.device == device:
            if shift_model not in keep_loaded and not shift_model.is_dead():
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    can_unload_sorted = sorted(can_unload)
    for x in can_unload_sorted:
        i = x[-1]
        memory_to_free = 1e32
        if not DISABLE_SMART_MEMORY or device is None:
            memory_to_free = 0 if device is None else memory_required - get_free_memory(device)
            if loaded_models[i].model.is_dynamic() and for_dynamic:
                #don't actually unload dynamic models for the sake of other dynamic models
                #as that works on-demand.
                memory_required -= loaded_models[i].model.loaded_size()
                memory_to_free = 0
        if memory_to_free > 0 and loaded_models[i].model_unload(memory_to_free):
            logging.debug(f"Unloading {loaded_models[i].model.model.__class__.__name__}")
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(loaded_models.pop(i))

    if not for_dynamic and pins_required > 0:
        manager.ensure_pin_budget(pins_required)
        manager.ensure_pin_registerable(pins_required)

    if len(unloaded_model) > 0:
        soft_empty_cache()
    elif device is not None:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()
    return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
    return get_model_manager().load_models_gpu(
        models,
        memory_required,
        force_patch_weights,
        minimum_memory_required,
        force_full_load,
    )


def _load_models_gpu(manager, models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
    manager.cleanup_models_gc()

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    # Order-preserving dedup. A plain set() would randomize iteration order across runs
    models_temp = {}
    for m in models:
        models_temp[m] = None
        for mm in m.model_patches_models():
            models_temp[mm] = None

    models = list(models_temp)
    models.reverse()

    models_to_load = []
    loaded_models = manager._loaded_models

    free_for_dynamic=True
    for x in models:
        if not x.is_dynamic():
            free_for_dynamic = False
        loaded_model = LoadedModel(x, manager)
        try:
            loaded_model_index = loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = loaded_models[loaded_model_index]
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            if hasattr(x, "model"):
                logging.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(loaded_models)):
            if loaded_model.model.is_clone(loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            model_to_unload = loaded_models.pop(i)
            model_to_unload.model.detach(unpatch_all=False)
            model_to_unload.model_finalizer.detach()

    total_memory_required = {}
    total_pins_required = {}
    for loaded_model in models_to_load:
        device = loaded_model.device
        total_memory_required[device] = total_memory_required.get(device, 0) + loaded_model.model_memory_required(device)
        if not loaded_model.model.is_dynamic():
            total_pins_required[device] = total_pins_required.get(device, 0) + loaded_model.model_memory()

    for device in total_memory_required:
        if device != torch.device("cpu"):
            manager.free_memory(total_memory_required[device] * 1.1 + extra_mem,
                                device,
                                for_dynamic=free_for_dynamic,
                                pins_required=total_pins_required.get(device, 0))

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_mem = get_free_memory(device)
            if free_mem < minimum_memory_required:
                models_l = manager.free_memory(minimum_memory_required, device, for_dynamic=free_for_dynamic)
                logging.info("{} models unloaded.".format(len(models_l)))

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM) and not force_full_load:
            loaded_memory = loaded_model.model_loaded_memory()
            current_free_mem = get_free_memory(torch_dev) + loaded_memory

            lowvram_model_memory = max(0, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
            lowvram_model_memory = lowvram_model_memory - loaded_memory

            if lowvram_model_memory == 0:
                lowvram_model_memory = 0.1

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 0.1

        loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        vram_used = 0 if is_device_cpu(torch_dev) else loaded_model.model_loaded_memory()
        ram_used = model.loaded_ram_size() if model.is_dynamic() else loaded_model.model_memory() - vram_used
        detail("Model loaded: patcher=%s model=%s ram_mb=%.1f vram_mb=%.1f", model.__class__.__name__, model.model.__class__.__name__, ram_used / (1024 ** 2), vram_used / (1024 ** 2))
        loaded_models.insert(0, loaded_model)
    return

def load_model_gpu(model):
    return load_models_gpu([model])

def loaded_models(only_currently_used=False):
    return get_model_manager().loaded_models(only_currently_used)


def _loaded_models(manager, only_currently_used=False):
    output = []
    for m in manager.loaded_model_records():
        if only_currently_used:
            if not m.currently_used:
                continue

        output.append(m.model)
    return output


def cleanup_models_gc():
    return get_model_manager().cleanup_models_gc()


def _cleanup_models_gc(manager):
    do_gc = False

    for cur in manager.loaded_model_records():
        if cur.is_dead():
            logging.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
            do_gc = True
            break

    if do_gc:
        gc.collect()
        soft_empty_cache()

        for cur in manager.loaded_model_records():
            if cur.is_dead():
                logging.warning("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(cur.real_model().__class__.__name__))


def archive_model_dtypes(model):
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            setattr(module, f"{param_name}_comfy_model_dtype", param.dtype)
        for buf_name, buf in module.named_buffers(recurse=False):
            setattr(module, f"{buf_name}_comfy_model_dtype", buf.dtype)


def cleanup_models():
    return get_model_manager().cleanup_models()


def _cleanup_models(manager):
    to_delete = []
    loaded_models = manager._loaded_models
    for i in range(len(loaded_models)):
        if loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = loaded_models.pop(i)
        del x

def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except: #Old pytorch doesn't have .itemsize
            pass
    return dtype_size

def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
    cpu_dev = torch.device("cpu")
    if dinkster_inference.memory_management.aimdo_enabled:
        return cpu_dev

    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.SHARED:
        return torch_dev

    if DISABLE_SMART_MEMORY or vram_state == VRAMState.NO_VRAM:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev

def maximum_vram_for_weights(device=None):
    return (get_total_memory(device) * 0.88 - minimum_inference_memory())

def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32], weight_dtype=None):
    if model_params < 0:
        model_params = 1000000000000000000000
    if args.fp32_unet:
        return torch.float32
    if args.fp64_unet:
        return torch.float64
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp16_unet:
        return torch.float16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return torch.float8_e5m2
    if args.fp8_e8m0fnu_unet:
        return torch.float8_e8m0fnu

    fp8_dtype = None
    if weight_dtype in FLOAT8_TYPES:
        fp8_dtype = weight_dtype

    if fp8_dtype is not None:
        if supports_fp8_compute(device): #if fp8 compute is supported the casting is most likely not expensive
            return fp8_dtype

        free_model_memory = maximum_vram_for_weights(device)
        if model_params * 2 > free_model_memory:
            return fp8_dtype

    if PRIORITIZE_FP16 or weight_dtype == torch.float16:
        if torch.float16 in supported_dtypes and should_use_fp16(device=device, model_params=model_params):
            return torch.float16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params, manual_cast=True):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    return torch.float32

# None means no manual cast
def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if weight_dtype == torch.float32 or weight_dtype == torch.float64:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)
    if PRIORITIZE_FP16 and fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    for dt in supported_dtypes:
        if dt == torch.float16 and fp16_supported:
            return torch.float16
        if dt == torch.bfloat16 and bf16_supported:
            return torch.bfloat16

    return torch.float32

def text_encoder_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def text_encoder_device():
    if args.gpu_only:
        return get_torch_device()
    if dinkster_inference.memory_management.aimdo_enabled:
        return get_torch_device()
    elif vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM):
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")

def text_encoder_initial_device(load_device, offload_device, model_size=0):
    if dinkster_inference.memory_management.aimdo_enabled:
        return offload_device

    if load_device == offload_device or model_size <= 1024 * 1024 * 1024:
        return offload_device

    if is_device_mps(load_device):
        return load_device

    mem_l = get_free_memory(load_device)
    mem_o = get_free_memory(offload_device)
    if mem_l > (mem_o * 0.5) and model_size * 1.2 < mem_l:
        return load_device
    else:
        return offload_device

def text_encoder_dtype(device=None):
    if args.fp8_e4m3fn_text_enc:
        return torch.float8_e4m3fn
    elif args.fp8_e5m2_text_enc:
        return torch.float8_e5m2
    elif args.fp16_text_enc:
        return torch.float16
    elif args.bf16_text_enc:
        return torch.bfloat16
    elif args.fp32_text_enc:
        return torch.float32

    if is_device_cpu(device):
        return torch.float16

    return torch.float16


def intermediate_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def intermediate_dtype():
    if args.fp16_intermediates:
        return torch.float16
    else:
        return torch.float32

def vae_device():
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()

def vae_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def vae_dtype(device=None, allowed_dtypes=[]):
    if args.fp16_vae:
        return torch.float16
    elif args.bf16_vae:
        return torch.bfloat16
    elif args.fp32_vae:
        return torch.float32

    for d in allowed_dtypes:
        if d == torch.float16 and should_use_fp16(device):
            return d

        if d == torch.bfloat16 and should_use_bf16(device):
            return d

    return torch.float32

def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"

def supports_dtype(device, dtype): #TODO
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False

def supports_cast(device, dtype): #TODO
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled: #TODO: test this
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True
    return False

def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype

def device_supports_non_blocking(device):
    if args.force_non_blocking:
        return True
    if is_device_mps(device):
        return False #pytorch bug? mps doesn't support non blocking
    if is_intel_xpu(): #xpu does support non blocking but it is slower on iGPUs for some reason so disable by default until situation changes
        return False
    if args.deterministic: #TODO: figure out why deterministic breaks non blocking from gpu to cpu (previews)
        return False
    if directml_enabled:
        return False
    return True

def force_channels_last():
    if args.force_channels_last:
        return True

    #TODO
    return False


NUM_STREAMS = 0
if args.async_offload is not None:
    NUM_STREAMS = args.async_offload
else:
    #  Enable by default on Nvidia and AMD
    if is_nvidia() or is_amd():
        NUM_STREAMS = 2

if args.disable_async_offload:
    NUM_STREAMS = 0

if NUM_STREAMS > 0:
    logging.info("Using async weight offloading with {} streams".format(NUM_STREAMS))

def current_stream(device):
    if device is None:
        return None
    if is_device_cuda(device):
        return torch.cuda.current_stream()
    elif is_device_xpu(device):
        return torch.xpu.current_stream()
    elif is_device_npu(device):
        return torch.npu.current_stream(device)
    else:
        return None

DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = 16 * 1024 ** 3

# NOTE: devs/agents: this is temporary and will be removed in a future dinkster_inference. Not supported for custom node use.
def _register_cross_step(module):
    get_model_manager().cross_step_state.add(module)

def get_cast_buffer(offload_stream, device, size, ref):
    manager = get_model_manager()

    if offload_stream is not None:
        wf_context = offload_stream
        if hasattr(wf_context, "as_context"):
            wf_context = wf_context.as_context(offload_stream)
    else:
        wf_context = nullcontext()

    cast_buffer = manager.stream_cast_buffers.get(offload_stream, None)
    if cast_buffer is None or cast_buffer.numel() < size:
        if ref is manager.largest_casted_weight[0]:
            #If there is one giant weight we do not want both streams to
            #allocate a buffer for it. It's up to the caster to get the other
            #offload stream in this corner case
            return None
        if cast_buffer is not None and cast_buffer.numel() > 50 * (1024 ** 2):
            #I want my wrongly sized 50MB+ of VRAM back from the caching allocator right now
            synchronize()
            del manager.stream_cast_buffers[offload_stream]
            del cast_buffer
            soft_empty_cache()
        with wf_context:
            cast_buffer = torch.empty((size), dtype=torch.int8, device=device)
            manager.stream_cast_buffers[offload_stream] = cast_buffer

        if size > manager.largest_casted_weight[1]:
            manager.largest_casted_weight = (ref, size)

    return cast_buffer

def get_aimdo_cast_buffer(offload_stream, device):
    manager = get_model_manager()
    cast_buffer = manager.stream_aimdo_cast_buffers.get(offload_stream, None)
    if cast_buffer is None:
        cast_buffer = comfy_aimdo.vram_buffer.VRAMBuffer(DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE, device.index)
        manager.stream_aimdo_cast_buffers[offload_stream] = cast_buffer
    return cast_buffer

def reset_cast_buffers():
    manager = get_model_manager()
    manager.largest_casted_weight = (None, 0)
    manager.largest_aimdo_casted_weight = (None, 0)
    for offload_stream in set(manager.stream_cast_buffers) | set(manager.stream_aimdo_cast_buffers):
        if offload_stream is not None:
            offload_stream.synchronize()
    synchronize()

    for mmap_obj in manager.dirty_mmaps:
        mmap_obj.bounce()
    manager.dirty_mmaps.clear()

    for module in manager.cross_step_state:
        del module._comfy_cross_step_state
    manager.cross_step_state.clear()

    for loaded_model in manager.loaded_model_records():
        model = loaded_model.model
        if model is not None and model.is_dynamic():
            pin_state = model.model.dynamic_pins[model.load_device]

            if pin_state["active"]:
                for subset in ("weights", "weights-loaded", "weights-fast"):
                    *_, buckets = pin_state[subset]
                    for size, bucket in list(buckets.items()):
                        bucket[:] = [ entry for entry in bucket if entry[-1] is not None ]
                        if not bucket:
                            del buckets[size]

            pin_state["active"] = False
            model.partially_unload_ram(1e30, subsets=[ "patches", "patches-loaded", "patches-fast" ])
            for subset in ("patches", "patches-loaded", "patches-fast"):
                pin_state[subset] = (comfy_aimdo.host_buffer.HostBuffer(0, 8 * 1024 * 1024, pinned_hostbuf_size(model.model_size())), [], [-1], [0], [0], {})

    manager.stream_cast_buffers.clear()
    manager.stream_aimdo_cast_buffers.clear()
    soft_empty_cache()

def get_offload_stream(device):
    manager = get_model_manager()
    stream_counter = manager.stream_counters.get(device, 0)
    if NUM_STREAMS == 0:
        return None

    if torch.compiler.is_compiling():
        return None

    if device in manager.streams:
        ss = manager.streams[device]
        #Sync the oldest stream in the queue with the current
        ss[stream_counter].wait_stream(current_stream(device))
        stream_counter = (stream_counter + 1) % len(ss)
        manager.stream_counters[device] = stream_counter
        return ss[stream_counter]
    elif is_device_cuda(device):
        ss = []
        for k in range(NUM_STREAMS):
            s1 = torch.cuda.Stream(device=device, priority=0)
            s1.as_context = torch.cuda.stream
            ss.append(s1)
        manager.streams[device] = ss
        s = ss[stream_counter]
        manager.stream_counters[device] = stream_counter
        return s
    elif is_device_xpu(device):
        ss = []
        for k in range(NUM_STREAMS):
            s1 = torch.xpu.Stream(device=device, priority=0)
            s1.as_context = torch.xpu.stream
            ss.append(s1)
        manager.streams[device] = ss
        s = ss[stream_counter]
        manager.stream_counters[device] = stream_counter
        return s
    elif is_device_npu(device):
        ss = []
        # Match the CUDA/XPU stream interface used by the shared offload path.
        # torch.npu.stream provides the context manager for torch-npu streams.
        for k in range(NUM_STREAMS):
            s1 = torch.npu.Stream(device=device, priority=0)
            s1.as_context = torch.npu.stream
            ss.append(s1)
        manager.streams[device] = ss
        s = ss[stream_counter]
        manager.stream_counters[device] = stream_counter
        return s
    return None

def sync_stream(device, stream):
    if stream is None or current_stream(device) is None:
        return
    current_stream(device).wait_stream(stream)


def cast_to_gathered(tensors, r, non_blocking=False, stream=None, r2=None):
    wf_context = nullcontext()
    if stream is not None:
       wf_context = stream
       if hasattr(wf_context, "as_context"):
           wf_context = wf_context.as_context(stream)

    dest_views = dinkster_inference.memory_management.interpret_gathered_like(tensors, r) if r is not None else [None] * len(tensors)
    dest2_views = dinkster_inference.memory_management.interpret_gathered_like(tensors, r2) if r2 is not None else None
    with wf_context:
        for tensor in tensors:
            dest_view = dest_views.pop(0)
            dest2_view = dest2_views.pop(0) if dest2_views is not None else None
            if tensor is None:
                continue
            if dinkster_inference.memory_management.read_tensor_file_slice_into(tensor, dest_view, stream=stream, destination2=dest2_view):
                continue
            storage = tensor._qdata.untyped_storage() if isinstance(tensor, dinkster_inference.quant_ops.QuantizedTensor) else tensor.untyped_storage()
            mark_mmap_dirty(storage)
            if dest_view is not None:
                dest_view.copy_(tensor, non_blocking=non_blocking)
            if dest2_view is not None:
                dest2_view.copy_(tensor if dest_view is None else dest_view, non_blocking=non_blocking)


def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        if stream is not None:
            wf_context = stream
            if hasattr(wf_context, "as_context"):
                wf_context = wf_context.as_context(stream)
            with wf_context:
                return weight.to(dtype=dtype, copy=copy)
        return weight.to(dtype=dtype, copy=copy)


    if stream is not None:
        wf_context = stream
        if hasattr(wf_context, "as_context"):
            wf_context = wf_context.as_context(stream)
        with wf_context:
            if r is None:
                r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight, non_blocking=non_blocking)
    else:
        if r is None:
            r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
    return r

def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)


def get_disk_swap_total():
    if not os.path.exists("/proc/swaps"):
        return 0

    total = 0
    try:
        with open("/proc/swaps", encoding="utf-8") as swaps:
            next(swaps, None)
            for line in swaps:
                filename, _, size, _, _ = line.rsplit(maxsplit=4)
                if os.path.basename(os.path.realpath(filename)).startswith("zram"):
                    continue
                total += int(size) * 1024
    except:
        logging.warning("Could not get amount of swap memory on system.")
    return total

def maximum_pinned_memory():
    if args.disable_pinned_memory or not (is_nvidia() or is_amd()):
        return -1
    ram = get_total_memory(torch.device("cpu"))
    if WINDOWS:
        maximum = ram * 0.40  # Windows limit is apparently 50%
    else:
        swap = 0 if dinkster_inference.system_memory.cgroup_memory_limit() is not None else get_disk_swap_total()
        maximum = max(ram * 0.40, min(ram * 0.90, ram - 4 * 1024 ** 3, ram + swap - 16 * 1024 ** 3))
    logging.info("Enabled pinned memory {}".format(maximum // (1024 * 1024)))
    return maximum


get_model_manager().max_pinned_memory = maximum_pinned_memory()

PINNING_ALLOWED_TYPES = set(["Tensor", "Parameter", "QuantizedTensor"])

def pinned_hostbuf_size(size):
    if args.high_ram:
        return max(0, int(size * 2))
    return max(0, int(min(size, get_model_manager().max_pinned_memory) * 2))

def discard_cuda_async_error():
    try:
        a = torch.tensor([1], dtype=torch.uint8, device=get_torch_device())
        b = torch.tensor([1], dtype=torch.uint8, device=get_torch_device())
        _ = a + b
        synchronize()
    except RuntimeError:
        #Dump it! We already know about it from the synchronous return
        pass

def pin_memory(tensor, evict_active=True):
    manager = get_model_manager()
    if manager.max_pinned_memory <= 0:
        return False

    if type(tensor).__name__ not in PINNING_ALLOWED_TYPES:
        return False

    if not is_device_cpu(tensor.device):
        return False

    if tensor.is_pinned():
        #NOTE: Cuda does detect when a tensor is already pinned and would
        #error below, but there are proven cases where this also queues an error
        #on the GPU async. So dont trust the CUDA API and guard here
        return False

    if not tensor.is_contiguous():
        return False

    size = tensor.nbytes
    dinkster_inference.memory_management.extra_ram_release(dinkster_inference.memory_management.RAM_CACHE_HEADROOM)
    if not ensure_pin_registerable(size, evict_active=evict_active):
        return False

    ptr = tensor.data_ptr()
    if ptr == 0:
        return False

    if torch.cuda.cudart().cudaHostRegister(ptr, size, 1) == 0:
        manager.pinned_memory[ptr] = size
        manager.add_pinned_memory(size)
        return True
    else:
        logging.warning("Pin error.")
        discard_cuda_async_error()

    return False

def unpin_memory(tensor):
    manager = get_model_manager()
    if manager.max_pinned_memory <= 0:
        return False

    if not is_device_cpu(tensor.device):
        return False

    ptr = tensor.data_ptr()
    size = tensor.nbytes

    size_stored = manager.pinned_memory.get(ptr, None)
    if size_stored is None:
        logging.warning("Tried to unpin tensor not pinned by ComfyUI")
        return False

    if size != size_stored:
        logging.warning("Size of pinned tensor changed")
        return False

    if torch.cuda.cudart().cudaHostUnregister(ptr) == 0:
        size = manager.pinned_memory.pop(ptr)
        manager.release_pinned_memory(size)
        return True
    else:
        logging.warning("Unpin error.")
        discard_cuda_async_error()

    return False

def sage_attention_enabled():
    return args.use_sage_attention

def comfy_kitchen_attention_enabled():
    return args.use_ck_attention

def flash_attention_enabled():
    return args.use_flash_attention

def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if is_ascend_npu():
        return False
    if is_mlu():
        return False
    if is_ixuca():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE

def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_enabled_vae():
    if is_amd():
        return False  # enabling pytorch attention on AMD currently causes crash when doing high res
    return pytorch_attention_enabled()

def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        #TODO: more reliable way of checking for flash attention?
        if is_nvidia():
            return True
        if is_intel_xpu():
            return True
        if is_ascend_npu():
            return True
        if is_mlu():
            return True
        if is_amd():
            return True #if you have pytorch attention enabled on AMD it probably supports at least mem efficient attention
        if is_ixuca():
            return True
    return False

def force_upcast_attention_dtype():
    upcast = args.force_upcast_attention

    macos_version = mac_version()
    if macos_version is not None and ((14, 5) <= macos_version):  # black image bug on recent versions of macOS, I don't think it's ever getting fixed
        upcast = True

    if upcast:
        return {torch.float16: torch.float32}
    else:
        return None

#Developers and agents: You almost never want to call this function from Model code as it does
#not account for ComfyUIs smart memory feature combining with Dynamic VRAM, where inactive models
#are preserved in VRAM right up until there is higher priority demand (I.E whatever you want to do
#that makes you meansure VRAM from model code). Instead call get_free_memory() on the ModelPatcher
#for your BaseModel object (.current_patcher) instead to count this VRAM as free and then Dynamic
#VRAM will evict that extra VRAM for you when you use it.
def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = dinkster_inference.system_memory.virtual_memory_available()
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_xpu + mem_free_torch
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_npu, _ = torch.npu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_npu + mem_free_torch
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_mlu, _ = torch.mlu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_mlu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total

def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU

def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS

def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False

def is_device_cpu(device):
    return is_device_type(device, 'cpu')

def is_device_mps(device):
    return is_device_type(device, 'mps')

def is_device_xpu(device):
    return is_device_type(device, 'xpu')

def is_device_cuda(device):
    return is_device_type(device, 'cuda')

def is_device_npu(device):
    return is_device_type(device, 'npu')

def set_torch_device(device):
    """Set the current device for the given torch device. Supports CUDA and XPU."""
    if is_device_cuda(device):
        torch.cuda.set_device(device)
    elif is_device_xpu(device):
        torch.xpu.set_device(device)

def is_directml_enabled():
    global directml_enabled
    if directml_enabled:
        return True

    return False

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None:
        if is_device_cpu(device):
            return False

    if args.force_fp16:
        return True

    if FORCE_FP32:
        return False

    if is_directml_enabled():
        return True

    if (device is not None and is_device_mps(device)) or mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return torch.xpu.get_device_properties(device).has_fp16

    if is_ascend_npu():
        return True

    if is_mlu():
        return True

    if is_ixuca():
        return True

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    #FP16 is confirmed working on a 1080 (GP104) and on latest pytorch actually seems faster than fp32
    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            if WINDOWS or manual_cast:
                return True
            else:
                return False #weird linux behavior where fp32 is faster

    if manual_cast:
        free_model_memory = maximum_vram_for_weights(device)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    #FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None:
        if is_device_cpu(device): #TODO ? bf16 works on CPU but is extremely slow
            return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if (device is not None and is_device_mps(device)) or mps_mode():
        if mac_version() < (14,):
            return False
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return torch.xpu.is_bf16_supported()

    if is_ascend_npu():
        return True

    if is_ixuca():
        return True

    if is_amd():
        arch = torch.cuda.get_device_properties(device).gcnArchName
        if any((a in arch) for a in AMD_RDNA2_AND_OLDER_ARCH):  # RDNA2 and older don't support bf16
            if manual_cast:
                return True
            return False

    props = torch.cuda.get_device_properties(device)

    if is_mlu():
        if props.major > 3:
            return True

    if props.major >= 8:
        return True

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works and manual_cast:
        free_model_memory = maximum_vram_for_weights(device)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False

def supports_fp8_compute(device=None):
    if SUPPORT_FP8_OPS:
        return True

    if not is_nvidia():
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major >= 9:
        return True
    if props.major < 8:
        return False
    if props.minor < 9:
        return False

    if torch_version_numeric < (2, 3):
        return False

    if WINDOWS:
        if torch_version_numeric < (2, 4):
            return False

    return True

def supports_nvfp4_compute(device=None):
    if not is_nvidia():
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major < 10:
        return False

    return True

def supports_mxfp8_compute(device=None):
    if not is_nvidia():
        return False

    if torch_version_numeric < (2, 10):
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major < 10:
        return False

    return True

def supports_fp64(device=None):
    if (device is not None and is_device_mps(device)) or mps_mode():
        return False

    if is_intel_xpu():
        return False

    if is_directml_enabled():
        return False

    if is_ixuca():
        return False

    return True

def supports_int8_compute(device=None):
    # The eager comfy_kitchen backend implements int8 weight-only quantized
    # matmul via torch._int_mm, which PyTorch does not implement for MPS.
    # https://github.com/pytorch/pytorch/issues/141287
    if (device is not None and is_device_mps(device)) or mps_mode():
        return False

    if is_intel_xpu():
        return False

    if is_directml_enabled():
        return False

    if is_ixuca():
        return False

    return True

def extended_fp16_support():
    # TODO: check why some models work with fp16 on newer torch versions but not on older
    if torch_version_numeric < (2, 7):
        return False

    return True

def lora_compute_dtype(device):
    manager = get_model_manager()
    dtype = manager.lora_compute_dtypes.get(device, None)
    if dtype is not None:
        return dtype

    if should_use_fp16(device):
        dtype = torch.float16
    else:
        dtype = torch.float32

    manager.lora_compute_dtypes[device] = dtype
    return dtype

def synchronize():
    if cpu_mode():
        return
    if is_intel_xpu():
        torch.xpu.synchronize()
    elif is_ascend_npu():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

def soft_empty_cache(force=False):
    if cpu_mode():
        return
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    elif is_ascend_npu():
        torch.npu.empty_cache()
    elif is_mlu():
        torch.mlu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def unload_all_models():
    return get_model_manager().unload_all_models()

def unload_model_and_clones(model: ModelPatcher, unload_additional_models=True, all_devices=False):
    return get_model_manager().unload_model_and_clones(
        model, unload_additional_models, all_devices
    )


def _unload_model_and_clones(manager, model: ModelPatcher, unload_additional_models=True, all_devices=False):
    'Unload only model and its clones - primarily for multigpu cloning purposes.'
    initial_keep_loaded = manager.loaded_model_records()
    additional_models = []
    if unload_additional_models:
        additional_models = model.get_nested_additional_models()
    keep_loaded = []
    for loaded_model in initial_keep_loaded:
        if loaded_model.model is not None:
            if model.clone_base_uuid == loaded_model.model.clone_base_uuid:
                continue
            # check additional models if they are a match
            skip = False
            for add_model in additional_models:
                if add_model.clone_base_uuid == loaded_model.model.clone_base_uuid:
                    skip = True
                    break
            if skip:
                continue
        keep_loaded.append(loaded_model)
    if not all_devices:
        manager.free_memory(1e30, get_torch_device(), keep_loaded)
    else:
        for device in get_all_torch_devices():
            manager.free_memory(1e30, device, keep_loaded)

def debug_memory_summary():
    if is_amd() or is_nvidia():
        return torch.cuda.memory.memory_summary()
    return ""

class InterruptProcessingException(BaseException):
    pass

def interrupt_current_processing(value=True):
    manager = get_model_manager()
    with manager.interrupt_processing_mutex:
        manager.interrupt_processing = value

def processing_interrupted():
    manager = get_model_manager()
    with manager.interrupt_processing_mutex:
        return manager.interrupt_processing

def throw_exception_if_processing_interrupted():
    manager = get_model_manager()
    with manager.interrupt_processing_mutex:
        if manager.interrupt_processing:
            manager.interrupt_processing = False
            raise InterruptProcessingException()
