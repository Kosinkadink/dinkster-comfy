# Fork ledger

Upstream fork point: `comfyanonymous/ComfyUI` commit
`fa98a189b4271c76f66210e15f81790b555eb610`.

| File or subsystem | Tier | Reason |
| --- | --- | --- |
| `dinkster_inference/` except entries below | kept upstream | ComfyUI inference implementation, mechanically renamed from `comfy`, rewritten to import `dinkster_inference`, and source-normalized to ASCII without runtime changes |
| `dinkster_inference/hooks.py` | changed by us | Owns `conditioning_set_values` instead of importing the deleted application helper |
| `dinkster_inference/window_plan.py` | ours only | Compiles layered media-axis window declarations into canonical joint windows with deterministic weighted merge semantics |
| `dinkster_inference/window_execution.py`, `dinkster_inference/samplers.py` window-plan dispatch | changed by us | Evaluates compiled joint windows through declared tensor kinds, gathers full-domain fields per window, and merges once with per-occurrence accumulation |
| `dinkster_inference/window_execution.py` invariant kinds | changed by us | Keeps media axes absent from a tensor kind unsliced and merges their repeated joint-window contributions without assigning raw tensor dimensions |
| `dinkster_inference/window_execution.py`, `dinkster_inference/samplers.py` window masks | changed by us | Compiles conditioning and ControlNet effect masks once in full-domain coordinates and gathers them through each declared semantic window |
| `dinkster_inference/minimax_control.py` | ours only | Loads and applies MiniMax H3 Fun ControlNet model patches, compiling control inputs once over the full video domain and gathering their latent per semantic window |
| `dinkster_inference/context_windows.py` temporal adapter | changed by us | Compiles stock temporal context schedules into the same layered media-axis plan used by spatial windows |
| `dinkster_inference/ldm/modules/attention.py` | changed by us | Supports caller-owned attention function registries for isolated worker processes while preserving the upstream default registry |
| `dinkster_inference/patch_program.py`, `dinkster_inference/model_patcher.py` | changed by us | Represents ordered weight changes as immutable, content-identified patch programs while preserving the existing model patcher calls |
| `dinkster_inference/patch_program.py`, `dinkster_inference/model_patcher.py`, `dinkster_inference/model_base.py` | changed by us | Adds symbolic module insertions with reversible materialization and clone, sharing, device, and offload policies behind existing patcher calls |
| `dinkster_inference/model_management.py` | changed by us | Routes model loading, unloading, partial offload, cleanup, and pin eviction through a process-owned manager while preserving existing calls |
| `dinkster_inference/contribution_gain.py`, `dinkster_inference/hooks.py`, `dinkster_inference/controlnet.py`, `dinkster_inference/samplers.py` | changed by us | Realizes shared timeline, site, guidance-lane, and effect-mask gains once against the executed sigma table for hooks, controls, and conditioning |
| `dinkster_inference/sampler_assembly.py`, `dinkster_inference/res4lyf_rk.py`, `dinkster_inference/res4lyf_sampler.py`, `dinkster_inference/samplers.py` | changed by us | Assembles typed samplers with ordered model-evaluation substeps and independent step/substep noise streams, including receipt-backed RES4LYF RK solvers, while preserving ordinary sampler behavior |
| `dinkster_inference/ldm/sam3d_body/face_landmarker.py` | changed by us | Relocates the upstream face landmarker required by the retained SAM 3D Body model from deleted `comfy_extras` |
| `dinkster_inference/ldm/sam3d_body/model/model.py` | changed by us | Imports the relocated face landmarker from the library |
| `tests-unit/comfy_test/`, `tests-unit/comfy_quant/`, `tests-unit/deploy_environment_test.py` | changed by us | Retains pure-library tests and rewrites their package imports |
| `pyproject.toml` | ours only | Builds distribution `dinkster-inference`, discovers `dinkster_inference`, and declares runtime dependencies |
| `tools/rewrite_upstream_imports.py` | ours only | Applies the required package import rewrite after upstream cherry-picks |
| `.github/workflows/ci.yml` | ours only | Builds and installs the wheel, verifies standalone imports, and runs the retained library tests |
| ComfyUI application and release workflows | changed by us | Removed because their server, API, frontend, packaging, and release targets are not part of this library |
| ComfyUI application, server, API, execution, nodes, extras, middleware, assets, model directories, and application tests | changed by us | Removed because this repository distributes the inference library only |
| `docs/ledger.md` and `README.md` | ours only | Documents package use, provenance, and maintained differences |
