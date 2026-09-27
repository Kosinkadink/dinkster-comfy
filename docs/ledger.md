# Fork ledger

Upstream fork point: `comfyanonymous/ComfyUI` commit
`b5cc8830279eae909a59de030af1e50761c36751`.

| File or subsystem | Tier | Reason |
| --- | --- | --- |
| `dinkster_comfy/` except entries below | kept upstream | ComfyUI inference implementation, mechanically renamed from `comfy` and rewritten to import `dinkster_comfy` |
| `dinkster_comfy/hooks.py` | changed by us | Owns `conditioning_set_values` instead of importing the deleted application helper |
| `dinkster_comfy/ops.py` | changed by us | Initializes Torch's CUDA SDPA chooser on the first real invocation, then restores the declared backend priority to keep cold and warm execution identical |
| `dinkster_comfy/ldm/sam3d_body/face_landmarker.py` | changed by us | Relocates the upstream face landmarker required by the retained SAM 3D Body model from deleted `comfy_extras` |
| `dinkster_comfy/ldm/sam3d_body/model/model.py` | changed by us | Imports the relocated face landmarker from the library |
| `tests-unit/comfy_test/`, `tests-unit/comfy_quant/`, `tests-unit/deploy_environment_test.py` | changed by us | Retains pure-library tests and rewrites their package imports |
| `pyproject.toml` | ours only | Builds distribution `dinkster-comfy`, discovers `dinkster_comfy`, and declares runtime dependencies |
| `tools/rewrite_upstream_imports.py` | ours only | Applies the required package import rewrite after upstream cherry-picks |
| ComfyUI application, server, API, execution, nodes, extras, middleware, assets, model directories, and application tests | changed by us | Removed because this repository distributes the inference library only |
| `docs/ledger.md` and `README.md` | ours only | Documents package use, provenance, and maintained differences |
