# Fork ledger

Upstream fork point: `comfyanonymous/ComfyUI` commit
`4ef23c34d950eecc37040a21ee1741a49d2e44b1`.

| File or subsystem | Tier | Reason |
| --- | --- | --- |
| `dinkster_comfy/` except entries below | kept upstream | ComfyUI inference implementation, mechanically renamed from `comfy`, rewritten to import `dinkster_comfy`, and source-normalized to ASCII without runtime changes |
| `dinkster_comfy/hooks.py` | changed by us | Owns `conditioning_set_values` instead of importing the deleted application helper |
| `dinkster_comfy/ldm/sam3d_body/face_landmarker.py` | changed by us | Relocates the upstream face landmarker required by the retained SAM 3D Body model from deleted `comfy_extras` |
| `dinkster_comfy/ldm/sam3d_body/model/model.py` | changed by us | Imports the relocated face landmarker from the library |
| `tests-unit/comfy_test/`, `tests-unit/comfy_quant/`, `tests-unit/deploy_environment_test.py` | changed by us | Retains pure-library tests and rewrites their package imports |
| `pyproject.toml` | ours only | Builds distribution `dinkster-comfy`, discovers `dinkster_comfy`, and declares runtime dependencies |
| `tools/rewrite_upstream_imports.py` | ours only | Applies the required package import rewrite after upstream cherry-picks |
| `.github/workflows/ci.yml` | ours only | Builds and installs the wheel, verifies standalone imports, and runs the retained library tests |
| ComfyUI application and release workflows | changed by us | Removed because their server, API, frontend, packaging, and release targets are not part of this library |
| ComfyUI application, server, API, execution, nodes, extras, middleware, assets, model directories, and application tests | changed by us | Removed because this repository distributes the inference library only |
| `docs/ledger.md` and `README.md` | ours only | Documents package use, provenance, and maintained differences |
