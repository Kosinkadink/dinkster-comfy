# dinkster-comfy

`dinkster-comfy` is Dinkster's inference library. It starts from ComfyUI's
`comfy/` package while excluding the ComfyUI application, server, nodes, and
web assets.

The distribution name is `dinkster-comfy`; Python imports use
`dinkster_comfy`:

```bash
python -m pip install .
python -c "import dinkster_comfy.sd, dinkster_comfy.samplers, dinkster_comfy.model_management"
```

The upstream fork point and maintained differences are recorded in
[`docs/ledger.md`](docs/ledger.md). Run `tools/rewrite_upstream_imports.py` on
Python files brought in from upstream before committing a sync.
