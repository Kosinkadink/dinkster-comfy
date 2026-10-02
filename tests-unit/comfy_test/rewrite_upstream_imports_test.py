import sys

from tools.rewrite_upstream_imports import main, rewrite


def test_rewrite_preserves_external_gguf_metadata_key(tmp_path):
    source = tmp_path / "source.py"
    source.write_text(
        "import comfy\n"
        "module = comfy.sd\n"
        'target = "comfy.ldm.models.autoencoder.Encoder"\n'
        'field = "comfy.gguf.orig_shape.weight"\n'
    )

    assert rewrite(source)
    assert source.read_text() == (
        "import dinkster_inference\n"
        "module = dinkster_inference.sd\n"
        'target = "dinkster_inference.ldm.models.autoencoder.Encoder"\n'
        'field = "comfy.gguf.orig_shape.weight"\n'
    )


def test_rewrite_cli_skips_its_fixture_file(tmp_path, monkeypatch):
    source = tmp_path / "rewrite_upstream_imports_test.py"
    source.write_text("import comfy\n")
    monkeypatch.setattr(sys, "argv", ["rewrite_upstream_imports.py", str(source)])

    main()

    assert source.read_text() == "import comfy\n"
