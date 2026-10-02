from tools.rewrite_upstream_imports import rewrite


def test_rewrite_preserves_external_gguf_metadata_key(tmp_path):
    source = tmp_path / "source.py"
    source.write_text(
        "import dinkster_inference\n"
        "module = dinkster_inference.sd\n"
        'target = "dinkster_inference.ldm.models.autoencoder.Encoder"\n'
        'field = "comfy.gguf.orig_shape.weight"\n'
    )

    assert rewrite(source)
    assert source.read_text() == (
        "import dinkster_inference\n"
        "module = dinkster_inference.sd\n"
        'target = "dinkster_inference.ldm.models.autoencoder.Encoder"\n'
        'field = "comfy.gguf.orig_shape.weight"\n'
    )
