from fable_local import prompts


def test_local_prompt_file_replaces_the_default(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_text("default")
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    prompts.load.cache_clear()
    assert prompts.load("x") == "default"
    (tmp_path / "x.local.md").write_text("mine")
    prompts.load.cache_clear()
    assert prompts.load("x") == "mine"
    prompts.load.cache_clear()
