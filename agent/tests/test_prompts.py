from frank_local import prompts


def test_local_prompt_file_replaces_the_default(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_text("default")
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    prompts.load.cache_clear()
    assert prompts.load("x") == "default"
    (tmp_path / "x.local.md").write_text("mine")
    prompts.load.cache_clear()
    assert prompts.load("x") == "mine"
    prompts.load.cache_clear()


def test_instruction_files_include_claude_codes_places(tmp_path, monkeypatch):
    from frank_local.prompt import instruction_files

    home = tmp_path / "home"
    proj = home / "work" / "proj"
    (home / ".claude").mkdir(parents=True)
    (proj / ".claude").mkdir(parents=True)
    (home / ".claude" / "CLAUDE.md").write_text("user rules")
    (home / "work" / "CLAUDE.md").write_text("outer")
    (proj / "CLAUDE.md").write_text("project")
    (proj / ".claude" / "CLAUDE.md").write_text("project, under .claude")
    (proj / "CLAUDE.local.md").write_text("mine")
    monkeypatch.setenv("HOME", str(home))
    found = [p.relative_to(home).as_posix() for p in instruction_files(proj)]
    assert found == [".claude/CLAUDE.md", "work/CLAUDE.md", "work/proj/CLAUDE.md", "work/proj/.claude/CLAUDE.md", "work/proj/CLAUDE.local.md"]
