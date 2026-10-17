from frank_local.permissions import is_safe_command


def test_safe_commands():
    for c in ["ls -la", "git status", "git log --oneline -5", "rg foo src | head", "cat a.txt && wc -l b.txt",
              "find . -name '*.py'", "FOO=1 ls", "git diff HEAD~1 -- file.py"]:
        assert is_safe_command(c), c


def test_unsafe_commands():
    for c in ["rm -rf /", "ls > out.txt", "git push", "git commit -m x", "find . -delete", "echo $(whoami)",
              "python3 script.py", "pytest", "sudo ls", "ls; make", "git branch -D main", "cat a | tee b"]:
        assert not is_safe_command(c), c


def test_extra_prefixes():
    assert is_safe_command("uv run pytest -q", ["uv run pytest"])
    assert not is_safe_command("uv run pytestx", ["uv run pytest"])
