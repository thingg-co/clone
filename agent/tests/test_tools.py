import signal
import subprocess
import time

import pytest

from fable_local.tools.base import truncate_middle
from fable_local.tools.fs import EditFile, GlobFiles, GrepFiles, ListDir, ReadFile, WriteFile
from fable_local.tools.shell import Bash


def test_write_read_edit(ctx, tmp_path):
    w = WriteFile().run({"path": "a/b.py", "content": "x = 1\ny = 2\nz = 3\n"}, ctx)
    assert not w.is_error and (tmp_path / "a" / "b.py").exists()
    r = ReadFile().run({"path": "a/b.py"}, ctx)
    assert r.content.splitlines()[1].endswith("\ty = 2")
    r2 = ReadFile().run({"path": "a/b.py", "offset": 2, "limit": 1}, ctx)
    assert r2.content.strip().startswith("2\ty = 2")
    e = EditFile().run({"path": "a/b.py", "old_string": "y = 2", "new_string": "y = 22"}, ctx)
    assert not e.is_error and "+y = 22" in e.content
    assert (tmp_path / "a" / "b.py").read_text() == "x = 1\ny = 22\nz = 3\n"


def test_edit_errors(ctx, tmp_path):
    (tmp_path / "f.txt").write_text("a\na\n")
    e = EditFile().run({"path": "f.txt", "old_string": "a", "new_string": "b"}, ctx)
    assert e.is_error and "read" in e.content  # must read first in ask mode
    ReadFile().run({"path": "f.txt"}, ctx)
    e = EditFile().run({"path": "f.txt", "old_string": "a", "new_string": "b"}, ctx)
    assert e.is_error and "2 times" in e.content
    e = EditFile().run({"path": "f.txt", "old_string": "zzz", "new_string": "b"}, ctx)
    assert e.is_error and "not found" in e.content
    e = EditFile().run({"path": "f.txt", "old_string": "a", "new_string": "b", "replace_all": True}, ctx)
    assert not e.is_error and (tmp_path / "f.txt").read_text() == "b\nb\n"


def test_read_missing_and_binary(ctx, tmp_path):
    assert ReadFile().run({"path": "nope"}, ctx).is_error
    (tmp_path / "bin").write_bytes(b"\x00\x01\x02")
    assert ReadFile().run({"path": "bin"}, ctx).is_error


def test_bash_cwd_persists_and_exit_code(ctx, tmp_path):
    (tmp_path / "sub").mkdir()
    b = Bash()
    r = b.run({"command": "cd sub && echo hi"}, ctx)
    assert r.content == "hi" and not r.is_error
    assert ctx.cwd == tmp_path / "sub"
    r = b.run({"command": "pwd"}, ctx)
    assert r.content.endswith("sub")
    r = b.run({"command": "echo oops >&2; exit 3"}, ctx)
    assert r.is_error and "exit code 3" in r.content and "oops" in r.content


def test_bash_timeout(ctx):
    r = Bash().run({"command": "sleep 5", "timeout": 1}, ctx)
    assert r.is_error and "timed out" in r.content


def _running(pid: int, wait: float = 3.0) -> bool:
    """Whether pid is still alive (a zombie counts as gone) after up to `wait` seconds."""
    end = time.monotonic() + wait
    while True:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if not state or state.startswith("Z"):
            return False
        if time.monotonic() > end:
            return True
        time.sleep(0.05)


# A job sent to the background ignores SIGINT and keeps the output pipe open, so stopping
# only bash, as subprocess.run does, leaves it running.
LEAVES_A_CHILD = "sleep 30 & echo $! > child.pid; sleep 30"


def test_bash_timeout_ends_what_the_command_started(ctx, tmp_path):
    r = Bash().run({"command": LEAVES_A_CHILD, "timeout": 1}, ctx)
    assert r.is_error and "timed out" in r.content
    assert not _running(int((tmp_path / "child.pid").read_text()))


def test_bash_interrupt_ends_what_the_command_started(ctx, tmp_path):
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    old = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        with pytest.raises(KeyboardInterrupt):
            Bash().run({"command": LEAVES_A_CHILD, "timeout": 60}, ctx)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    assert not _running(int((tmp_path / "child.pid").read_text()))


def test_glob_grep_list(ctx, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text("def alpha():\n    return 1\n")
    (tmp_path / "src" / "two.txt").write_text("alpha beta\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.py").write_text("alpha\n")
    g = GlobFiles().run({"pattern": "**/*.py"}, ctx)
    assert "src/one.py" in g.content and "node_modules" not in g.content
    r = GrepFiles().run({"pattern": "alpha", "glob": "*.py"}, ctx)
    assert "src/one.py:1:" in r.content and "two.txt" not in r.content
    r = GrepFiles().run({"pattern": "nomatch"}, ctx)
    assert r.content == "no matches"
    ls = ListDir().run({}, ctx)
    assert "src/" in ls.content


def test_truncate_middle():
    s = "a" * 100 + "b" * 100
    t = truncate_middle(s, 60)
    assert "truncated" in t and t.startswith("aaaa") and t.endswith("bbbb") and len(t) < 160
    assert truncate_middle("short", 60) == "short"
