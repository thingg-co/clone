import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from frank_local.config import Settings
from frank_local.render import UI
from frank_local.tools.base import ToolContext


class FakeAgent:
    """Just enough of Agent for tools to run."""

    def __init__(self, settings: Settings, cwd: Path):
        self.settings = settings
        self.cwd = cwd
        self.tool_output_limit = settings.max_tool_output_chars


@pytest.fixture
def settings():
    return Settings(permission_mode="ask", bash_timeout=10, max_tool_output_chars=2000)


@pytest.fixture
def ctx(tmp_path, settings):
    return ToolContext(cwd=tmp_path, agent=FakeAgent(settings, tmp_path))


@pytest.fixture
def ui():
    return UI(Console(file=io.StringIO(), width=100), interactive=False)


def delta(content=None, reasoning=None, tool_calls=None):
    return SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)


def tc(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))
