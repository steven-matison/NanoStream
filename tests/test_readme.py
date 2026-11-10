import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAPPER_SET = {"--source", "--url", "--log", "--daemon", "--cache"}  # nano-stream start and NANOSTREAM_CACHE set these


def renderer_flags():
    with open(os.path.join(ROOT, "nanostream", "render_stream.py")) as f:
        tree = ast.parse(f.read())
    return {n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
            and n.args and isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith("--")}


def table_flags():
    # the first cell of every row of the "Renderer options go in NANO_STREAM_ARGS" table
    with open(os.path.join(ROOT, "README.md")) as f:
        text = f.read()
    table = text.split("Renderer options go in `NANO_STREAM_ARGS`", 1)[1].split("\n\n", 2)[1]
    return {flag for row in table.splitlines()[2:] for flag in re.findall(r"--[a-z][a-z-]*", row.split("|")[1])}


def test_every_renderer_option_is_in_the_readme_table():
    assert renderer_flags() - WRAPPER_SET - table_flags() == set()


def test_every_option_in_the_readme_table_exists():
    assert table_flags() - renderer_flags() == set()
