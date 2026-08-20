"""Architecture guard for the unified model accounting boundary."""

import ast
from pathlib import Path

RUNTIME_ROOT = Path(__file__).parents[1] / "src" / "open_deep_research"

# These calls are runtime/tool protocol invocations rather than physical model calls.
# Physical LangChain calls are only allowed inside observability/core.py.
ALLOWED_DIRECT_INVOCATIONS = {
    ("agents/query_engine.py", "ainvoke"),
    ("observability/core.py", "ainvoke"),
    ("sandbox/worker.py", "ainvoke"),
    ("tools/adapters.py", "ainvoke"),
}
MODEL_METHODS = {"invoke", "ainvoke", "astream"}
ACCOUNTED_WRAPPERS = {
    "invoke_model_with_observability",
    "invoke_model_with_retry_observability",
}


def test_runtime_model_calls_cannot_bypass_accounting_boundary() -> None:
    """Fail when a new direct model-style invocation bypasses accounting."""
    violations: list[str] = []

    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        relative = path.relative_to(RUNTIME_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method not in MODEL_METHODS:
                continue
            if (relative, method) not in ALLOWED_DIRECT_INVOCATIONS:
                violations.append(f"{relative}:{node.lineno} .{method}()")

    assert not violations, (
        "Model-style calls must use invoke_model_with_retry_observability; "
        f"review or explicitly whitelist non-model calls: {violations}"
    )


def test_runtime_accounted_calls_declare_research_stage() -> None:
    """Require new model calls to choose one of the six business stages."""
    violations: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        relative = path.relative_to(RUNTIME_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in ACCOUNTED_WRAPPERS:
                continue
            if not any(keyword.arg == "stage" for keyword in node.keywords):
                violations.append(f"{relative}:{node.lineno}")

    assert not violations, (
        "Accounted model calls must explicitly declare a research stage: "
        f"{violations}"
    )
