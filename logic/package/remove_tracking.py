"""Script to clean up and remove the tracking module from WSmart-Route, replacing loggers with prints."""

import re
import shutil
from pathlib import Path


def get_project_root() -> Path:
    """Find WSmart-Route root directory."""
    return Path(__file__).resolve().parents[2]


def remove_path(path: Path):
    """Delete a file or directory safely."""
    if not path.exists():
        return
    root = get_project_root()
    rel_path = path.relative_to(root) if path.is_absolute() else path
    if path.is_dir():
        print(f"Removing directory: {rel_path}")
        shutil.rmtree(path)
    else:
        print(f"Removing file: {rel_path}")
        path.unlink()


def fix_empty_try_blocks(content: str) -> str:
    """Fix try blocks that became empty after commenting out tracking imports by inserting pass."""
    pattern = r"(?m)^(\s*try\s*:\s*\n)((?:\s*#[^\n]*\n)+)(\s*except\b)"

    def repl(match):
        try_line = match.group(1)
        comments = match.group(2)
        except_line = match.group(3)
        indent_str = try_line.split("try")[0] + "    "
        return f"{try_line}{indent_str}pass\n{comments}{except_line}"

    return re.sub(pattern, repl, content)


def append_to_class_body(content: str, class_name: str, method_lines: str) -> str:
    """Append method_lines (unindented) to the end of the named class body in content."""
    lines = content.splitlines(keepends=True)

    class_start = -1
    class_indent_len = 0
    for i, line in enumerate(lines):
        if re.match(rf"\s*class\s+{re.escape(class_name)}\b", line):
            class_start = i
            class_indent_len = len(line) - len(line.lstrip())
            break

    if class_start == -1:
        return content

    body_indent = " " * (class_indent_len + 4)
    for i in range(class_start + 1, len(lines)):
        if lines[i].strip():
            body_indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            break

    class_end = len(lines)
    for i in range(class_start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        line_indent_len = len(line) - len(line.lstrip())
        if line_indent_len <= class_indent_len:
            class_end = i
            break

    insert_after = class_end - 1
    while insert_after > class_start and not lines[insert_after].strip():
        insert_after -= 1

    method_text = "\n"
    for raw_line in method_lines.splitlines():
        if raw_line.strip():
            method_text += body_indent + raw_line + "\n"
        else:
            method_text += "\n"

    lines.insert(insert_after + 1, method_text)
    return "".join(lines)


def remove_viz_mixin_from_file(file_path: Path):
    """Modify Python files to remove PolicyVizMixin and PolicyStateRecorder references."""
    try:
        content = file_path.read_text(errors="ignore")

        has_mixin = "PolicyVizMixin" in content
        has_recorder = "PolicyStateRecorder" in content

        if not has_mixin and not has_recorder:
            return

        print(f"Cleaning viz mixin from: {file_path.relative_to(get_project_root())}")

        content = re.sub(
            r"(?m)^(\s*from\s+logic\.src\.tracking\.viz_mixin\s+import\s+.*)$",
            r"from typing import Any  # AUTO-REPLACED",
            content,
        )

        content = re.sub(r"Optional\[PolicyStateRecorder\]", "Optional[Any]", content)
        content = re.sub(r"PolicyStateRecorder", "Any", content)

        affected_classes: list = []

        def repl_class(match):
            class_name = match.group(1)
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(",")]
            parents = [p for p in parents if p != "PolicyVizMixin"]
            affected_classes.append(class_name)
            if parents:
                return f"class {class_name}({', '.join(parents)}):"
            return f"class {class_name}:"

        content = re.sub(r"class\s+(\w+)\(([^)]*PolicyVizMixin[^)]*)\):", repl_class, content)

        def repl_class_single(match):
            class_name = match.group(1)
            affected_classes.append(class_name)
            return f"class {class_name}:"

        content = re.sub(r"class\s+(\w+)\(PolicyVizMixin\):", repl_class_single, content)

        dummy_method = "def _viz_record(self, **kwargs: Any) -> None:  # noqa: B027\n    pass"
        for class_name in affected_classes:
            content = append_to_class_body(content, class_name, dummy_method)

        content = fix_empty_try_blocks(content)
        file_path.write_text(content)
    except Exception as e:
        print(f"Error cleaning viz mixin from {file_path}: {e}")


def patch_specific_files(file_path: Path):
    """Specific clean replacements for key pipeline/dispatch files."""
    name = file_path.name
    try:
        content = file_path.read_text(errors="ignore")

        if name == "engine.py":
            print(f"Patching engine.py: {file_path.relative_to(get_project_root())}")
            content = re.sub(
                r"(?m)^(\s*import\s+logic\.src\.tracking\s+as\s+wst.*)$",
                r"# \1  # AUTO-REMOVED",
                content,
            )
            pattern = r"(?s)try:\s*\n\s*from logic\.src\.tracking\.integrations\.zenml_bridge import configure_zenml_stack\s*\nexcept ImportError:\s*\n\s*configure_zenml_stack = None\s*# type: ignore\[assignment\]"
            content = re.sub(
                pattern,
                "# from logic.src.tracking.integrations.zenml_bridge import configure_zenml_stack  # AUTO-REMOVED",
                content,
            )
            content = re.sub(r"run\s*:\s*wst\.Run", "run: Any", content)
            content += """

# Dummy tracking fallbacks to avoid E402 and NameErrors
class DummyRun:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def add_sink(self, *args): pass
    def log_dataset_event(self, *args, **kwargs): pass
    def watch_file(self, *args): pass
    def log_artifact(self, *args, **kwargs): pass
    def log_metric(self, *args, **kwargs): pass
    def log_params(self, *args): pass
    @property
    def run_id(self): return "dummy_run"

class DummyTracker:
    def start_run(self, *args, **kwargs): return DummyRun()

class DummyWst:
    Run = DummyRun
    @staticmethod
    def init(*args, **kwargs): return DummyTracker()

wst = DummyWst
configure_zenml_stack = None
"""

        elif name in ("zenml_train_pipeline.py", "zenml_eval_pipeline.py", "zenml_sim_pipeline.py"):
            print(f"Patching {name}: {file_path.relative_to(get_project_root())}")
            content = re.sub(
                r"(?m)^(\s*from\s+logic\.src\.tracking\.integrations\.zenml_bridge\s+import\s+ZenMLBridge.*)$",
                r"# \1  # AUTO-REMOVED",
                content,
            )
            content += """

# Dummy ZenMLBridge fallback to avoid E402 and NameErrors
class ZenMLBridge:
    pass
"""

        elif name == "trainer.py":
            print(f"Patching trainer.py: {file_path.relative_to(get_project_root())}")
            content = re.sub(
                r"(?m)^(\s*from\s+logic\.src\.tracking\.integrations\.lightning\s+import\s+TrackingCallback.*)$",
                r"# \1  # AUTO-REMOVED",
                content,
            )
            content += """

# Dummy TrackingCallback fallback to avoid E402 and NameErrors
class TrackingCallback(pl.Callback):
    def __init__(self, *args, **kwargs): pass
"""

        elif name == "logging.py":
            print(f"Patching logging.py: {file_path.relative_to(get_project_root())}")
            content = re.sub(
                r"(?m)^(\s*from\s+logic\.src\.tracking\.integrations\.simulation\s+import\s+get_sim_tracker.*)$",
                r"# \1  # AUTO-REMOVED",
                content,
            )
            content += """

# Dummy get_sim_tracker fallback to avoid E402 and NameErrors
def get_sim_tracker(*args, **kwargs):
    return None
"""

        elif name == "finishing.py":
            print(f"Patching finishing.py: {file_path.relative_to(get_project_root())}")
            pattern = r"(?s)try:\s*\n\s*from logic\.src\.tracking\.core\.run import get_active_run\s*\n\s*from logic\.src\.tracking\.integrations\.simulation import get_sim_tracker\s*\nexcept ImportError:\s*\n\s*get_active_run = None\s*# type: ignore\[assignment,misc\]\s*\n\s*get_sim_tracker = None\s*# type: ignore\[assignment,misc\]"
            content = re.sub(
                pattern,
                "# from logic.src.tracking.core.run import get_active_run  # AUTO-REMOVED\n# from logic.src.tracking.integrations.simulation import get_sim_tracker  # AUTO-REMOVED",
                content,
            )
            content = content.replace("run = get_active_run()", "run = None  # AUTO-REMOVED")
            content += """

# Dummy tracking fallbacks to avoid E402 and NameErrors
get_active_run = None
get_sim_tracker = None
"""

        elif name == "running.py":
            print(f"Patching running.py: {file_path.relative_to(get_project_root())}")
            pattern = r"(?s)try:\s*\n\s*from logic\.src\.tracking\.integrations\.data_lineage import DataLineageCallback\s*\nexcept ImportError:\s*\n\s*DataLineageCallback = None\s*# type: ignore\[assignment,misc\]"
            content = re.sub(
                pattern,
                "# from logic.src.tracking.integrations.data_lineage import DataLineageCallback  # AUTO-REMOVED",
                content,
            )
            content += """

# Dummy DataLineageCallback fallback to avoid E402 and NameErrors
DataLineageCallback = None
"""

        elif name == "hydra_dispatch.py":
            print(f"Patching hydra_dispatch.py: {file_path.relative_to(get_project_root())}")
            content = re.sub(
                r"(?m)^(\s*import\s+logic\.src\.tracking\s+as\s+wst.*)$",
                r"# \1  # AUTO-REMOVED",
                content,
            )
            content = re.sub(
                r"(?m)^(\s*from\s+logic\.src\.tracking\.profiling\s+import\s+start_global_profiling,\s*stop_global_profiling.*)$",
                r"# \1  # AUTO-REMOVED",
                content,
            )
            content += """

# Dummy tracking fallbacks to avoid E402 and NameErrors
class DummyRun:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def add_sink(self, *args): pass
    def log_dataset_event(self, *args, **kwargs): pass
    def watch_file(self, *args): pass
    def log_artifact(self, *args, **kwargs): pass
    def log_metric(self, *args, **kwargs): pass
    def log_params(self, *args): pass
    def flush(self, *args): pass
    def set_tag(self, *args, **kwargs): pass
    @property
    def run_id(self): return "dummy_run"

class DummyTracker:
    def start_run(self, *args, **kwargs): return DummyRun()

class DummyWst:
    Run = DummyRun
    @staticmethod
    def init(*args, **kwargs): return DummyTracker()
    @staticmethod
    def get_active_run(*args, **kwargs): return None

wst = DummyWst
def start_global_profiling(*args, **kwargs): pass
def stop_global_profiling(*args, **kwargs): pass
"""

        elif name == "parser_dispatch.py":
            print(f"Patching parser_dispatch.py: {file_path.relative_to(get_project_root())}")
            content = re.sub(
                r"(?m)^(\s*from\s+logic\.src\.tracking\.profiling\.profiler\s+import\s+start_global_profiling.*)$",
                r"# \1  # AUTO-REMOVED",
                content,
            )
            content += """

# Dummy tracking fallbacks to avoid E402 and NameErrors
def start_global_profiling(*args, **kwargs): pass
"""

        file_path.write_text(content)
    except Exception as e:
        print(f"Error patching {file_path}: {e}")


def comment_tracking_imports(file_path: Path):
    """Comment out all tracking imports and get_active_run usages."""
    try:
        content = file_path.read_text(errors="ignore")

        if "logic.src.tracking" not in content:
            return

        print(f"Removing tracking imports from: {file_path.relative_to(get_project_root())}")

        needs_dummy_hook = False
        if re.search(r"from logic\.src\.tracking\.hooks(?:\.attention_hooks)?\s+import\s+add_attention_hooks", content):
            needs_dummy_hook = True
            content = re.sub(
                r"from logic\.src\.tracking\.hooks(?:\.attention_hooks)?\s+import\s+add_attention_hooks",
                "# from logic.src.tracking.hooks.attention_hooks import add_attention_hooks  # AUTO-REMOVED",
                content,
            )

        content = re.sub(r"(?m)^(\s*from\s+logic\.src\.tracking\b.*)$", r"# \1  # AUTO-REMOVED", content)
        content = re.sub(r"(?m)^(\s*import\s+logic\.src\.tracking\b.*)$", r"# \1  # AUTO-REMOVED", content)

        content = re.sub(r"run\s*=\s*get_active_run\(\)", "run = None  # AUTO-REMOVED", content)
        content = re.sub(r"_active_run\s*=\s*wst\.get_active_run\(\)", "_active_run = None  # AUTO-REMOVED", content)
        content = re.sub(r"_tracker\s*=\s*wst\.get_tracker\(\)", "_tracker = None  # AUTO-REMOVED", content)

        if needs_dummy_hook:
            content += '\n\n# Dummy fallback for removed attention hooks\ndef add_attention_hooks(model):\n    return {"weights": [], "masks": [], "handles": []}\n'

        content = fix_empty_try_blocks(content)
        file_path.write_text(content)
    except Exception as e:
        print(f"Error commenting tracking imports in {file_path}: {e}")


_LOG_UTILS_STUBS = """
# Dummy log_utils fallbacks (tracking.logging removed)
def log_values(*args, **kwargs): pass
def log_epoch(*args, **kwargs): pass
def get_loss_stats(*args, **kwargs): return {}
def setup_system_logger(*args, **kwargs): pass
def sort_log(*args, **kwargs): return []
def log_to_json(*args, **kwargs): pass
def log_to_json2(*args, **kwargs): pass
def log_to_pickle(*args, **kwargs): pass
def update_log(*args, **kwargs): pass
def load_log_dict(*args, **kwargs): return {}
def output_stats(*args, **kwargs): pass
def runs_per_policy(*args, **kwargs): return {}
def final_simulation_summary(*args, **kwargs): pass
def display_simulation_summary_table(*args, **kwargs): pass
def display_per_policy_simulation_summary(*args, **kwargs): pass
def send_daily_output_to_gui(*args, **kwargs): pass
def send_final_output_to_gui(*args, **kwargs): pass
def update_policy_log_section(*args, **kwargs): pass
"""

_LOGGER_WRITER_STUBS = """
# Dummy LoggerWriter / setup stubs (tracking.logging removed)
class LoggerWriter:
    def __init__(self, *args, **kwargs): pass
    def write(self, msg): pass
    def flush(self): pass

def setup_logger_redirection(*args, **kwargs): pass
def setup_system_logger(*args, **kwargs): pass
"""


def replace_logger_calls_with_print(file_path: Path):
    """Replace get_pylogger imports/assignments with print()-based logging."""
    try:
        content = file_path.read_text(errors="ignore")

        if "logic.src.tracking.logging" not in content:
            return

        print(f"Replacing logger calls with print() in: {file_path.relative_to(get_project_root())}")

        var_names = set(re.findall(r"^(\w+)\s*=\s*get_pylogger\(", content, re.MULTILINE))

        needs_log_utils = bool(re.search(r"logic\.src\.tracking\.logging\.log_utils", content))
        needs_logger_writer = bool(
            re.search(r"logic\.src\.tracking\.logging\.logger_writer", content) or "LoggerWriter" in content
        )

        content = re.sub(r"(?m)^(\s*from\s+logic\.src\.tracking\.logging\b.*)$", r"# \1  # AUTO-REMOVED", content)
        content = re.sub(r"(?m)^(\s*\w+\s*=\s*get_pylogger\([^)]*\).*)$", r"# \1  # AUTO-REMOVED", content)

        log_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
        for var in var_names:
            for level in log_levels:
                content = re.sub(rf"(?m)^(\s*){re.escape(var)}\.{level}\(", r"\1print(", content)

        stubs_to_add = ""
        if needs_log_utils:
            stubs_to_add += _LOG_UTILS_STUBS
        if needs_logger_writer:
            stubs_to_add += _LOGGER_WRITER_STUBS

        if stubs_to_add:
            content += stubs_to_add

        file_path.write_text(content)
    except Exception as e:
        print(f"Error replacing logger calls in {file_path}: {e}")


def remove_tracking_logging(file_path: Path):
    """Remove the tracking.logging module directory and comment out its uses across the codebase."""
    root = get_project_root()
    tracking_logging_dir = root / "logic/src/tracking/logging"
    if tracking_logging_dir.exists():
        remove_path(tracking_logging_dir)

    test_logging_dir = root / "logic/test/unit/utils/logging"
    if test_logging_dir.exists():
        remove_path(test_logging_dir)

    try:
        if not file_path.exists():
            return
        content = file_path.read_text(errors="ignore")

        pattern = r"logic\.src\.tracking\.logging\b"
        if not re.search(pattern, content):
            return

        print(f"Commenting out tracking.logging imports in: {file_path.relative_to(get_project_root())}")

        content = re.sub(
            r"(?m)^(\s*from\s+logic\.src\.tracking\.logging\b.*)$",
            r"# \1  # AUTO-REMOVED",
            content,
        )
        content = re.sub(
            r"(?m)^(\s*import\s+logic\.src\.tracking\.logging\b.*)$",
            r"# \1  # AUTO-REMOVED",
            content,
        )
        file_path.write_text(content)
    except Exception as e:
        print(f"Error removing tracking.logging in {file_path}: {e}")


def clean_plotting_submodule(file_path: Path):
    """Clean the newly organized visualization and plotting submodule located at logic/src/utils/plotting."""
    root = get_project_root()
    plotting_dir = root / "logic/src/utils/plotting"
    if plotting_dir.exists():
        remove_path(plotting_dir)

    test_plotting_dir = root / "logic/test/unit/utils/plotting"
    if test_plotting_dir.exists():
        remove_path(test_plotting_dir)

    try:
        if not file_path.exists():
            return
        content = file_path.read_text(errors="ignore")

        if "logic.src.utils.plotting" not in content:
            return

        print(f"Commenting out plotting imports in: {file_path.relative_to(get_project_root())}")

        content = re.sub(
            r"(?m)^(\s*from\s+logic\.src\.utils\.plotting\b.*)$",
            r"# \1  # AUTO-REMOVED",
            content,
        )
        content = re.sub(
            r"(?m)^(\s*import\s+logic\.src\.utils\.plotting\b.*)$",
            r"# \1  # AUTO-REMOVED",
            content,
        )
        file_path.write_text(content)
    except Exception as e:
        print(f"Error cleaning plotting submodule in {file_path}: {e}")


def comment_justfile(justfile_path: Path):
    """Comment out database commands in justfile."""
    if not justfile_path.exists():
        return
    print("Commenting out database targets in justfile...")
    content = justfile_path.read_text(errors="ignore")

    targets = ["db-inspect", "db-clean", "db-compact", "db-prune", "db-export", "db-stats", "db-metrics"]
    for target in targets:
        pattern = rf"(?m)^({target}(?:\s+[^:\n]+)?:[ \t]*\n(?:\s+.*\n)*)"

        def repl(match):
            lines = match.group(1).splitlines()
            commented = "\n".join(f"# {line}" for line in lines)
            return commented + "\n"

        content = re.sub(pattern, repl, content)

    justfile_path.write_text(content)


def clean_test_functions(file_path: Path):
    """Specific cleanup for logic/test/unit/utils/functions/test_functions.py."""
    if not file_path.exists():
        return
    print(f"Cleaning test_functions.py: {file_path}")
    content = file_path.read_text(errors="ignore")
    content = re.sub(
        r"from logic\.src\.tracking\.hooks\.attention_hooks import add_attention_hooks",
        "# from logic.src.tracking.hooks.attention_hooks import add_attention_hooks  # AUTO-REMOVED",
        content,
    )

    target = """    def test_add_attention_hooks(self):
        \"\"\"Test adding attention hooks to a model.\"\"\"
        model = MagicMock()
        mock_layer = MagicMock()
        mock_layer.att.module = MagicMock()
        model.layers = [mock_layer]
        hook_data = add_attention_hooks(model)
        assert "weights" in hook_data
        assert len(hook_data["handles"]) == 1"""
    content = content.replace(target, "")
    file_path.write_text(content)


_SPECIFIC_PATCH_NAMES = frozenset(
    {
        "engine.py",
        "zenml_train_pipeline.py",
        "zenml_eval_pipeline.py",
        "zenml_sim_pipeline.py",
        "trainer.py",
        "logging.py",
        "finishing.py",
        "running.py",
        "hydra_dispatch.py",
        "parser_dispatch.py",
    }
)


def main():
    root = get_project_root()
    print(f"Project root is: {root}")

    # 1. Delete tracking folders and files
    to_delete = [
        root / "logic/test/unit/tracking",
        root / "logic/src/tracking/core",
        root / "logic/src/tracking/database",
        root / "logic/src/tracking/helpers",
        root / "logic/src/tracking/hooks",
        root / "logic/src/tracking/integrations",
        root / "logic/src/tracking/logging",
        root / "logic/src/tracking/profiling",
        root / "logic/src/tracking/validation",
        root / "logic/src/tracking/viz_mixin.py",
        root / "logic/configs/tracking",
        root / "logic/test/unit/utils/logging",
        root / "logic/src/utils/plotting",
        root / "logic/test/unit/utils/plotting",
    ]

    for path in to_delete:
        remove_path(path)

    # 2. Overwrite logic/src/tracking/__init__.py to be completely empty
    tracking_init = root / "logic/src/tracking/__init__.py"
    if tracking_init.exists():
        print(f"Overwriting {tracking_init} with empty content")
        tracking_init.write_text("")

    # 3. Recursively process all Python files in logic/
    logic_dir = root / "logic"
    for p in logic_dir.glob("**/*.py"):
        if "remove_tracking.py" in p.name:
            continue

        remove_viz_mixin_from_file(p)
        replace_logger_calls_with_print(p)

        if p.name in _SPECIFIC_PATCH_NAMES:
            patch_specific_files(p)
        else:
            comment_tracking_imports(p)

        remove_tracking_logging(p)
        clean_plotting_submodule(p)

    clean_test_functions(root / "logic/test/unit/utils/functions/test_functions.py")

    # 4. Modify justfile
    comment_justfile(root / "justfile")

    print("\n--- Tracking Cleanup Complete! ---")


if __name__ == "__main__":
    main()
