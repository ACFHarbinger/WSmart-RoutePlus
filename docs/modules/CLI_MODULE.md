# Command-Line Interface & Argument Parsing

> **Status (Q-22):** narrative guide. The API reference is generated into
> `docs/api/` by `logic/gen/gen_api_docs.py`; this file is not the source of
> truth for signatures and module inventories.


**Module**: `logic/controllers/cli`
**Purpose**: Comprehensive technical specification of the modular CLI framework—handling argument parsing, validation, and command orchestration.
**Version**: 3.0
**Last Updated**: February 2026

---

## Table of Contents

1.  [**Overview**](#1-overview)
2.  [**Module Organization**](#2-module-organization)
3.  [**Base Parser Components**](#3-base-parser-components)
4.  [**Command Parsers**](#4-command-parsers)
5.  [**Central Registry**](#5-central-registry)
6.  [**Usage Examples**](#6-usage-examples)
7.  [**Custom Actions**](#7-custom-actions)
8.  [**Extending the CLI**](#8-extending-the-cli)
9.  [**Best Practices**](#9-best-practices)
10. [**Quick Reference**](#10-quick-reference)

---

## 1. Overview

The `logic/controllers/cli` module provides a comprehensive command-line interface framework for WSmart+ Route. It uses a modular architecture built on top of Python's `argparse`, with custom parser extensions and specialized action classes.

### Key Features

- **Modular Command Structure**: Subcommand-based architecture for clean separation
- **Custom Argument Parser**: Enhanced `ConfigsParser` with improved nargs handling
- **Type-Safe Actions**: Custom action classes for validation and transformation
- **Extensible Registry**: Centralized parser registration for easy expansion
- **Comprehensive Validation**: Command-specific validation functions
- **Integration with Hydra**: Seamless connection to config system

### Architecture Principles

The CLI follows a **hierarchical subcommand pattern**:

```
main.py
└── logic.controllers.cli.parse_params()
    ├── ConfigsParser (base parser)
    ├── Registry (command registration)
    └── Command-specific parsers
        ├── file_system (update, delete, cryptography)
        ├── test_suite (module, class, test selection)
        └── benchmark (subset, device, output)
```

### Design Goals

1. **Modularity**: Each command has its own parser module
2. **Extensibility**: Easy to add new commands and arguments
3. **Validation**: Command-specific validation at parse time
4. **Type Safety**: Custom actions ensure correct types
5. **User Experience**: Clear help messages and error reporting

---

## 2. Module Organization

### Directory Structure

```
logic/controllers/cli/
├── __init__.py              # Unified entry point (parse_params)
├── registry.py              # Central parser registration
│
├── base/                    # Base parser components
│   ├── __init__.py
│   ├── parser.py            # ConfigsParser class
│   ├── lowercase_action.py  # LowercaseAction
│   ├── store_dict_key.py    # StoreDictKeyPair
│   └── update_function_factory.py  # UpdateFunctionMapActionFactory
│
├── bench_parser.py      # Benchmark suite arguments
├── ts_parser.py             # Test suite arguments
└── fs_parser.py             # File system operations
```

### Module Exports

The `__init__.py` exports the main parse function and base components:

```python
from logic.controllers.cli import (
    parse_params,           # Main parsing function
    ConfigsParser,          # Custom parser class
    LowercaseAction,        # Lowercase conversion action
    StoreDictKeyPair,       # Key=value parsing action
    UpdateFunctionMapActionFactory,  # Function mapping factory
)
```

### Entry Point Flow

```python
# main.py
from logic.controllers.cli import parse_params

# Parse command and options
command, opts = parse_params()

# Dispatch to appropriate handler
if command == "test_suite":
    run_test_suite(opts)
elif command == "benchmark":
    run_benchmarks(opts)
elif command == ("file_system", "update"):
    update_file_system_entries(opts)
# ... etc
```

---

## 3. Base Parser Components

**Directory**: `logic/controllers/cli/base/`
**Purpose**: Core parser infrastructure and custom action classes

### ConfigsParser

**File**: `base/parser.py`

Custom `ArgumentParser` with enhanced nargs handling and advanced parsing capabilities.

```python
class ConfigsParser(argparse.ArgumentParser):
    """Custom ArgumentParser to handle string-based nargs correctly.

    This class extends ArgumentParser to provide additional functionality for
    processing command-line arguments, especially those involving sequence types
    and nested parser structures.
    """

    def _str_to_nargs(self, nargs: Union[str, Sequence]) -> Union[str, Sequence]:
        """Convert a single string argument into a list if expected by nargs."""
        ...

    def _process_args(self, namespace: argparse.Namespace) -> None:
        """Post-process arguments in the namespace, handling special narg conversions."""
        ...

    def parse_command(self, args: Optional[Sequence[str]] = None) -> Optional[str]:
        """Parse only the command from the arguments."""
        ...

    def parse_process_args(
        self, args: Optional[List[str]] = None, command: Optional[str] = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Parse arguments and returns the command and options dictionary.

        This method performs a more complex parsing that handles space-separated
        strings intended for multi-value arguments (nargs) by splitting them
        before full parsing.

        Returns:
            A tuple containing (command_name, dictionary_of_filtered_arguments).
        """
        ...

    def error_message(self, message: str, print_help: bool = True) -> None:
        """Print error message and optionally help."""
        ...
```

**Key Features**

1. **Enhanced nargs Handling**: Properly splits space-separated strings into lists
2. **Command Extraction**: Can extract just the command without full parsing
3. **Dict-Based Output**: Returns clean dictionaries instead of Namespace objects
4. **Custom Error Messages**: Better error reporting with optional help display

**Usage Example**

```python
from logic.controllers.cli.base import ConfigsParser

# Create parser
parser = ConfigsParser(description="My CLI Tool")
subparsers = parser.add_subparsers(dest="command")

# Add subcommand
train_parser = subparsers.add_parser("train")
train_parser.add_argument("--epochs", type=int, default=100)

# Parse with enhanced features
command, opts = parser.parse_process_args()
# command: "train"
# opts: {"epochs": 100}
```

### LowercaseAction

**File**: `base/lowercase_action.py`

Custom action to automatically convert argument values to lowercase.

```python
class LowercaseAction(argparse.Action):
    """Action to convert argument value to lowercase."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        """Invoke action to lowercase input values."""
        if values is not None:
            values = str(values).lower()
        setattr(namespace, self.dest, values)
```

**Usage Example**

```python
parser.add_argument(
    "--app_style",
    action=LowercaseAction,
    type=str,
    default="fusion",
    help="Style for the GUI application"
)

# Input: --app_style FUSION
# Result: opts["app_style"] = "fusion"
```

**Use Cases**

- GUI style names (fusion, windows, macintosh)
- Algorithm names (ppo, reinforce, pomo)
- Problem names (vrpp, wcvrp)
- Any case-insensitive string matching

### StoreDictKeyPair

**File**: `base/store_dict_key.py`

Custom action to parse `key=value` pairs into a dictionary.

```python
class StoreDictKeyPair(argparse.Action):
    """Custom action to parse key=value into a dictionary."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: Optional[str] = None,
    ) -> None:
        """Invoke action to parse key=value strings into a dictionary.

        Raises:
            argparse.ArgumentError: If a string does not follow the 'key=value' format.
        """
        my_dict = {}
        if values is None:
            values = []
        elif isinstance(values, str):
            values = [values]

        for kv in values:
            if "=" in kv:
                k, v = kv.split("=", 1)
                my_dict[k] = v
            else:
                raise argparse.ArgumentError(
                    self, f"Could not parse argument '{kv}' as key=value format"
                )
        setattr(namespace, self.dest, my_dict)
```

**Usage Example**

```python
parser.add_argument(
    "--params",
    action=StoreDictKeyPair,
    nargs="+",
    help="Additional parameters as key=value pairs"
)

# Input: --params lr=0.001 batch_size=256 dropout=0.1
# Result: opts["params"] = {"lr": "0.001", "batch_size": "256", "dropout": "0.1"}
```

**Use Cases**

- Hyperparameter overrides
- Environment variables
- Custom configuration parameters
- Dynamic key-value settings

### UpdateFunctionMapActionFactory

**File**: `base/update_function_factory.py`

Factory function that creates custom actions for mapping string inputs to predefined functions.

```python
def UpdateFunctionMapActionFactory(inplace: bool = False) -> type:
    """Factory for mapping string update functions.

    Args:
        inplace: Whether to use the in-place operation map or the stats function map.

    Returns:
        A class type for UpdateFunctionMapAction.
    """

    class UpdateFunctionMapAction(argparse.Action):
        """Action that maps input strings to update functions."""

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Any,
            option_string: Optional[str] = None,
        ) -> None:
            """Invoke action: map string to update function.

            Raises:
                ValueError: If the input value does not map to a valid function.
            """
            if values is not None:
                if self.inplace:
                    values = OPERATION_MAP.get(str(values).replace(" ", ""), None)
                else:
                    values = STATS_FUNCTION_MAP.get(str(values).replace(" ", ""), None)
            if values is None:
                raise ValueError(f"Invalid update function: {values}")
            setattr(namespace, self.dest, values)

    return UpdateFunctionMapAction
```

**Usage Example**

```python
from logic.controllers.cli.base import UpdateFunctionMapActionFactory

# For in-place operations (OPERATION_MAP)
parser.add_argument(
    "--update_operation",
    type=str,
    action=UpdateFunctionMapActionFactory(inplace=True),
    help="Operation to update file values"
)

# For statistical functions (STATS_FUNCTION_MAP)
parser.add_argument(
    "--stats_function",
    type=str,
    action=UpdateFunctionMapActionFactory(inplace=False),
    help="Statistical function to apply"
)
```

**Function Maps**

From `logic.src.constants.system`:

```python
OPERATION_MAP = {
    "+=": lambda x, y: x + y,
    "-=": lambda x, y: x - y,
    "*=": lambda x, y: x * y,
    "/=": lambda x, y: x / y,
    "//=": lambda x, y: x // y,
    "%=": lambda x, y: x % y,
    "**=": lambda x, y: x ** y,
    # ... 40+ operators
}
```

From `logic.src.constants.stats`:

```python
STATS_FUNCTION_MAP = {
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
    "variance": statistics.variance,
    # ... more functions
}
```

---

## 4. Command Parsers

Each command has its own parser module with an `add_*_args` function and a `validate_*_args` function.

### File System Parser

**File**: `fs_parser.py`

Handles file system operations: update, delete, and cryptography.

#### Main Function

```python
def add_files_args(parser):
    """Adds all arguments related to file system operations (as subparsers)."""
    files_subparsers = parser.add_subparsers(
        help="file system command",
        dest="fs_command",
        required=True
    )

    # Update file system entries
    update_parser = files_subparsers.add_parser("update", help="Update file system entries")
    add_files_update_args(update_parser)

    # Delete file system entries
    delete_parser = files_subparsers.add_parser("delete", help="Delete file system entries")
    add_files_delete_args(delete_parser)

    # Cryptography
    crypto_parser = files_subparsers.add_parser(
        "cryptography",
        help="Perform cryptographic operations on file system entries"
    )
    add_files_crypto_args(crypto_parser)
```

#### Update Arguments

```python
def add_files_update_args(parser):
    """Adds file system 'update' sub-command arguments."""
    parser.add_argument(
        "--target_entry", type=str,
        help="Path to the file system entry we want to update"
    )
    parser.add_argument(
        "--output_key", type=str, default=None,
        help="Key of the values we want to update in the files"
    )
    parser.add_argument(
        "--filename_pattern", type=str, default=None,
        help="Pattern to match names of files to update"
    )
    parser.add_argument(
        "--update_operation", type=str, default=None,
        action=UpdateFunctionMapActionFactory(inplace=True),
        help="Operation to update the file values"
    )
    parser.add_argument(
        "--update_value", type=float, default=0.0,
        help="Value for the update operation"
    )
    parser.add_argument(
        "--update_preview", action="store_true",
        help="Preview how files/directories will look like after the update"
    )
    parser.add_argument(
        "--stats_function", type=str, default=None,
        action=UpdateFunctionMapActionFactory(),
        help="Function to perform over the file values"
    )
    parser.add_argument(
        "--output_filename", type=str, default=None,
        help="Name of the file we want to save the output values to"
    )
    parser.add_argument(
        "--input_keys", type=str, default=(None, None), nargs="*",
        help="Key(s) of the values we want to use as input"
    )
```

**Usage Examples**

```bash
# Update file values with operation
python main.py file_system update \
    --target_entry data/results.json \
    --output_key "reward" \
    --update_operation "+=" \
    --update_value 10.0

# Preview update
python main.py file_system update \
    --target_entry data/ \
    --filename_pattern "*.json" \
    --output_key "cost" \
    --update_operation "*=" \
    --update_value 0.9 \
    --update_preview

# Compute statistics
python main.py file_system update \
    --target_entry data/results.json \
    --output_key "rewards" \
    --stats_function "mean" \
    --output_filename summary.json
```

#### Delete Arguments

```python
def add_files_delete_args(parser):
    """Adds file system 'delete' sub-command arguments."""
    parser.add_argument(
        "--wandb", action="store_false",
        help="Flag to delete the train wandb log directory"
    )
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--log", action="store_false")
    parser.add_argument("--output_dir", default="model_weights")
    parser.add_argument("--output", action="store_false")
    parser.add_argument("--data_dir", default="datasets")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--eval_dir", default="results")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test_dir", default="output")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_checkpoint_dir", default="temp")
    parser.add_argument("--test_checkpoint", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--delete_preview", action="store_true",
        help="Preview which files/directories will be removed"
    )
```

**Usage Examples**

```bash
# Delete datasets directory
python main.py file_system delete --data

# Delete evaluation results
python main.py file_system delete --eval

# Delete multiple directories
python main.py file_system delete --data --eval --cache

# Preview deletion
python main.py file_system delete --data --delete_preview
```

#### Cryptography Arguments

```python
def add_files_crypto_args(parser):
    """Adds file system 'cryptography' sub-command arguments."""
    parser.add_argument(
        "--symkey_name", type=str, default=None,
        help="Name of the key for the files"
    )
    parser.add_argument(
        "--env_file", type=str, default="vars.env",
        help="Name of the file that contains environment variables"
    )
    parser.add_argument("--salt_size", type=int, default=16)
    parser.add_argument("--key_length", type=int, default=32)
    parser.add_argument("--hash_iterations", type=int, default=100_000)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
```

**Usage Examples**

```bash
# Generate encryption key
python main.py file_system cryptography \
    --symkey_name my_secret_key \
    --salt_size 16 \
    --key_length 32

# Encrypt file
python main.py file_system cryptography \
    --symkey_name my_secret_key \
    --input_path data/sensitive.json \
    --output_path data/sensitive.enc
```

#### Validation

```python
def validate_file_system_args(args):
    """Validates and post-processes arguments for file_system.
    Returns a tuple: (fs_command, validated_args)
    """
    args = args.copy()
    fs_comm = args.pop("fs_command", None)
    if fs_comm not in FS_COMMANDS:
        raise argparse.ArgumentError(
            None, "ERROR: unknown File System (inner) command " + str(fs_comm)
        )

    # Mutual exclusivity check
    assert not (
        "stats_function" in args and args["stats_function"] is not None
    ) or not (
        "update_operation" in args and args["update_operation"] is not None
    ), "'update_operation' and 'stats_function' arguments are mutually exclusive"

    return fs_comm, args
```

### Test Suite Parser

**File**: `ts_parser.py`

Configures pytest execution with various options.

```python
def add_test_suite_args(parser):
    """Adds all arguments related to the test suite to the given parser."""
    # Test selection
    parser.add_argument(
        "-m", "--module", nargs="+",
        choices=list(TEST_MODULES.keys()),
        help="Specific test module(s) to run"
    )
    parser.add_argument(
        "-c", "--class", dest="test_class",
        help="Specific test class to run"
    )
    parser.add_argument(
        "-t", "--test", dest="test_method",
        help="Specific test method to run"
    )
    parser.add_argument(
        "-k", "--keyword",
        help="Run tests matching the given keyword expression"
    )
    parser.add_argument(
        "--markers",
        help="Run tests matching the given marker expression"
    )

    # Test execution options
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--ff", "--failed-first", dest="failed_first", action="store_true")
    parser.add_argument("-x", "--exitfirst", dest="maxfail", action="store_const", const=1)
    parser.add_argument("--maxfail", type=int)
    parser.add_argument(
        "--tb", choices=["auto", "long", "short", "line", "native", "no"],
        default="auto", help="Traceback print mode"
    )
    parser.add_argument(
        "--capture", choices=["auto", "no", "sys", "fd"],
        default="auto", help="Capture mode for output"
    )
    parser.add_argument(
        "-n", "--parallel", action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )

    # Information commands
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--list-tests", action="store_true")
    parser.add_argument("--test-dir", default="tests")
```

**Usage Examples**

```bash
# Run all tests
python main.py test_suite

# Run specific module
python main.py test_suite -m test_models

# Run multiple modules
python main.py test_suite -m test_models test_policies

# Run specific test class
python main.py test_suite -m test_models -c TestAttentionModel

# Run specific test method
python main.py test_suite -m test_models -c TestAttentionModel -t test_forward_pass

# Run with keyword filter
python main.py test_suite -k "encoder"

# Run with coverage
python main.py test_suite --coverage

# Run in parallel
python main.py test_suite -n

# Run with verbose output
python main.py test_suite -v

# Exit on first failure
python main.py test_suite -x

# List available modules
python main.py test_suite -l

# List all tests
python main.py test_suite --list-tests
```

### Benchmark Parser

**File**: `bench_parser.py`

Configures performance benchmark execution.

```python
def add_benchmark_args(parser: ConfigsParser) -> None:
    """Adds arguments for the benchmark command."""
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "neural", "ls", "solvers"],
        help="Subset of benchmarks to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, or auto)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON"
    )

def validate_benchmark_args(opts: Dict[str, Any]) -> Dict[str, Any]:
    """Validates benchmark arguments."""
    # Currently no complex validation needed
    return opts
```

**Usage Examples**

```bash
# Run all benchmarks
python main.py benchmark

# Run neural model benchmarks only
python main.py benchmark --subset neural

# Run local search benchmarks
python main.py benchmark --subset ls

# Run on CPU
python main.py benchmark --device cpu

# Save results to file
python main.py benchmark --output benchmarks/results.json

# Combine options
python main.py benchmark --subset solvers --device cuda --output results.json
```

---

## 5. Central Registry

**File**: `registry.py`

Centralized parser registration that creates and configures the main parser with all subcommands.

```python
def get_parser() -> ConfigsParser:
    """Creates and returns the main ConfigsParser with all subcommands registered."""
    parser = ConfigsParser(description="WSmart+ Route Unified CLI Framework")
    subparsers = parser.add_subparsers(
        dest="command",
        help="The command to execute",
        required=True
    )

    # File System
    files_parser = subparsers.add_parser("file_system", help="File system operations")
    add_files_args(files_parser)

    # Test Suite
    ts_parser = subparsers.add_parser("test_suite", help="Run the test suite")
    add_test_suite_args(ts_parser)

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    add_benchmark_args(bench_parser)

    return parser
```

**Registered Commands**

| Command       | Description                | Sub-commands                       |
| ------------- | -------------------------- | ---------------------------------- |
| `file_system` | File system operations     | `update`, `delete`, `cryptography` |
| `test_suite`  | Run test suite             | -                                  |
| `benchmark`   | Run performance benchmarks | -                                  |

---

## 6. Usage Examples

### Main Entry Point

**File**: `__init__.py`

```python
def parse_params():
    """
    Parses arguments, determines the command, and performs necessary validation.
    Returns: (command, validated_opts) where 'command' might be a tuple (comm, inner_comm)
    """
    parser = get_parser()

    try:
        # Parse arguments into a dictionary using the custom handler
        command, opts = parser.parse_process_args()

        # --- COMMAND-SPECIFIC VALIDATION AND route improvement ---
        if command == "file_system":
            # This returns a tuple: (fs_command, validated_opts)
            command, opts = validate_file_system_args(opts)
            command = ("file_system", command)  # Re-wrap for main() function handling
        elif command == "test_suite":
            opts = validate_test_suite_args(opts)
        elif command == "benchmark":
            opts = validate_benchmark_args(opts)
        return command, opts
    except (argparse.ArgumentError, AssertionError) as e:
        parser.error_message(f"Error: {e}", print_help=True)
    except Exception as e:
        parser.error_message(f"An unexpected error occurred: {e}", print_help=False)
```

### Integration in main.py

```python
#!/usr/bin/env python
"""Main Entry Point for the WSmart-Route Application."""

from logic.controllers.cli import parse_params

def main():
    """Main entry point."""
    # Parse CLI arguments
    command, opts = parse_params()

    # Dispatch to appropriate handler
    if command == "test_suite":
        exit_code = run_test_suite(opts)
        sys.exit(exit_code)

    elif command == "benchmark":
        from logic.benchmark.benchmark_suite import run_benchmarks
        run_benchmarks(opts)

    elif isinstance(command, tuple) and command[0] == "file_system":
        fs_command = command[1]
        if fs_command == "update":
            update_file_system_entries(opts)
        elif fs_command == "delete":
            delete_file_system_entries(opts)
        elif fs_command == "cryptography":
            perform_cryptographic_operations(opts)
```

### Complete CLI Examples

**Test Suite Execution**

```bash
# Run all tests
python main.py test_suite

# Run specific module with coverage
python main.py test_suite -m test_models --coverage -v

# Run tests matching keyword
python main.py test_suite -k "attention" --tb=short

# Run failed tests first, stop on first error
python main.py test_suite --ff -x

# Run tests in parallel
python main.py test_suite -n
```

**File System Operations**

```bash
# Update JSON values
python main.py file_system update \
    --target_entry results/metrics.json \
    --output_key reward \
    --update_operation "+=" \
    --update_value 5.0

# Delete training artifacts
python main.py file_system delete \
    --log --output --cache

# Generate encryption key
python main.py file_system cryptography \
    --symkey_name production_key \
    --salt_size 32
```

**Benchmarks**

```bash
# Run all benchmarks
python main.py benchmark

# Neural models only on GPU
python main.py benchmark --subset neural --device cuda

# Save results
python main.py benchmark --output benchmark_results.json
```

---

## 7. Custom Actions

### Creating Custom Actions

Custom actions extend `argparse.Action` to provide specialized behavior:

```python
import argparse
from typing import Any, Optional

class MyCustomAction(argparse.Action):
    """Custom action for specialized processing."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        """Process the argument value."""
        # Custom processing logic
        processed_value = self.process(values)
        setattr(namespace, self.dest, processed_value)

    def process(self, values: Any) -> Any:
        """Implement custom processing."""
        # Transform values
        return transformed_values
```

### Using Custom Actions

```python
parser.add_argument(
    "--my_arg",
    action=MyCustomAction,
    help="Argument with custom processing"
)
```

### Action Factory Pattern

```python
def MyActionFactory(param: bool = False) -> type:
    """Factory for creating custom actions with parameters."""

    class MyAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, **kwargs)
            self.param = param

        def __call__(self, parser, namespace, values, option_string=None):
            # Use self.param in processing
            result = self.process(values, self.param)
            setattr(namespace, self.dest, result)

    return MyAction

# Usage
parser.add_argument(
    "--arg",
    action=MyActionFactory(param=True)
)
```

---

## 8. Extending the CLI

### Adding a New Command

**Step 1**: Create parser module `logic/controllers/cli/new_command_parser.py`

```python
"""New command argument parser."""

def add_new_command_args(parser):
    """Adds arguments for the new command."""
    parser.add_argument(
        "--option1",
        type=str,
        default="default_value",
        help="Description of option1"
    )
    parser.add_argument(
        "--option2",
        type=int,
        default=42,
        help="Description of option2"
    )
    return parser

def validate_new_command_args(args):
    """Validates new command arguments."""
    args = args.copy()
    # Add validation logic
    assert args["option2"] > 0, "option2 must be positive"
    return args
```

**Step 2**: Register in `registry.py`

```python
from logic.controllers.cli.new_command_parser import add_new_command_args

def get_parser() -> ConfigsParser:
    parser = ConfigsParser(description="WSmart+ Route Unified CLI Framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ... existing commands ...

    # New command
    new_parser = subparsers.add_parser("new_command", help="Description of new command")
    add_new_command_args(new_parser)

    return parser
```

**Step 3**: Add validation in `__init__.py`

```python
from logic.controllers.cli.new_command_parser import validate_new_command_args

def parse_params():
    parser = get_parser()
    try:
        command, opts = parser.parse_process_args()

        # ... existing validations ...

        elif command == "new_command":
            opts = validate_new_command_args(opts)

        return command, opts
```

**Step 4**: Implement handler in `main.py`

```python
def handle_new_command(opts):
    """Execute new command logic."""
    print(f"Executing new command with options: {opts}")
    # Implementation here

def main():
    command, opts = parse_params()

    # ... existing command handlers ...

    elif command == "new_command":
        handle_new_command(opts)
```

### Adding Sub-commands

```python
def add_command_args(parser):
    """Adds arguments with sub-commands."""
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Sub-command 1
    sub1 = subparsers.add_parser("sub1", help="First sub-command")
    sub1.add_argument("--arg1", type=str)

    # Sub-command 2
    sub2 = subparsers.add_parser("sub2", help="Second sub-command")
    sub2.add_argument("--arg2", type=int)

    return parser
```

---

## 9. Best Practices

### ✅ Good Practices

**Clear Argument Names**

```python
# ✅ GOOD: Descriptive, unambiguous
parser.add_argument("--model_weights_path", type=str)
parser.add_argument("--training_epochs", type=int)
parser.add_argument("--enable_validation", action="store_true")
```

**Comprehensive Help Messages**

```python
# ✅ GOOD: Detailed help with examples
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to dataset file (e.g., data/train.pkl). Must be a valid pickle file."
)
```

**Sensible Defaults**

```python
# ✅ GOOD: Provide defaults for optional args
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="auto")
```

**Validation Functions**

```python
# ✅ GOOD: Separate validation logic
def validate_args(args):
    """Validate parsed arguments."""
    args = args.copy()

    # Type validation
    assert isinstance(args["epochs"], int), "epochs must be integer"

    # Range validation
    assert args["epochs"] > 0, "epochs must be positive"

    # Mutual exclusivity
    if args["option_a"] and args["option_b"]:
        raise ValueError("option_a and option_b are mutually exclusive")

    return args
```

**Custom Actions for Complex Types**

```python
# ✅ GOOD: Use custom actions for special parsing
parser.add_argument(
    "--params",
    action=StoreDictKeyPair,
    nargs="+",
    help="Parameters as key=value pairs"
)
```

### ❌ Anti-Patterns

**Vague Argument Names**

```python
# ❌ BAD: Unclear what these mean
parser.add_argument("--p", type=str)
parser.add_argument("--data", type=str)  # Too generic
parser.add_argument("--flag", action="store_true")  # What flag?
```

**Missing Help Messages**

```python
# ❌ BAD: No help text
parser.add_argument("--model", type=str)

# ✅ GOOD: Clear help
parser.add_argument(
    "--model",
    type=str,
    help="Model architecture (am, deep_decoder, tam)"
)
```

**No Validation**

```python
# ❌ BAD: No validation, errors occur later
def handle_command(opts):
    epochs = opts["epochs"]  # Could be negative, non-integer, etc.
    train(epochs)

# ✅ GOOD: Validate early
def handle_command(opts):
    assert isinstance(opts["epochs"], int) and opts["epochs"] > 0
    train(opts["epochs"])
```

**Hardcoded Values**

```python
# ❌ BAD: Hardcoded in code
def process_data():
    batch_size = 256  # Should be configurable

# ✅ GOOD: CLI argument
parser.add_argument("--batch_size", type=int, default=256)
```

### Error Handling

**Good Error Messages**

```python
# ✅ GOOD: Informative error messages
try:
    value = int(args["param"])
except ValueError:
    raise ValueError(
        f"Invalid value for --param: '{args['param']}'. "
        f"Expected integer, got {type(args['param']).__name__}"
    )
```

**Graceful Failure**

```python
# ✅ GOOD: Catch and report errors gracefully
try:
    command, opts = parse_params()
except argparse.ArgumentError as e:
    print(f"Argument error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}", file=sys.stderr)
    sys.exit(2)
```

---

## 10. Quick Reference

### Common Imports

```python
# Parse CLI arguments
from logic.controllers.cli import parse_params

# Base components
from logic.controllers.cli.base import (
    ConfigsParser,
    LowercaseAction,
    StoreDictKeyPair,
    UpdateFunctionMapActionFactory
)

# Command parsers
from logic.controllers.cli.bench_parser import add_benchmark_args, validate_benchmark_args
from logic.controllers.cli.fs_parser import add_files_args, validate_file_system_args
from logic.controllers.cli.ts_parser import add_test_suite_args, validate_test_suite_args

# Registry
from logic.controllers.cli.registry import get_parser
```

### Command Summary

| Command       | Description            | Example Usage                                     |
| ------------- | ---------------------- | ------------------------------------------------- |
| `file_system` | File operations        | `python main.py file_system update --target ... ` |
| `test_suite`  | Run tests              | `python main.py test_suite -m test_models`        |
| `benchmark`   | Performance benchmarks | `python main.py benchmark --subset neural`        |
| `trace_policy` | Write one ALNS execution trace | `python main.py trace_policy --problem ptp --graph-size 5 --seed 1 --acceptance oi --output /tmp/alns-trace.json` |
| `list_policy_modules` | Write the policy and model catalog | `python main.py list_policy_modules --output /tmp/catalog.json` |

### File System Sub-commands

| Sub-command    | Description           | Example Usage                                            |
| -------------- | --------------------- | -------------------------------------------------------- |
| `update`       | Update file values    | `... update --output_key reward --update_operation "+="` |
| `delete`       | Delete directories    | `... delete --data --cache`                              |
| `cryptography` | Encryption operations | `... cryptography --symkey_name my_key`                  |

### Custom Actions

| Action                           | Purpose                  | Example                                   |
| -------------------------------- | ------------------------ | ----------------------------------------- |
| `LowercaseAction`                | Convert to lowercase     | `action=LowercaseAction`                  |
| `StoreDictKeyPair`               | Parse key=value pairs    | `action=StoreDictKeyPair, nargs="+"`      |
| `UpdateFunctionMapActionFactory` | Map strings to functions | `action=UpdateFunctionMapActionFactory()` |

### Parser Methods

| Method                 | Returns            | Purpose                          |
| ---------------------- | ------------------ | -------------------------------- |
| `parse_command()`      | `str`              | Extract command name only        |
| `parse_process_args()` | `Tuple[str, Dict]` | Full parsing with dict output    |
| `error_message()`      | `None` (raises)    | Display error with optional help |

### File Locations

| File                              | Lines | Description                     |
| --------------------------------- | ----- | ------------------------------- |
| `__init__.py`                     | 49    | Main entry point (parse_params) |
| `registry.py`                     | 38    | Central parser registration     |
| `base/parser.py`                  | 134   | ConfigsParser class             |
| `base/lowercase_action.py`        | 34    | LowercaseAction                 |
| `base/store_dict_key.py`          | 47    | StoreDictKeyPair                |
| `base/update_function_factory.py` | 75    | UpdateFunctionMapActionFactory  |
| `bench_parser.py`             | 41    | Benchmark arguments             |
| `ts_parser.py`                    | 100   | Test suite arguments            |
| `fs_parser.py`                    | 211   | File system arguments           |

### Related Documentation

- [CONFIGS_MODULE.md](CONFIGS_MODULE.md) - Configuration system (Hydra integration)
- [CONSTANTS_MODULE.md](CONSTANTS_MODULE.md) - System constants (TEST_MODULES, FS_COMMANDS, etc.)
- [UTILS_MODULE.md](UTILS_MODULE.md) - Utility functions
- [CLAUDE.md](../CLAUDE.md) - Agent instructions and coding standards

---

**Last Updated**: January 2026
**Maintainer**: WSmart+ Route Development Team
**Status**: ✅ Active - Comprehensive CLI framework documentation
