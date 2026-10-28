"""Tests for CLI argument parsing utilities."""

import sys
from unittest.mock import patch

import pytest
from logic.controllers.cli import ConfigsParser, LowercaseAction, parse_params

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestConfigsParser:
    """Test suite for ConfigsParser class"""

    @pytest.mark.arg_parser
    def test_lowercase_action(self):
        """Test LowercaseAction converts strings to lowercase"""
        parser = ConfigsParser()
        parser.add_argument("--test", action=LowercaseAction)
        args = parser.parse_args(["--test", "UPPERCASE"])
        assert args.test == "uppercase"

    @pytest.mark.arg_parser
    def test_lowercase_action_none(self):
        """Test LowercaseAction handles None values"""
        parser = ConfigsParser()
        parser.add_argument("--test", action=LowercaseAction, default=None)
        args = parser.parse_args([])
        assert args.test is None

    @pytest.mark.arg_parser
    def test_parse_process_args_basic(self):
        """
        Test basic parsing functionality where arguments are separated normally.
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest="command")
        cmd_parser = subparsers.add_parser("run")
        cmd_parser.add_argument("--value", type=int)

        # Test input: ['run', '--value', '10']
        command, args = parser.parse_process_args(["run", "--value", "10"])

        assert command == "run"
        assert args["value"] == 10

    @pytest.mark.arg_parser
    def test_parse_process_args_nargs_string_splitting(self):
        """
        Test the custom logic: splitting a single string with spaces
        into multiple arguments when nargs is defined.
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest="command")
        cmd_parser = subparsers.add_parser("run")
        # Arg with nargs='+'
        cmd_parser.add_argument("--list_items", nargs="+", type=str)

        # Scenario: Arguments received as a single string "a b c" instead of 'a', 'b', 'c'
        # This triggers lines 66-70 in parse_process_args
        raw_args = ["run", "--list_items", "item1 item2 item3"]
        command, args = parser.parse_process_args(raw_args)

        assert command == "run"
        assert args["list_items"] == ["item1", "item2", "item3"]

    @pytest.mark.arg_parser
    def test_parse_process_args_nargs_standard(self):
        """
        Test that standard space-separated arguments still work
        and aren't negatively affected by the splitting logic.
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest="command")
        cmd_parser = subparsers.add_parser("run")
        cmd_parser.add_argument("--list_items", nargs="+", type=str)

        # Standard input: ['run', '--list_items', 'item1', 'item2']
        raw_args = ["run", "--list_items", "item1", "item2"]
        command, args = parser.parse_process_args(raw_args)

        assert command == "run"
        assert args["list_items"] == ["item1", "item2"]

    @pytest.mark.arg_parser
    def test_parse_process_args_default_sys_argv(self):
        """
        Test that the method defaults to sys.argv[1:] if args=None.
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest="command")
        cmd_parser = subparsers.add_parser("test_cmd")
        cmd_parser.add_argument("--flag", action="store_true")

        # Mock sys.argv
        with patch.object(sys, "argv", ["script_name.py", "test_cmd", "--flag"]):
            command, args = parser.parse_process_args(None)

            assert command == "test_cmd"
            assert args["flag"] is True

    @pytest.mark.arg_parser
    def test_parser_initialization(self):
        """Test that parser initializes correctly"""
        parser = ConfigsParser(description="Test parser")
        assert parser is not None
        assert isinstance(parser, ConfigsParser)

    @pytest.mark.arg_parser
    def test_parse_command_valid(self):
        """Test parsing valid commands"""
        with patch.object(sys, "argv", ["script.py", "train"]):
            parser = ConfigsParser()
            subparsers = parser.add_subparsers(dest="command")
            subparsers.add_parser("train")
            command = parser.parse_command()
            assert command == "train"

    @pytest.mark.arg_parser
    def test_parse_command_invalid(self):
        """Test parsing invalid commands exits"""
        with patch.object(sys, "argv", ["script.py", "invalid_command"]):
            parser = ConfigsParser()
            subparsers = parser.add_subparsers(dest="command")
            subparsers.add_parser("train")
            with pytest.raises(SystemExit):
                parser.parse_command()

    @pytest.mark.arg_parser
    def test_error_message(self, capsys):
        """Test error message printing"""
        parser = ConfigsParser()
        with pytest.raises(RuntimeError):
            parser.error_message("Test error", print_help=False)
        captured = capsys.readouterr()
        assert "Test error" in captured.out

    @pytest.mark.edge_case
    def test_missing_required_command(self):
        """Test error when no command is provided"""
        with patch.object(sys, "argv", ["script.py"]):
            ConfigsParser()
            with pytest.raises(SystemExit):
                parse_params()

    @pytest.mark.edge_case
    def test_parse_process_args_mixed_types(self):
        """
        Test that splitting logic works correctly when mixed with other arguments
        and ignores flags (starts with '-').
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest="command")
        cmd_parser = subparsers.add_parser("run")
        cmd_parser.add_argument("--numbers", nargs="+", type=int)
        cmd_parser.add_argument("--other", type=str)

        # Input with numbers as a single string and a separate flag following it
        raw_args = ["run", "--numbers", "1 2 3", "--other", "value"]
        command, args = parser.parse_process_args(raw_args)

        assert command == "run"
        # Should be converted to ints by the type=int in add_argument after splitting
        assert args["numbers"] == [1, 2, 3]
        assert args["other"] == "value"


class TestFileSystemCommand:
    """Test suite for file system command"""

    @pytest.mark.file_system
    def test_fs_update_command(self, base_file_system_update_args):
        """Test file system update command"""
        args = base_file_system_update_args + [
            "--output_key",
            "waste",
            "--update_value",
            "1.5",
        ]
        with patch.object(sys, "argv", args):
            command, args_dict = parse_params()
            assert command == ("file_system", "update")
            assert args_dict["target_entry"] == "path/to/file.pkl"
            assert args_dict["update_value"] == 1.5

    @pytest.mark.file_system
    def test_fs_delete_command(self, base_file_system_delete_args):
        """Test file system delete command"""
        args = base_file_system_delete_args + [
            "--log_dir",
            "logs",
            "--log",
            "--delete_preview",
        ]
        with patch.object(sys, "argv", args):
            command, args_dict = parse_params()
            assert command == ("file_system", "delete")
            assert args_dict["log_dir"] == "logs"
            assert args_dict["delete_preview"] is True

    @pytest.mark.file_system
    def test_fs_cryptography_command(self, base_file_system_crypto_args):
        """Test file system cryptography command"""
        args = base_file_system_crypto_args + [
            "--symkey_name",
            "mykey",
            "--salt_size",
            "16",
            "--key_length",
            "32",
        ]
        with patch.object(sys, "argv", args):
            command, args_dict = parse_params()
            assert command == ("file_system", "cryptography")
            assert args_dict["symkey_name"] == "mykey"
            assert args_dict["salt_size"] == 16

    @pytest.mark.file_system
    def test_missing_fs_subcommand(self):
        """Test error when no file system subcommand is provided"""
        # Scenario: User enters 'file_system' but forgets 'update', 'delete', or 'cryptography'
        with patch.object(sys, "argv", ["script.py", "file_system"]), pytest.raises(SystemExit):
            parse_params()

    @pytest.mark.file_system
    def test_invalid_fs_subcommand(self):
        """Test error when an invalid file system subcommand is provided"""
        # Scenario: User enters a subcommand that doesn't exist
        with patch.object(sys, "argv", ["script.py", "file_system", "destroy"]), pytest.raises(SystemExit):
            parse_params()

    @pytest.mark.file_system
    def test_fs_update_mutual_exclusivity(self, base_file_system_update_args):
        """
        Test that update_operation and stats_function are mutually exclusive.
        We mock the mapping dictionaries to ensure the ActionFactory accepts the values,
        allowing the code to reach the validation logic in validate_file_system_args.
        """
        args = base_file_system_update_args + [
            "--update_operation",
            "op_test",
            "--stats_function",
            "stat_test",
        ]

        # Mock the maps used by UpdateFunctionMapActionFactory so our test inputs are considered valid
        # This allows us to skip the ValueError in the Action and hit the AssertionError in validation
        with (
            patch("logic.controllers.cli.base.update_function_factory.OPERATION_MAP", {"op_test": 1}),
            patch("logic.controllers.cli.base.update_function_factory.STATS_FUNCTION_MAP", {"stat_test": 1}),
            patch.object(sys, "argv", args),
            pytest.raises(AssertionError, match="mutually exclusive"),
        ):
            # Expect AssertionError: "'update_operation' and 'stats_function' arguments are mutually exclusive"
            # note: parse_params catches the AssertionError and re-raises it via error_message()
            parse_params()

    @pytest.mark.file_system
    def test_fs_input_keys_multiple(self, base_file_system_update_args):
        """Test input_keys argument accepting multiple values (nargs='*')"""
        args = base_file_system_update_args + ["--input_keys", "key1", "key2", "key3"]
        with patch.object(sys, "argv", args):
            command, args_dict = parse_params()
            assert command == ("file_system", "update")
            assert args_dict["input_keys"] == ["key1", "key2", "key3"]


class TestTrackingDatabaseCommand:
    """Test the tracking database CLI registration on main.py."""

    @pytest.mark.arg_parser
    def test_tracking_db_parser_captures_subcommand_args(self):
        from logic.controllers.cli.base.registry import get_parser

        command, opts = get_parser().parse_process_args(
            ["tracking_db", "stats", "--experiment", "AM-VRPP-50"]
        )

        assert command == "tracking_db"
        assert opts["tracking_args"] == ["stats", "--experiment", "AM-VRPP-50"]

    @pytest.mark.arg_parser
    def test_tracking_db_dispatch_forwards_args(self, monkeypatch):
        from logic.controllers.parser_dispatch import parser_entry_point

        calls = []
        monkeypatch.setattr(
            "logic.src.tracking.database.__main__.main",
            lambda argv: calls.append(argv),
        )

        parser_entry_point(("tracking_db", {"tracking_args": ["inspect"]}))

        assert calls == [["inspect"]]

    @pytest.mark.arg_parser
    def test_tracking_db_main_forwards_argv(self):
        from unittest.mock import patch

        from logic.src.tracking.database.__main__ import main as tracking_main

        with patch("logic.src.tracking.database.__main__.stats_database") as stats:
            tracking_main(["stats", "--experiment", "AM-VRPP-50"])

        stats.assert_called_once_with("AM-VRPP-50")

    @pytest.mark.arg_parser
    def test_main_routes_tracking_db(self, monkeypatch):
        import main as entry_point

        calls = []
        monkeypatch.setattr(
            "logic.src.tracking.database.__main__.main",
            lambda argv: calls.append(argv),
        )
        monkeypatch.setattr(sys, "argv", ["main.py", "tracking_db", "stats", "--experiment", "AM"])

        entry_point.main()

        assert calls == [["stats", "--experiment", "AM"]]
