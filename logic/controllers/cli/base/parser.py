"""
Base parser utilities for the WSmart Route+ modular CLI.

This module provides the core ConfigsParser class and shared argument actions
used across all command-specific parsers.

Attributes:
    ConfigsParser: Custom ArgumentParser to handle string-based nargs correctly.

Example:
    >>> from logic.controllers.cli.base.parser import ConfigsParser
    >>> parser = ConfigsParser()
    >>> parser.add_argument('--test', action='store_true')
    >>> parser.parse_args(['--test'])
    Namespace(test=True)
"""

import argparse
import sys
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple, Union


class ConfigsParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser to handle string-based nargs correctly.

    This class extends ArgumentParser to provide additional functionality for
    processing command-line arguments, especially those involving sequence types
    and nested parser structures.

    Attributes:
        _actions: List of actions for this parser.
    """

    def _str_to_nargs(self, nargs: Union[str, Sequence]) -> Union[str, Sequence]:
        """
        Convert a single string argument into a list if expected by nargs.

        Args:
            nargs: The value of the argument, potentially a single-element sequence containing a string.

        Returns:
            The processed value, split into a list if it was a space-separated string within a sequence.
        """
        if isinstance(nargs, Sequence) and len(nargs) == 1:
            return nargs[0].split() if isinstance(nargs[0], str) else nargs
        else:
            return nargs

    def _process_args(self, namespace: argparse.Namespace) -> None:
        """
        Post-process arguments in the namespace, handling special narg conversions.

        Args:
            namespace: The namespace containing parsed arguments.

        Returns:
            None
        """
        for action in self._actions:
            if action.nargs is not None:
                if action.dest == "help":
                    continue
                value = getattr(namespace, action.dest)
                if value is not None:
                    transformed_value = self._str_to_nargs(value)
                    setattr(namespace, action.dest, transformed_value)

    def parse_command(self, args: Optional[Sequence[str]] = None) -> Optional[str]:
        """
        Parse only the command from the arguments.

        Args:
            args: The sequence of arguments to parse. Defaults to sys.argv[1:].

        Returns:
            The name of the command found in the arguments, if any.
        """
        if args is None:
            args = sys.argv[1:]
        namespace = super().parse_args(args)
        return getattr(namespace, "command", None)

    def parse_process_args(
        self, args: Optional[List[str]] = None, command: Optional[str] = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parse arguments and returns the command and options dictionary.

        This method performs a more complex parsing that handles space-separated
        strings intended for multi-value arguments (nargs) by splitting them
        before full parsing.

        Args:
            args: The sequence of arguments to parse. Defaults to sys.argv[1:].
            command: Not used, kept for signature compatibility. Defaults to None.

        Returns:
            A tuple containing (command_name, dictionary_of_filtered_arguments).
        """
        if args is None:
            args = sys.argv[1:]

        actions_to_check = list(self._actions)

        command_name = None
        if args and not args[0].startswith("-"):
            command_name = args[0]

        if command_name:
            subparsers_action = next(
                (a for a in actions_to_check if isinstance(a, argparse._SubParsersAction)),
                None,
            )

            if subparsers_action and command_name in subparsers_action.choices:
                sub_parser = subparsers_action.choices[command_name]
                actions_to_check.extend(sub_parser._actions)

        for action in actions_to_check:
            if action.dest == "help":
                continue

            if action.nargs is not None and action.type is not None:
                opts = action.option_strings
                idx = next((i for i, x in enumerate(args) if x in opts), None)
                if idx is not None and (idx + 1) < len(args):
                    arg_val = args[idx + 1]
                    if isinstance(arg_val, str) and not arg_val.startswith("-"):
                        arg_parts = arg_val.split()
                        if len(arg_parts) > 1:
                            args[idx + 1 : idx + 2] = arg_parts

        subnamespace = super().parse_args(args)
        parsed_args_dict = vars(subnamespace)
        filtered_args = {key: value if value != "" else None for key, value in parsed_args_dict.items()}

        command = filtered_args.pop("command")
        return command, filtered_args

    def error_message(self, message: str, print_help: bool = True) -> NoReturn:
        """
        Print error message and optionally help.

        Args:
            message: The error message to display.
            print_help: Whether to print the help text as well. Defaults to True.

        Raises:
            The original exception if this is caught in a block, or simply exits the program via ArgumentParser mechanism.
        """
        print(message, end=" ")
        if print_help:
            self.print_help()
        raise
