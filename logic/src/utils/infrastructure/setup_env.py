"""
Environment and cost setup utilities.

Attributes:
    setup_env: Sets up the solver environment (e.g., Gurobi).

Example:
    setup_env("swc_tcf")
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import gurobipy as gp
from dotenv import dotenv_values

import logic.src.constants as udef

try:
    from logic.src.utils.security import (  # pyrefly: ignore[missing-import]  # security utils are excluded from the minimal package
        decrypt_file_data,
        load_key,
    )
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False


def setup_env(
    policy: str,
    server: bool = False,
    gplic_filename: Optional[str] = None,
    symkey_name: Optional[str] = None,
    env_filename: Optional[str] = None,
) -> Optional[gp.Env]:
    """
    Sets up the solver environment (e.g., Gurobi).

    Args:
        policy: Policy name to determine environment type.
        server: Whether running on a server (requires specific license handling).
        gplic_filename: Gurobi license filename.
        symkey_name: Symmetric key name for decryption.
        env_filename: Environment variables filename.

    Returns:
        The Gurobi environment, or None if not applicable.
    """
    if "swc_tcf" in policy:
        params: Dict[str, Any] = {}
        if server:
            def convert_int(param: str) -> Union[int, str]:
                """Helper to convert string parameters to int if possible."""
                return int(param) if param.isdigit() else param

            if gplic_filename is not None:
                gplic_path: str = os.path.join(udef.ROOT_DIR, "assets", "api", gplic_filename)
                if symkey_name:
                    if not _SECURITY_AVAILABLE:
                        raise RuntimeError(
                            "symkey_name was provided but the security module is not available. "
                            "Remove symkey_name or provide the Gurobi license file unencrypted."
                        )
                    key = load_key(symkey_name=symkey_name, env_filename=env_filename or ".env")
                    data = decrypt_file_data(key, gplic_path)
                else:
                    with open(gplic_path, "r") as gp_file:
                        data = gp_file.read()
                if isinstance(data, bytes):
                    data = data.decode()
                params = {
                    line.split("=")[0]: convert_int(line.split("=")[1]) for line in data.split("\n") if "=" in line
                }
            else:
                if env_filename is None:
                    raise ValueError(
                        "env_filename must be provided when gplic_filename is not specified. "
                        "Please provide either a Gurobi license file path or an environment file name."
                    )
                env_path: str = os.path.join(udef.ROOT_DIR, "env", env_filename)
                config: Dict[str, Optional[str]] = dotenv_values(env_path)
                glp_ls: List[str] = ["WLSACCESSID", "WLSSECRET", "LICENSEID"]
                params = {glp: convert_int(config.get(glp, "")) for glp in glp_ls}  # type: ignore
                for glp_key, glp_val in params.items():
                    if isinstance(glp_val, str) and glp_val == "":
                        raise ValueError(f"Missing parameter {glp_key} for Gurobi license")
        elif gplic_filename is not None:
            gplic_path = os.path.join(udef.ROOT_DIR, "assets", "api", gplic_filename)
            if os.path.exists(gplic_path):
                os.environ["GRB_LICENSE_FILE"] = gplic_path
        params["OutputFlag"] = udef.SOLVER_OUTPUT_FLAG
        params["LogToConsole"] = udef.SOLVER_OUTPUT_FLAG
        return gp.Env(params=params)
    return None
