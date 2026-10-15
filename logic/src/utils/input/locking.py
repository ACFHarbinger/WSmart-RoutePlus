"""
Persistence utilities (locking, thread-safe I/O).

Attributes:
    read_output: Reads a JSON file and extracts specific policy values into a transposed list.

Example:
    >>> from logic.src.utils.io.locking import read_output
    >>> logs, logs_dict = read_output("path/to/log.json", ["policy1", "policy2"], "data_distribution")
"""

import json

import pandas as pd

import logic.src.constants as udef


def read_output(json_path, policies, data_distribution, lock=None):
    """
    Reads a JSON file and extracts specific policy values into a transposed list.

    Args:
        json_path: Path to the JSON file.
        policies: List of policy keys to extract.
        data_distribution: String of data distribution to extract.
        lock: Thread lock for safe access.

    Returns:
        list: Transposed list of values for the specified policies.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(json_path) as json_file:
            json_data = json.load(json_file)
    finally:
        if lock is not None:
            lock.release()
    logs = []
    logs_dict = {}
    for policy in policies:
        for key, val in json_data.items():
            if policy in key and data_distribution in key:
                logs.append(val)
                logs_dict[key] = val
    logs = pd.DataFrame(logs).values.transpose().tolist()
    return (logs, logs_dict)
