"""Task packing onto machines from CR-11 memory predictions (CR-09).

A warm pool worker is one machine with ``cpus`` and ``memory_gb``; it runs several tasks
at once. :func:`pack` assigns tasks to machines first-fit decreasing by memory (then CPUs),
so the number of machines a queue needs is ``len(pack(...))`` and a worker's slot count is
how many of the predicted tasks fit on it.

Attributes:
    Item: A task's ``(id, cpus, memory_gb)``.
    pack: First-fit decreasing bins.
    slots_per_machine: Tasks of a given size that fit on one machine.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Item = Tuple[str, float, float]


def pack(items: Sequence[Item], cpus: float, memory_gb: float) -> List[List[str]]:
    """Machines (lists of task ids) holding ``items``; an item larger than a machine gets one alone."""
    bins: List[Tuple[float, float, List[str]]] = []  # free cpus, free memory, ids
    for task_id, need_cpus, need_mem in sorted(items, key=lambda i: (-i[2], -i[1], i[0])):
        for index, (free_cpus, free_mem, ids) in enumerate(bins):
            if need_cpus <= free_cpus + 1e-9 and need_mem <= free_mem + 1e-9:
                bins[index] = (free_cpus - need_cpus, free_mem - need_mem, ids + [task_id])
                break
        else:
            bins.append((max(0.0, cpus - need_cpus), max(0.0, memory_gb - need_mem), [task_id]))
    return [ids for _, _, ids in bins]


def slots_per_machine(cpus: float, memory_gb: float, task_cpus: float, task_memory_gb: float) -> int:
    """How many tasks of this size fit on one machine (at least 1)."""
    by_cpu = math.floor(cpus / task_cpus + 1e-9) if task_cpus > 0 else 1
    by_mem = math.floor(memory_gb / task_memory_gb + 1e-9) if task_memory_gb > 0 else by_cpu
    return max(1, min(by_cpu, by_mem))
