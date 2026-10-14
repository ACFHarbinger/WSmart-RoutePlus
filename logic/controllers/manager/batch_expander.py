"""
Batch expander — Cartesian-product expansion of dimension axes.

When a batch YAML entry contains an ``expand`` block, the user declares one or
more *dimensions* (e.g. ``policies``, ``packing``, ``rl_algorithm``)
each with a list of values.  The expander computes the Cartesian product of all
dimensions and returns one ``BatchJob`` per combination.

Dimension-to-override mapping
------------------------------
The expander maps each dimension name to a Hydra override string using either the
built-in mappings below or a user-supplied ``dim_overrides`` table (specified
inside the ``expand`` block as ``dim_overrides: {dim_name: "override_template"}``).

Built-in dimension mappings (all as ``override_template``):
    policies            → ``sim.policies=[{value}]``
    packing → ``sim.policy.packing.strategy={value}``
    routing algorithms        → ``sim.policy.constructor={value}``
    route_improvers     → ``sim.policy.route_improver={value}``
    acceptance_criteria → ``sim.policy.acceptance_criterion={value}``
    models              → ``train.policy.model.name={value}``
    rl_algorithms       → ``rl.algorithm={value}``

The ``{value}`` placeholder is replaced with each element of the dimension's
value list during expansion.

Example YAML expand block
--------------------------
.. code-block:: yaml

    - task: test_sim
      expand:
        policies: [[alns], [hgs], [psoma]]
        packing: [lookahead, last_minute_cf70]
      name_template: "{policies[0]}_{packing}"
      base_overrides:
        - "sim.graph.area=figueiradafoz"
      post_steps:
        - type: git_commit
          args:
            message: "Results: {name}"

This produces 6 jobs (3 policies × 2 ms strategies), each with a unique name
and the appropriate Hydra overrides applied.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterator, List, Optional

from .batch_job import BatchJob
from .batch_step import BatchStep

__all__ = ["BatchExpander", "BUILTIN_DIM_OVERRIDES"]

# ---------------------------------------------------------------------------
# Built-in dimension → Hydra override template mapping
# ---------------------------------------------------------------------------

BUILTIN_DIM_OVERRIDES: Dict[str, str] = {
    "policies": "sim.policies=[{value}]",
    "packing": "sim.policy.packing.strategy={value}",
    "constructors": "sim.policy.constructor={value}",
    "route_improvers": "sim.policy.route_improver={value}",
    "acceptance_criteria": "sim.policy.acceptance_criterion={value}",
    "models": "train.policy.model.name={value}",
    "rl_algorithms": "rl.algorithm={value}",
}


def _render_override(template: str, value: Any) -> str:
    """Substitute ``{value}`` in *template* with *value*.

    If *value* is a list the rendered string joins its elements with commas so
    it remains a valid Hydra list override, e.g. ``[alns,hgs]``.

    Args:
        template: Override template containing ``{value}``.
        value: Scalar string, list, or other value.

    Returns:
        Rendered override string.
    """
    if isinstance(value, (list, tuple)):
        rendered_value = ",".join(str(v) for v in value)
        return template.replace("{value}", rendered_value)
    return template.replace("{value}", str(value))


def _name_from_expand_vars(
    name_template: Optional[str],
    expand_vars: Dict[str, Any],
    task: str,
    job_index: int,
) -> str:
    """Derive a human-readable job name.

    Args:
        name_template: Optional Python format string with keys from
            *expand_vars* plus ``{task}`` and ``{index}``.
        expand_vars: Dimension-value mapping for this combination.
        task: Hydra task key.
        job_index: Batch-wide job index.

    Returns:
        Rendered job name.
    """
    if not name_template:
        parts = []
        for _key, val in expand_vars.items():
            if isinstance(val, (list, tuple)):
                parts.append("_".join(str(v) for v in val))
            else:
                parts.append(str(val))
        return "_".join(parts) if parts else f"{task}_{job_index}"

    ctx: Dict[str, Any] = {"task": task, "index": job_index}
    # Add dimension values; list values are also available as indexed access in
    # the template via dict values (Python format strings don't support list
    # indexing natively, so we flatten them to string first).
    for key, val in expand_vars.items():
        if isinstance(val, (list, tuple)):
            ctx[key] = "_".join(str(v) for v in val)
        else:
            ctx[key] = val

    try:
        return name_template.format(**ctx)
    except (KeyError, IndexError):
        return f"{task}_{job_index}"


class BatchExpander:
    """Expand a single batch YAML entry into multiple ``BatchJob`` instances.

    Parameters
    ----------
    entry:
        A single parsed YAML run entry dict containing at minimum:
        - ``task`` (str)
        - ``expand`` (dict mapping dimension names to value lists)
        - ``base_overrides`` (optional list of additional Hydra overrides)
        - ``name_template`` (optional format string)
        - ``dim_overrides`` (optional dict overriding built-in mappings)
        - ``pre_steps`` / ``post_steps`` (optional step lists)
    start_index:
        The next available job index in the batch.
    """

    def __init__(self, entry: Dict[str, Any], start_index: int = 0) -> None:
        self._entry = entry
        self._start_index = start_index

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def expand(self) -> List[BatchJob]:
        """Return the list of ``BatchJob`` instances produced by this entry.

        Returns:
            List of expanded jobs, one per Cartesian product combination.
        """
        return list(self._iter_jobs())

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _iter_jobs(self) -> Iterator[BatchJob]:
        """Yield individual jobs from the Cartesian product of dimensions."""
        entry = self._entry
        task: str = entry.get("task", "test_sim")
        expand_block: Dict[str, List[Any]] = entry.get("expand", {})
        base_overrides: List[str] = list(entry.get("base_overrides", []) or [])
        name_template: Optional[str] = entry.get("name_template")
        user_dim_overrides: Dict[str, str] = dict(entry.get("dim_overrides", {}) or {})
        metadata: Dict[str, Any] = dict(entry.get("metadata", {}) or {})

        # Merge built-in and user-supplied override templates (user wins)
        dim_override_map: Dict[str, str] = {**BUILTIN_DIM_OVERRIDES, **user_dim_overrides}

        pre_steps = _parse_steps(entry.get("pre_steps") or [])
        post_steps = _parse_steps(entry.get("post_steps") or [])

        if not expand_block:
            # No expansion — emit a single job from base_overrides
            yield BatchJob(
                task=task,
                name=entry.get("name", task),
                overrides=base_overrides,
                pre_steps=pre_steps,
                post_steps=post_steps,
                index=self._start_index,
                metadata=metadata,
            )
            return

        dim_names = list(expand_block.keys())
        dim_values = [expand_block[d] for d in dim_names]

        for combo_index, combo in enumerate(itertools.product(*dim_values)):
            expand_vars: Dict[str, Any] = dict(zip(dim_names, combo, strict=True))

            # Build Hydra overrides for this combination
            combo_overrides: List[str] = list(base_overrides)
            for dim, val in expand_vars.items():
                tmpl = dim_override_map.get(dim)
                if tmpl:
                    combo_overrides.append(_render_override(tmpl, val))

            job_index = self._start_index + combo_index
            name = _name_from_expand_vars(name_template, expand_vars, task, job_index)

            yield BatchJob(
                task=task,
                name=name,
                overrides=combo_overrides,
                pre_steps=pre_steps,
                post_steps=post_steps,
                index=job_index,
                expand_vars=expand_vars,
                metadata=metadata,
            )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_steps(raw: List[Any]) -> List[BatchStep]:
    """Convert a list of raw dicts to ``BatchStep`` instances.

    Args:
        raw: List of step dicts from YAML.

    Returns:
        List of ``BatchStep`` objects.
    """
    steps: List[BatchStep] = []
    for item in raw:
        if isinstance(item, dict):
            steps.append(
                BatchStep(
                    type=item.get("type", ""),
                    args=dict(item.get("args", {})),
                    condition=item.get("condition", "always"),
                    name=item.get("name"),
                )
            )
        elif isinstance(item, BatchStep):
            steps.append(item)
    return steps
