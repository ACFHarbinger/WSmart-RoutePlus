"""Ban references to deleted policy ``Params`` classes outside configs and tests.

P-10 deletes the parallel ``params.py`` layer: solvers read the typed config
dataclass, and every adapter-level literal or class-attribute fallback becomes
a read from the settled config (P-52). Two failure modes shipped through the
gap between the layers: adapters kept ``values.get("field", XParams.attr)``
fallbacks that re-imported the params class at solve time (SWC-TCF's gurobi.py
read ``SWCTCFParams.no_solution_time_factor`` as its own default), and silent
``from_config`` copies drifted from the config until the YAML was the only
agreeing definition. This check makes the deletion durable: a reference to a
banned ``Params`` symbol anywhere in ``logic/src`` outside ``configs/`` fails
CI, so the sites cannot regenerate.

WP-04 bans the four released routing configs it migrates (BPC, SWC-TCF,
ACO-HH, CVRP). WP-05 adds the ALNS-family classes and WP-06 the rest; when the
last params.py is gone, replace the list with a ban on any ``*Params`` symbol.

Usage::

    python tools/check_params_refs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[1]
LOGIC_SRC = REPO / "logic" / "src"

# Deleted params classes (WP-04, WP-05). Extend per work package; see module docstring.
# Deleted params classes (WP-04, WP-05, WP-06). Extend per work package; see
# module docstring. When the last params.py is gone (calm, hmlns remain),
# replace the list with a ban on any ``*Params`` symbol.
BANNED_SYMBOLS = frozenset(
    {
        "ABCParams",
        "ABPCHGParams",
        "ACOParams",
        "ADPRolloutParams",
        "AKSParams",
        "ALNSIPOParams",
        "ALNSParams",
        "AMPHHParams",
        "ARCOParams",
        "BBParams",
        "BCParams",
        "BPCParams",
        "BPParams",
        "CFRSParams",
        "CGHParams",
        "CVRPParams",
        "DEParams",
        "DistancePSOParams",
        "ExactGuidedHeuristicParams",
        "FAParams",
        "FILOParams",
        "GAParams",
        "GENIUSParams",
        "GIHHParams",
        "GLSParams",
        "GPHHParams",
        "GP_MP_HH_Params",
        "HGSADCParams",
        "HGSALNSParams",
        "HGSParams",
        "HGSRRParams",
        "HMMGDHHParams",
        "HNAParams",
        "HSParams",
        "HULKParams",
        "HVPLParams",
        "HyperACOParams",
        "ILSBDParams",
        "ILSParams",
        "ILSRVNDSPParams",
        "JointGreedyParams",
        "JointSAParams",
        "KGLSParams",
        "KSACOParams",
        "KSParams",
        "LASMPipelineParams",
        "LBParams",
        "LBVNSParams",
        "LCAParams",
        "LKH3Params",
        "LNSMIPParams",
        "LNSParams",
        "LRHParams",
        "MAParams",
        "MHHParams",
        "MP_ACO_Params",
        "MP_BMC_Params",
        "MP_ILS_Params",
        "MP_PSO_Params",
        "MSBPCSPParams",
        "MemeticAlgorithmDualPopulationParams",
        "MemeticAlgorithmIslandModelParams",
        "MemeticAlgorithmToleranceBasedSelectionParams",
        "MuCommaLambdaESParams",
        "MuKappaLambdaESParams",
        "MuPlusLambdaESParams",
        "NDSBRKGAParams",
        "NeuralParams",
        "PGCLNSParams",
        "PHHParams",
        "POPMUSICParams",
        "PSOMAParams",
        "PSOParams",
        "QDEParams",
        "RENSParams",
        "RFOParams",
        "RLALNSParams",
        "RLGDHHParams",
        "RLHVPLParams",
        "RTSParams",
        "SANSParams",
        "SAParams",
        "SCAParams",
        "SDPParams",
        "SHHParams",
        "SISRParams",
        "SLCParams",
        "SRCParams",
        "SSHHParams",
        "SWCTCFParams",
        "TPKSParams",
        "TSPParams",
        "TSParams",
        "VNSParams",
        "VPLParams",
    }
)

# Configs are the surviving source of truth; tests are exempt by owner ruling.
EXEMPT_PARTS = ("configs",)

_PATTERN = re.compile(r"\b(" + "|".join(BANNED_SYMBOLS) + r")\b")


def find_violations() -> List[Tuple[Path, int, str]]:
    """Scan logic/src for banned Params symbol references.

    Returns:
        List of (path, line_number, stripped_line) tuples for each hit.
    """
    violations: List[Tuple[Path, int, str]] = []
    for path in sorted(LOGIC_SRC.rglob("*.py")):
        relative = path.relative_to(LOGIC_SRC)
        if any(part in EXEMPT_PARTS for part in relative.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.search(line):
                violations.append((path, lineno, line.strip()))
    return violations


def main() -> int:
    """Entry point; exits 1 with a report when banned references exist."""
    violations = find_violations()
    if violations:
        print(f"params-ref check failed: {len(violations)} banned reference(s):")
        for path, lineno, line in violations:
            print(f"  {path}:{lineno}: {line}")
        print(
            "P-10/P-52: read the settled config dataclass instead; "
            "Params symbols are allowed only under logic/src/configs/ and in tests."
        )
        return 1
    print("params-ref check passed: no banned Params references in logic/src.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
