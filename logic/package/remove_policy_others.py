"""Script to clean up other Policy components (Acceptance Criteria, Route Improvers, Packing, Selection & Construction)."""

import os
import sys

# Ensure cleanup_helper is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .cleanup_helper import POLICY_OTHERS, clean_by_acronym


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_policy_others.py <acronyms...>")
        sys.exit(1)

    acronyms = sys.argv[1:]
    for acronym in acronyms:
        for acr in acronym.split(','):
            acr = acr.strip()
            if not acr:
                continue
            print(f"\n--- Cleaning up Policy Other: {acr} ---")
            clean_by_acronym(
                acronym=acr,
                yaml_dirs=POLICY_OTHERS["yaml_dirs"],
                config_dirs=POLICY_OTHERS["config_dirs"],
                impl_dirs=POLICY_OTHERS["impl_dirs"],
                yaml_prefixes=POLICY_OTHERS["yaml_prefixes"],
                config_prefixes=POLICY_OTHERS["config_prefixes"]
            )

if __name__ == "__main__":
    main()
