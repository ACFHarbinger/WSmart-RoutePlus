"""Script to clean up Routing Algorithm Policies used for training models with Imitation Learning."""

import os
import sys

# Ensure cleanup_helper is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .cleanup_helper import IMITATION_POLICIES, clean_by_acronym


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_imitation_policies.py <acronyms...>")
        sys.exit(1)

    acronyms = sys.argv[1:]
    for acronym in acronyms:
        for acr in acronym.split(','):
            acr = acr.strip()
            if not acr:
                continue
            print(f"\n--- Cleaning up Imitation Learning Policy: {acr} ---")
            clean_by_acronym(
                acronym=acr,
                yaml_dirs=IMITATION_POLICIES["yaml_dirs"],
                config_dirs=IMITATION_POLICIES["config_dirs"],
                impl_dirs=IMITATION_POLICIES["impl_dirs"],
                yaml_prefixes=IMITATION_POLICIES["yaml_prefixes"],
                config_prefixes=IMITATION_POLICIES["config_prefixes"]
            )

if __name__ == "__main__":
    main()
