"""
Unified data export for BDH visualization frontend.

Runs all export scripts in the correct order so you only need one command
before starting the backend or frontend. Generates:

  - static/data/activations.json   (activation atlas, specialists, sequences)
  - static/data/battle_data.json  (BDH vs Transformer "Strawman Showdown")
  - static/data/comparison_data.json (sparsity, concept battle, visual sample)
  - static/data/explainer.json     (dual-pipeline Battle Arena narrative)

Usage:
  From project root:  python backend/export_all.py
  From backend:       python export_all.py
"""

import os
import sys

# Ensure backend is on path when run from project root
BACKEND = os.path.dirname(os.path.abspath(__file__))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.chdir(BACKEND)


def main():
    print("=" * 60)
    print("BDH Visualization — unified data export")
    print("=" * 60)

    # 1. Activations (metadata, vocabulary, specialists, sequences)
    print("\n[1/4] Exporting activations...")
    import export_activations
    export_activations.main()

    # 2. Battle (BDH vs dense transformer, layer loads, graphs)
    print("\n[2/4] Exporting battle data...")
    import export_battle
    export_battle.main()

    # 3. Comparison (sparsity, concept battle, visual sample)
    print("\n[3/4] Exporting comparison data...")
    import export_comparison
    export_comparison.main()

    # 4. Explainer (dual-pipeline narrative for Battle Arena)
    print("\n[4/4] Exporting explainer data...")
    import export_explainer
    export_explainer.main()

    print("\n" + "=" * 60)
    print("All exports finished. You can now run the backend and frontend.")
    print("=" * 60)


if __name__ == "__main__":
    main()
