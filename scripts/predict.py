#!/usr/bin/env python
"""Thin wrapper around the packaged inference CLI.

The implementation lives in ``aftoolkit.inference`` and is also exposed as the
``aftoolkit-predict`` console command. This shim lets you run it directly:

    python scripts/predict.py --pdb p53.pdb --chain A --mutation E285K \
        --models "data/models/monomer/pair+lddt_logits+plddt/trained_*.pkl"
"""

from aftoolkit.inference import main

if __name__ == "__main__":
    main()
