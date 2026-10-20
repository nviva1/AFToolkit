"""The ``aftoolkit`` command line.

One entry point with subcommands, replacing four separate console scripts
(``run_protein_task``, ``run_protein_complex_task``, ``aftoolkit-predict`` and
``aftoolkit-score-complex``) that each re-declared the same arguments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from aftoolkit.cli import common
from aftoolkit.logging_config import configure_logging

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="aftoolkit",
        description="Protein engineering with AlphaFold2-derived representations.",
    )
    parser.add_argument("--version", action="store_true",
                        help="Print the installed version and exit.")
    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    _add_task_command(subcommands)
    _add_complex_command(subcommands)
    _add_predict_command(subcommands)
    _add_score_command(subcommands)
    _add_train_command(subcommands)
    return parser


def _add_task_command(subcommands) -> None:
    p = subcommands.add_parser(
        "task", help="Extract AlphaFold features for a mutated protein.")
    common.add_structure_args(p)
    p.add_argument("--chain", help="Chain(s) of the protein, comma-separated.")
    common.add_inference_args(p)
    common.add_output_args(p)
    p.set_defaults(handler=_run_task, requires=("chain",))


def _add_complex_command(subcommands) -> None:
    p = subcommands.add_parser(
        "complex", help="Extract features for a complex, bound and unbound.")
    common.add_structure_args(p)
    p.add_argument("--chains-for-protein",
                   help="Chains of the mutated side, comma-separated.")
    p.add_argument("--chains-for-complex",
                   help="All chains of the complex, comma-separated.")
    common.add_inference_args(p)
    common.add_output_args(p)
    p.set_defaults(handler=_run_complex, requires=("chains_for_protein",))


def _add_predict_command(subcommands) -> None:
    p = subcommands.add_parser(
        "predict",
        help="Predict mutation impact (ddG + confidence) with trained adapters.")
    p.add_argument("--pdb", required=True, help="Input PDB structure.")
    p.add_argument("--chain", required=True, help="Chain the mutations apply to.")
    p.add_argument("--mutations", "--mutation", dest="mutations", required=True,
                   help="One mutation (E285K) or several, comma-separated.")
    p.add_argument("--models", nargs="*", default=[],
                   help="Trained adapter .pkl paths/globs "
                        "(omit for confidence only).")
    common.add_inference_args(p)
    p.add_argument("-o", "--out-dir", default="results", help="Output directory.")
    p.add_argument("--save-structures", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Write the predicted wildtype and mutant PDB files "
                        "(default: on).")
    p.add_argument("--pymol", action="store_true",
                   help="Also write a structure whose B-factor column is the "
                        "mutant-minus-wildtype pLDDT, a PyMOL script colouring "
                        "it red (confidence lost) to blue (gained), and a .pse "
                        "session if PyMOL is installed.")
    p.set_defaults(handler=_run_predict)


def _add_score_command(subcommands) -> None:
    p = subcommands.add_parser(
        "score", help="Interface PAE (iPAE) and confidence for a complex.")
    p.add_argument("--pdb", required=True)
    p.add_argument("--antigen", required=True,
                   help="Antigen chain id(s), comma-separated.")
    p.add_argument("--antibody", required=True,
                   help="Antibody chain id(s), comma-separated.")
    p.add_argument("--contact-cutoff", type=float, default=None,
                   help="Restrict iPAE to residue pairs within this CA-CA "
                        "distance, in Angstrom (e.g. 12.0).")
    common.add_inference_args(p)
    p.add_argument("-o", "--out-dir", default="results", help="Output directory.")
    p.set_defaults(handler=_run_score)


def _add_train_command(subcommands) -> None:
    p = subcommands.add_parser(
        "train", help="Train a stability or affinity adapter.")
    p.add_argument("--model", required=True,
                   choices=("svm", "mlp", "catboost"),
                   help="Estimator behind the adapter.")
    p.add_argument("--target", default="monomer", choices=("monomer", "multimer"),
                   help="Monomer stability (ddG) or multimer affinity.")
    p.add_argument("--train-csv", required=True,
                   help="CSV of mutations with train/test splits.")
    p.add_argument("--features-dir", required=True,
                   help="Directory of per-mutation ProteinTask .pkl files.")
    p.add_argument("--npy-dir", default="./",
                   help="Directory to cache assembled .npy datasets in.")
    p.add_argument("--out-dir", default="data/models",
                   help="Directory to write the trained adapter into.")
    p.add_argument("--features", nargs="*",
                   default=["pair", "lddt_logits", "plddt"],
                   help="Feature names the adapter consumes.")
    p.add_argument("--protein-aggregation", default="mutpos",
                   choices=("mutpos", "sum", "mean"))
    p.add_argument("--multi-aggregation", default="sum", choices=("sum", "mean"))
    p.add_argument("--no-concat", dest="concat_features", action="store_false",
                   help="Use the mutant-minus-wildtype difference instead of "
                        "concatenating the two embeddings.")
    p.set_defaults(handler=_run_train)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def _require_structure_input(parser, args) -> None:
    """Both task subcommands accept either one structure or a batch CSV."""
    if args.input:
        return
    if not args.pdb:
        parser.error("Provide either --pdb (single structure) or --input (task CSV).")
    import os

    if not os.path.exists(args.pdb):
        parser.error(f"PDB file does not exist: {args.pdb}")
    for name in getattr(args, "requires", ()):
        if not getattr(args, name, None):
            parser.error(f"--{name.replace('_', '-')} is required when --pdb is given")


def _run_task(parser, args) -> None:
    from aftoolkit.cli.tasks import run_protein

    _require_structure_input(parser, args)
    run_protein(args)


def _run_complex(parser, args) -> None:
    from aftoolkit.cli.tasks import run_complex

    _require_structure_input(parser, args)
    run_complex(args)


def _run_predict(parser, args) -> None:
    from aftoolkit.inference import predict_mutation

    report = predict_mutation(
        pdb=args.pdb, chain=args.chain, mutations=args.mutations,
        model_paths=args.models, device=args.device, n_recycle=args.n_recycle,
        out_dir=args.out_dir, pipeline=args.pipeline, chain_gap=args.chain_gap,
        save_structures=args.save_structures, pymol_session=args.pymol,
    )
    print(json.dumps(dataclasses.asdict(report), indent=2))


def _run_score(parser, args) -> None:
    from aftoolkit.scoring import score_complex

    result = score_complex(
        pdb=args.pdb,
        antigen_chains=args.antigen.split(","),
        antibody_chains=args.antibody.split(","),
        device=args.device, n_recycle=args.n_recycle, out_dir=args.out_dir,
        chain_gap=args.chain_gap, pipeline=args.pipeline,
        contact_cutoff=args.contact_cutoff,
    )
    print(json.dumps(result, indent=2))


def _run_train(parser, args) -> None:
    from aftoolkit.training import train_adapter

    path = train_adapter(
        model=args.model,
        target=args.target,
        train_csv=args.train_csv,
        features_dir=args.features_dir,
        npy_dir=args.npy_dir,
        out_dir=args.out_dir,
        features=args.features,
        concat_features=args.concat_features,
        protein_aggregation=args.protein_aggregation,
        multi_aggregation=args.multi_aggregation,
    )
    print(path)


def main(argv=None) -> None:
    """Entry point for the ``aftoolkit`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from aftoolkit import __version__

        print(__version__)
        return
    if not getattr(args, "handler", None):
        parser.print_help()
        sys.exit(2)

    configure_logging()
    try:
        args.handler(parser, args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
