import logging
import os
import pickle
import sys
import warnings
import argparse

import pandas as pd

from aftoolkit.processing.protein_task import ProteinTask
from aftoolkit.processing.arg_parser import parser, parse_mutations, parse_positions
from aftoolkit.processing.utils import safe_filename_part
from aftoolkit.logging_config import configure_logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def protein_task_parser():
    parent_parser = parser()
    parser_ = argparse.ArgumentParser(parents=[parent_parser])
    parser_.add_argument('--chain', help='Chain of protein',
                         default=None)
    # parser_.add_argument('--regions', help='Regions of protein in format {"A": [(1, 36)]}',
    #                      default=None)
    return parser_


def prepare_task(df):
    df.reset_index(drop=True)
    task_dict = {}
    if 'positions' not in df.columns:
        df.loc[:, 'positions'] = None
    if 'mutations' not in df.columns:
        df.loc[:, 'mutations'] = None
    for i in df.index:
        row = df.loc[i]
        task = {
            "input_protein": {
                "path": row['path'],
                "chains": row['chain'].split(','),
                "regions": None,
            },
            "mutants": parse_mutations(row['mutations']),
            "obs_positions": parse_positions(row['positions'])
        }
        task_dict[f"{row['pdb_id']}_{row['chain']}_{row['mut_info']}"] = task

    return task_dict


def main():
    parser_ = protein_task_parser()
    args = parser_.parse_args()

    configure_logging()

    tasks = None
    if args.input:
        task_table = pd.read_csv(args.input)
        tasks = prepare_task(task_table)
    elif args.pdb:
        if not os.path.exists(args.pdb):
            parser_.error(f"PDB file does not exist: {args.pdb}")
        if not args.chain:
            parser_.error("--chain is required when --pdb is given")
    else:
        parser_.error(
            "Provide either --pdb (single structure) or --input (task CSV)."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Imported here (not at module top) so that --help and argument validation
    # work without the optional torch/OpenFold stack installed.
    from aftoolkit.processing.openfold_wrapper import OpenFoldWrapper

    logger.info("Initializing OpenFold wrapper (device=%s)", args.device or "auto")
    of_wrapper = OpenFoldWrapper(device=args.device,
                                 inference_n_recycle=args.inference_n_recycle,
                                 always_use_template=args.always_use_template,
                                 side_chain_mask=args.side_chain_mask,
                                 return_all_cycles=args.return_all_cycles)
    of_wrapper.init_model()

    if args.pdb:
        protein_task = ProteinTask()
        protein_task.set_input_protein_task(protein_path=args.pdb,
                                            chains=args.chain.split(','))
        protein_task.set_task_mutants(parse_mutations(args.mutations))
        protein_task.set_observable_positions(parse_positions(args.positions))
        logger.info("Calculating embeddings")
        protein_task.evaluate(of_wrapper=of_wrapper,
                              store_of_protein=args.store_protein
                              )

        out_name = "_".join(
            safe_filename_part(p) for p in (args.pdb, args.chain, args.mutations)
        )
        out_path = os.path.join(args.output_dir, f"{out_name}.pkl")
        protein_task.save(out_path)
        logger.info("Saved results to %s", out_path)

    else:
        n_ok = 0
        for task_id in tasks:
            protein_task = ProteinTask()
            try:
                logger.info("Processing task %s", task_id)
                protein_task.set_input_protein_task(task=tasks[task_id])
                protein_task.evaluate(of_wrapper=of_wrapper,
                                      store_of_protein=args.store_protein)
                out_path = os.path.join(
                    args.output_dir, f"{safe_filename_part(task_id)}.pkl"
                )
                with open(out_path, "wb") as f:
                    pickle.dump(protein_task, f)
                n_ok += 1
            except Exception:
                logger.exception("Failed to process task %s; skipping", task_id)
                continue
        logger.info("Processed %d/%d tasks successfully", n_ok, len(tasks))


if __name__ == "__main__":
    main()
