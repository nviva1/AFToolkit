import logging
import os
import pickle
import argparse

import pandas as pd

from aftoolkit.processing.protein_complex_task import ProteinComplexTask
from aftoolkit.processing.arg_parser import parser, parse_mutations, parse_positions
from aftoolkit.processing.utils import safe_filename_part
from aftoolkit.logging_config import configure_logging

logger = logging.getLogger(__name__)
COLUMNS_INPUT = ['pdb_id', 'chains_for_protein', 'mutations', 'positions']


def protein_complex_task_parser():
    parent_parser = parser()
    parser_ = argparse.ArgumentParser(parents=[parent_parser])
    parser_.add_argument('--chains-for-complex', help='Chains of protein complex',
                         default=None)
    parser_.add_argument('--chains-for-protein', help='Chain of protein',
                         default=None)
    return parser_


def prepare_task(df):
    df.reset_index(drop=True)
    task_dict = {}
    if 'chains_for_complex' not in df.columns:
        df.loc[:, 'chains_for_complex'] = None
    else:
        df.loc[:, 'chains_for_complex'] = df['chains_for_complex'].apply(lambda s: s.split(','))
    if 'positions' not in df.columns:
        df.loc[:, 'positions'] = None
    if 'mutations' not in df.columns:
        df.loc[:, 'mutations'] = None
    for i in df.index:
        row = df.loc[i]
        task = {
            "input_protein": {
                "path": row['path'],
                "chains_for_protein": row['chains_for_protein'].split(','),
                "chains_for_complex": row['chains_for_complex'],
            },
            "mutants": parse_mutations(row['mutations']),
            "obs_positions": parse_positions(row['positions']),
        }
        task_dict[f"{row['pdb_id']}_{row['chains_for_protein']}_{row['mut_info']}"] = task

    return task_dict


def main():
    parser_ = protein_complex_task_parser()
    args = parser_.parse_args()

    configure_logging()

    tasks = None
    if args.input:
        task_table = pd.read_csv(args.input)
        tasks = prepare_task(task_table)
    elif args.pdb:
        if not os.path.exists(args.pdb):
            parser_.error(f"PDB file does not exist: {args.pdb}")
        if not args.chains_for_protein:
            parser_.error("--chains-for-protein is required when --pdb is given")
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
        protein_complex_task = ProteinComplexTask()
        chains_for_protein = args.chains_for_protein.split(',')
        if args.chains_for_complex:
            chains_for_complex = args.chains_for_complex.split(',')
        else:
            chains_for_complex = None
        protein_complex_task.set_input_protein_task(protein_path=args.pdb,
                                                    chains_for_protein=chains_for_protein,
                                                    chains_for_complex=chains_for_complex,
                                                    )
        protein_complex_task.set_task_mutants(parse_mutations(args.mutations))
        protein_complex_task.set_observable_positions(parse_positions(args.positions))
        protein_complex_task.set_wildtype()
        protein_complex_task.set_mutate_protein()
        logger.info("Calculating embeddings")
        protein_complex_task.evaluate(of_wrapper=of_wrapper,
                                      store_of_protein=args.store_protein)

        out_name = "_".join(
            safe_filename_part(p)
            for p in (args.pdb, args.chains_for_protein, args.mutations)
        )
        out_path = os.path.join(args.output_dir, f"{out_name}.pkl")
        protein_complex_task.save(out_path)
        logger.info("Saved results to %s", out_path)

    else:
        n_ok = 0
        for task_id in tasks:
            protein_complex_task = ProteinComplexTask()
            try:
                logger.info("Processing task %s", task_id)
                protein_complex_task.set_input_protein_task(task=tasks[task_id])
                protein_complex_task.set_observable_positions(tasks[task_id]["obs_positions"])
                protein_complex_task.set_wildtype()
                protein_complex_task.set_mutate_protein()
                protein_complex_task.evaluate(of_wrapper=of_wrapper,
                                              store_of_protein=args.store_protein)
                out_path = os.path.join(
                    args.output_dir, f"{safe_filename_part(task_id)}.pkl"
                )
                with open(out_path, "wb") as f:
                    pickle.dump(protein_complex_task, f)
                n_ok += 1
            except Exception:
                logger.exception("Failed to process task %s; skipping", task_id)
                continue
        logger.info("Processed %d/%d tasks successfully", n_ok, len(tasks))


if __name__ == "__main__":
    main()
