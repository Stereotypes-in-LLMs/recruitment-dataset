import os
import sys

import dvc.api
import logging
import pandas as pd
from src.matching import FinalMatching


# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# check arguments
if len(sys.argv) != 6:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare.py input_candidates_dir groups_folder_path emb_jobs_folder_path emb_candidates_folder_path output_dir\n'
    )
    sys.exit(1)


def main():
    # Create output directory
    output_dir = sys.argv[5]
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    input_candidates_folder = sys.argv[1]

    groups_folder_path = sys.argv[2]
    emb_jobs_folder_path = sys.argv[3]
    emb_candidates_folder_path = sys.argv[4]


    params = dvc.api.params_show()
    input_file_path = os.path.join(input_candidates_folder, params['groups_matching_final']['input_file'])
    output_file_path = os.path.join(output_dir, params['groups_matching_final']['output_file'])

    # Create FinalMatching object
    group_matcher = FinalMatching(
        job_groups_folder_path = groups_folder_path,
        df_candidates_path = input_file_path,
        emb_candidates_folder_path = emb_candidates_folder_path,
        emb_jobs_folder_path = emb_jobs_folder_path,
        output_path = output_file_path,
        vec_dim = params['vec_dim'],
    )

    # Match candidates with jobs
    group_matcher.process()

    # mean number of jobs per candidate
    jobs_per_candidate = [len(jobs_ids) if jobs_ids else 0 for _, jobs_ids in group_matcher.job_groups.items()]
    logging.warning('Mean number of jobs per candidate: {}'.format(
        round(sum(jobs_per_candidate) / len(jobs_per_candidate), 2)
    ))

    # Save matched groups
    group_matcher.save()


if __name__ == "__main__":
    logging.info("Starting Finall Step Jobs Groups creation for each candidate...")
    main()
    logging.info("Finished Finall Step Jobs Groups creation for each candidate.")
