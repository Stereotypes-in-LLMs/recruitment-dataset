import os
import sys

import dvc.api
import logging
import pandas as pd
from src.matching import GroupMatcher


# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# check arguments
if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare.py input_candidates_dir input_jobs_dir output_dir\n'
    )
    sys.exit(1)


def main():
    # Create output directory
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    input_candidates_folder = sys.argv[1]
    input_jobs_folder = sys.argv[2]
    params = dvc.api.params_show()
    input_candidates_file_path = os.path.join(input_candidates_folder, params['groups_matching']['input_candidates'])
    input_jobs_file_path = os.path.join(input_jobs_folder, params['groups_matching']['input_jobs'])
    output_emb_file_path = os.path.join(output_dir, params['groups_matching']['output_file'])

    # Create GroupMatcher object
    group_matcher = GroupMatcher(
        df_candidates_path=input_candidates_file_path,
        df_jobs_path=input_jobs_file_path,
        output_path=output_emb_file_path
    )

    # Match candidates with jobs
    group_matcher.process()

    # Save matched groups
    group_matcher.save()


if __name__ == "__main__":
    logging.info("Starting First Step Jobs Groups creation for each candidate...")
    main()
    logging.info("Finished First Step Jobs Groups creation for each candidate.")
