import os
import sys
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import dvc.api
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare.py input_dir output-dir\n'
    )
    sys.exit(1)

def main():
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    input_folder = sys.argv[1]

    params = dvc.api.params_show()
    jobs_file = params['prepare']['jobs_file']
    candidates_file = params['prepare']['candidates_file']

    jobs = pd.read_csv(os.path.join(input_folder, jobs_file))
    candidates = pd.read_csv(os.path.join(input_folder, candidates_file))

    print("Jobs shape: ", jobs.shape)
    print("Candidates shape: ", candidates.shape)

    from IPython.display import display
    display(jobs.head())
    display(candidates.head())


    # save
    # dataset.to_csv(os.path.join(output_dir, 'preprocessed.csv'),index=False)


if __name__ == "__main__":
    logging.info("Starting preparation...")
    main()
    logging.info("Finished preparation.")