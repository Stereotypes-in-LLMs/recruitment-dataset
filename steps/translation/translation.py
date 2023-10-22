"""
This script is not implemented yet. 
It will be used to translate the job descriptions and CV from the original language.
But it is not necessary for the first version of the project.
"""

import os
import sys

import dvc.api
import logging
from src.preprocessing import JobsPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare.py input_dir output-dir\n'
    )
    sys.exit(1)


def main():
    # Create output directory
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    input_folder = sys.argv[1]
    params = dvc.api.params_show()
    candidates_input_file_path = os.path.join(input_folder, params['job_prepare']['input_file'])
    candidates_output_file_path = os.path.join(output_dir, params['job_prepare']['output_file'])

    # Preprocessing
    preprocessor = JobsPreprocessor(
        input_path=candidates_input_file_path,
        output_path=candidates_output_file_path
    )
    preprocessor.process()

    # logging output dataset shape
    logging.info(f"Output Jobs dataset shape: {preprocessor.dataset.shape}")

    # Save data
    preprocessor.save_dataset()


if __name__ == "__main__":
    logging.info("Starting Jobs dataset preparation...")
    main()
    logging.info("Finished Jobs dataset preparation.")
