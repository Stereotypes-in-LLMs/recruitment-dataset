import os
import sys

import dvc.api
import logging
from src.preprocessing import CandidatesPreprocessor

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
    candidates_input_file_path = os.path.join(input_folder, params['candidate_prepare']['input_file'])
    candidates_output_file_path = os.path.join(output_dir, params['candidate_prepare']['output_file'])

    # Preprocessing
    preprocessor = CandidatesPreprocessor(
        input_path=candidates_input_file_path,
        output_path=candidates_output_file_path
    )
    preprocessor.process()

    # logging output dataset shape
    logging.info(f"Output Candidates dataset shape: {preprocessor.dataset.shape}")

    # Save data
    preprocessor.save_dataset()


if __name__ == "__main__":
    logging.info("Starting Candidates dataset preparation...")
    main()
    logging.info("Finished Candidates dataset preparation.")
