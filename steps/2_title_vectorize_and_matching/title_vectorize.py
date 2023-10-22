import os
import sys

import dvc.api
import logging
import pandas as pd
from src.embedding_store import EmbeddingStore
from src.constants import MODELS_DIM, EMB_LANG_MODELS

# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# check arguments
if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare.py input_dir output_dir param_type\n'
    )
    sys.exit(1)


def main():
    # Create output directory
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    input_folder = sys.argv[1]
    param_type = sys.argv[3]
    params = dvc.api.params_show()
    input_file_path = os.path.join(input_folder, params[param_type]['input_file'])
    output_emb_file_path = os.path.join(output_dir, params['general_matching_vect']['output_emb_file'])
    output_id_file_path = os.path.join(output_dir, params['general_matching_vect']['output_id_file'])
    vectorize_column = params['general_matching_vect']['vectorize_column']

    # load dataframe
    df = pd.read_csv(input_file_path)

    # create embedding store for each language
    for lang in EMB_LANG_MODELS.keys():
        logging.info(f"Processing {lang} language...")
        # create embedding store
        embedding_store = EmbeddingStore(
            model_id=EMB_LANG_MODELS[lang],
            index_path=output_emb_file_path.format(lang=lang),
            id_mapping_path=output_id_file_path.format(lang=lang),
            dim=MODELS_DIM[lang]
        )

        # process dataframe
        lang_column = [col for col in df.columns if col.endswith('_lang')][0]
        embedding_store.process(df[df[lang_column]==lang], vectorize_column)

        # save embedding store
        embedding_store.save()


if __name__ == "__main__":
    logging.info("Starting Embedding Store creation...")
    main()
    logging.info("Finished Embedding Store creation.")
