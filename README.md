# Recruitment Dataset Preprocessing and Recommender System

## Project Overview
This project aims to preprocess raw data from [Djinni](https://djinni.co) service and develop a recommender system for matching candidates with potential jobs based on anonymized profiles of candidates and job descriptions. The preprocessing involves cleaning and organizing the data, while the recommender system utilizes natural language processing techniques to match candidates with suitable job descriptions.

## Exploratory Data Analysis
The exploratory data analysis (EDA) is provided in the notebook/EDA folder. These analyses offer insights into the characteristics of job descriptions and candidate profiles, aiding in understanding the data distribution and potential patterns.

## Dataset Split and Loading
The preprocessed dataset has been split by languages and loaded into the HuggingFace Dataset Hub for easy access. The following datasets are available:
- [Job Descriptions English](https://huggingface.co/datasets/Stereotypes-in-LLMs/recruitment-dataset-job-descriptions-english)
- [Job Descriptions Ukrainian](https://huggingface.co/datasets/Stereotypes-in-LLMs/recruitment-dataset-job-descriptions-ukrainian)
- [Candidates Profiles English](https://huggingface.co/datasets/Stereotypes-in-LLMs/recruitment-dataset-candidate-profiles-english)
- [Candidates Profiles Ukrainian](https://huggingface.co/datasets/Stereotypes-in-LLMs/recruitment-dataset-candidate-profiles-ukrainian)

## Pipeline Management with DVC
The pipeline for preprocessing and creating the recommender system has been managed using Data Version Control (DVC). DVC ensures reproducibility and tracks the dependencies and outputs of each step in the pipeline. Final outputs are JSON files with candidate IDs as keys and a list of matched job description IDs as values.

## Installation Instructions
Follow these steps to install and set up the project:

### Prerequisites
- Git installed on your system
- Conda installed (for creating and managing virtual environments)
- Python 3.11 installed

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/Stereotypes-in-LLMs/recruitment-dataset
    ```

2. Create a virtual environment using Conda:
    ```bash
    conda create --name py311 python=3.11
    ```

3. Activate the virtual environment:
    ```bash
    conda activate py311
    ```

4. Install Poetry for dependency management:
    ```bash
    pip install poetry
    ```

5. Install dependencies using Poetry:
    ```bash
    poetry install
    ```

6. Pull the necessary data using DVC (this may take some time):
    ```bash
    dvc pull -v
    ```

7. Reproduce the training pipeline (all steps should be skipped if data is already up to date locally):
    ```bash
    dvc repro -v
    ```

## Running the Pipeline
- To run a single step of the pipeline:
    ```bash
    dvc repro -v -sf STEPNAME
    ```

- To run all steps of the pipeline after a certain step:
    ```bash
    dvc repro -v -f STEPNAME --downstream
    ```

- To simulate running all steps without actually running them:
    ```bash
    dvc repro -v -f STEPNAME --downstream --dry
    ```

For more information on DVC, refer to the [documentation](https://dvc.org/doc/start/data-management/data-versioning).

## Contributors
- [Stereotypes-in-LLMs](https://github.com/Stereotypes-in-LLMs)

## License
This project is licensed under the [Apache 2.0 License](LICENSE).
