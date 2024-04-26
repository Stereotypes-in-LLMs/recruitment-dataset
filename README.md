# Recruitment Dataset Preprocessing and Recommender System

## Project Overview
This project aims to preprocess raw data from [Djinni](https://djinni.co) service and develop a recommender system for matching candidates with potential jobs based on anonymized profiles of candidates and job descriptions. The preprocessing involves cleaning and organizing the data, while the recommender system utilizes natural language processing techniques to match candidates with suitable job descriptions.

## Dataset Information
 The Djinni Recruitment Dataset contains 150,000 job descriptions and 230,000 anonymized candidate CVs, posted between 2020-2023 on the [Djinni](https://djinni.co/) IT job platform. The dataset includes samples in English and Ukrainian. 

## Exploratory Data Analysis
The exploratory data analysis (EDA) is provided in the notebook/EDA folder. These analyses offer insights into the characteristics of job descriptions and candidate profiles, aiding in understanding the data distribution and potential patterns.

## Dataset Split and Loading
The preprocessed dataset has been split by languages and loaded into the HuggingFace Dataset Hub for easy access. The following datasets are available:
- [Job Descriptions English](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english)
- [Job Descriptions Ukrainian](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-ukrainian)
- [Candidates Profiles English](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english)
- [Candidates Profiles Ukrainian](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-ukrainian)

## Intended Use

The Djinni dataset is designed with versatility in mind, supporting a wide range of applications:

- **Recommender Systems and Semantic Search:** It serves as a key resource for enhancing job recommendation engines and semantic search functionalities, making the job search process more intuitive and tailored to individual preferences.

- **Advancement of Large Language Models (LLMs):** The dataset provides invaluable training data for both English and Ukrainian domain-specific LLMs. It is instrumental in improving the models' understanding and generation capabilities, particularly in specialized recruitment contexts.

- **Fairness in AI-assisted Hiring:** By serving as a benchmark for AI fairness, the Djinni dataset helps mitigate biases in AI-assisted recruitment processes, promoting more equitable hiring practices.

- **Recruitment Automation:** The dataset enables the development of tools for automated creation of resumes and job descriptions, streamlining the recruitment process.

- **Market Analysis:** It offers insights into the dynamics of Ukraine's tech sector, including the impacts of conflicts, aiding in comprehensive market analysis.

- **Trend Analysis and Topic Discovery:** The dataset facilitates modeling and classification for trend analysis and topic discovery within the tech industry.

- **Strategic Planning:** By enabling the automatic identification of company domains, the dataset assists in strategic market planning.


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

## BibTeX entry and citation info
*When publishing results based on this dataset please refer to:*
```bibtex
@inproceedings{djinni,
    title = "Introducing the {D}jinni {R}ecruitment {D}ataset: A Corpus of Anonymized {CV}s and Job Postings",
    author = "Drushchak, Nazarii  and
              Romanyshyn, Mariana",
    booktitle = "Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @LREC-COLING-2024",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "European Language Resources Association",
}
```

## Contributors
- [Stereotypes-in-LLMs](https://github.com/Stereotypes-in-LLMs)

## License
This project is licensed under the [Apache License 2.0](LICENSE).
