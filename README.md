# Stereotypes-LLMs

**Project install steps:**
- clone repo
`git clone https://github.com/naz2001r/Stereotypes-LLMs`
- create virtual environment
`conda create --name py311 python=3.11`
- activate virtual environment 
`conda activate py311` (conda)

We are using poetry to manage dependencies.
- install poetry
`pip install poetry`
- install dependecies using poetry. It could take some time
`poetry install`
- pull needed data using dvc (it can take up to 30 minutes, please be patient, to see the progress we use verbose flag in here)
`dvc pull -v`
- reproduce training pipeline (actually all steps should be skipped, that will mean that you have latest data locally)
`dvc repro -v`
- to run single step of pipeline
`dvc repro -v -sf STEPNAME`
- to run all steps of pipeline after some step
`dvc repro -v -f STEPNAME --downstream`
- to run all steps of pipeline after some step without running them actually
`dvc repro -v -f STEPNAME --downstream --dry`

[DVC documentation](https://dvc.org/doc/start/data-management/data-versioning)