MarchMadness
==============================

An exploration of March Madness data from the Kaggle data set.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
This is the order that you should run the notebooks. For each notebook, make sure you are reading in the correct data. 

1) 2.0-sethchart-data_engineering_tournament-results.ipynb: This notebook reads in MNCAATourneyCompactResults.csv data and processes it into season, teamID (low), teamID (high), and Win which is 0 or 1. 0 means that the LowID lost to HighID, 1 means the LowID beat the HighID. The data is then saved as tournament_outcomes.csv

2) 1.0-theberling_data_explore_2021.ipynb: This notebook reads in and explains what all the data is then uses MRegularSeasonDetailedResults.csv data to make season averages for the stats of the teams. This is combined with the seed data and the average ranking on the last day before the tournament. This data is combined into one dataframe and saved as data_averages.csv

3) 2.0-theberling_merge_datasets.ipynb: This notebook takes tournament_outcomes.csv and data_averages.csv and merges them together by teamID. So for each LowID and HighID in tournament_outcomes.csv, we then have all the LowID stats and HighID stats. This is the data that we use to train the model. This data is saved as model_dataset.csv.

4) 2.1-theberling_merge_datasets.ipynb: This notebook takes the sample submission file, MSampleSubmissionStage1.csv for phase 1, and merges the teamIDs in this data with the data_averages.csv data the same way as the previous notebook. This is data that we will use the trained model to make predictions on for the submission. This data is saved as model_dataset2.csv.

5) 3.1-theberling-classifier.ipynb: Takes data_averages.csv and uses it to train xg boost and gradient boost classifiers. The winning xg boost model is then run on data_averages2.csv to produce phase1_submissions1.csv. This notebook is a little confusing. There is a grid search cv that is run but then I ended up tinkering by hand to get the best accuracy. That's what ends up being used as the model. But I don't want to delete the grid search since that's what we will probably want to use in the future on different hyperparameters. 

Then you're done. Submit your submission file.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
