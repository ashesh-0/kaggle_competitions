First step:
    1. Unzip preprocessed_data.zip
    2. Appropriately set the directory in constants.py. Specifically,
        a. COMPETITION_DATA_DIRECTORY should be the directory which has the data given in the competition
        b. DATA_FPATH should be the directory of the extracted preprocessed_data

    In notebooks, please set the directories appropriately

1. For simply running the prediction, run the notebook FinalSubmission.ipynb (~1 minute)
2. For running the feature generation and training, run the following in order:
    a. For generating a balanced sales dataframe, run the script "test_data_like_train_data.py" (~25 minutes)
    b. For creating a pre-processed train data, run the script "model_data.py" (~ 1 hour)
    c. Notebook "Catboost Model.ipynb" (~15 minutes)
3. Run 	Data Analysis.ipynb for EDA.



Logically speaking, following is the step by step process which the code follows.
1. We first add zero sales entries in sales.csv so that it matches the distribution of monthly sales in test data. We
    know the mean value of monthly sales in test from leaderboard probing.
2. We then create features on top of the sales.
3. We fit a Catboost model.
