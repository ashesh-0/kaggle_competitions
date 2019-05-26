1. For simply running the prediction, run the notebook FinalSubmission.ipynb (~1 minute)
2. For running the training, run the notebook "Catboost Model.ipynb" (~15 minutes)
3. For creating a pre-processed train data, run the script "model_data.py" (~ 1 hour)
4. For generating a balanced sales dataframe, run the script "test_data_like_train_data.py" (~25 minutes)

Step by step procedure is
1. We first add zero sales entries in sales.csv so that it matches the distribution of monthly sales in test data.
2. We then create features on top of the sales.
3. We fit a Catboost model.
