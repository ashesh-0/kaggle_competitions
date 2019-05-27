First step:
    1. Unzip preprocessed_data.zip
    2. Appropriately set the directory in constants.py and in notebooks which you wish to run. Specifically,
        a. COMPETITION_DATA_DIRECTORY should be the directory which has the data given in the competition
        b. Directory of DATA_FPATH,TEXT_FEATURE_FPATH,COMBINED_DATA_FPATH should be the path of the extracted preprocessed_data.


1. For simply running the prediction, run the notebook FinalSubmission.ipynb (~1 minute)
2. For running the feature generation and training, run the following in same order:
    a. For generating a balanced sales dataframe, run the script "test_data_like_train_data.py" (~25 minutes)
    b. For creating a numeric features train data, run the script "model_data.py" (~ 1 hour)
    c. For creating text features for item_id, shop_id and item_category_id, run TextFeatures.ipynb
    d. For creating the combined data of text and numeric features, run TextDataNumericDataCombined.ipynb
    e. For training the model, use AllDataModel.ipynb

3. Run 	Data Analysis.ipynb for EDA.



Logically speaking, following is the step by step process which the code follows.
1. We first add zero sales entries in sales.csv so that it matches the distribution of monthly sales in test data. We
    know the mean value of monthly sales in test from leaderboard probing.
2. We then create features on top of the sales. They include lagged features, rolling features, oldness of shops, items,
    item shop pairs, text features from name of shops, items and item_categories.
3. We fit a Catboost model.
