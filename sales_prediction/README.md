# Look at Data analysis
[Notebook link](https://github.com/ashesh-0/kaggle_competitions/blob/master/sales_prediction/notebooks/data-analysis.ipynb)
# Steps to run the code:
1. Unzip the code.
2. Appropriately set the directory in constants.py and in notebooks which you wish to run. Specifically,<br/>
    a. COMPETITION_DATA_DIRECTORY should be the directory which has the data given in the competition<br/>
    b. Directory of DATA_FPATH,TEXT_FEATURE_FPATH,COMBINED_DATA_FPATH should be the path of the extracted preprocessed_data.<br/>

## Packages needed
Install catboost. `pip install catboost`

1. For simply running the prediction, run the notebook FinalSubmission.ipynb (~1 minute)<br/>
2. For running the feature generation and training, run the following in same order:<br/>
    a. For generating a balanced sales dataframe, run the script "test_data_like_train_data.py" (~25 minutes)<br/>
    b. For creating a numeric features train data, run the script "model_data.py" (~1 hour)<br/>
    c. For creating text features for item_id, shop_id and item_category_id, run TextFeatures.ipynb<br/>
    d. For creating the combined data of text and numeric features, run TextDataNumericDataCombined.ipynb<br/>
    e. For training the model, use AllDataModel.ipynb<br/>

3. Run 	Data Analysis.ipynb for EDA.<br/>


# Short Description of what has been done.
Logically speaking, following is the step by step process which the code follows.
1. We first add zero sales entries in sales.csv so that it matches the distribution of monthly sales in test data. We
    know the mean value of monthly sales in test from leaderboard probing.
2. We then create features on top of the sales. They include lagged features, rolling features, oldness of shops, items,
    item shop pairs, text features from name of shops, items and item_categories.
3. We fit a Catboost model.


# Best configuration:
 1. Nearest neighbours [2,4,6,8] with no weights.
