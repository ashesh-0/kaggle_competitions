# Packages needed
Install catboost. `pip install catboost`

# Steps to run the code:
1. Run addingzerototrain.ipynb notebook to add zero sales rows to sales_train.csv to generate a new csv file.
2. Run treebasedapproachfeaturegeneration.ipynb to generate features on this new sales data.
    1. They include mean encoding, lag features, rolling features, city based features.
3. Run nn-features.ipynb to add Nearest neighbour features to already generated features.
4. Run textfeatures.ipynb to generate text based features from item name, category name and shop name.
5. Run textdatanumericdatacombined.ipynb to combine text features with numeric features computed in steps 2 and 3.
6. Run alldatamodel.ipynb to fit a catboost model on the generated features. When FINAL_MODEL_FOR_TEST is set to False,
model is trained on data with date_block_num <32 and validation set is last two months data (date_block_num being 32 and 33.). When FINAL_MODEL_FOR_TEST is True, whole data is used to generate the model and to predict on test data.

# Analysis
1. Run data-analysis.ipynb for data analysis.

# Short Description of what has been done.
Following is the step by step process which the code follows.
1. We first add zero sales entries in sales.csv so that it matches the distribution of monthly sales in test data. From analysis, we know that in test data, there is an entry for every item which was getting traded last month and sho which had sales in last month. So there must be lot of (shop_id, item_id) pair entries which will have zero sales. This is also clear from the fact that test data has about 8 times number of entries than train sales data for october 2015.

2. We then create features on top of the sales. They include
    1. Lagged features
    2. Rolling features
    3. Oldness features (How early was the first month when it got traded) on shop_id, item_id, item_category_id, item shop pairs.
    4. Last traded features on same ids as Oldness features.
    5. Text features (TFIDF + PCA) from name of shops, items and item_categories.
    6. Nearest neighbour features. Nearest neighbour is computed using text feature on item names.

3. We fit a Catboost model.
