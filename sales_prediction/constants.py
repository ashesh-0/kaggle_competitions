# Input data which is given in the competition.
COMPETITION_DATA_DIRECTORY = './data/'
SALES_FPATH = COMPETITION_DATA_DIRECTORY + 'sales_train.csv'
ITEMS_FPATH = COMPETITION_DATA_DIRECTORY + 'items.csv'
SHOPS_FPATH = COMPETITION_DATA_DIRECTORY + 'shops.csv'
TEST_SALES_FPATH = COMPETITION_DATA_DIRECTORY + 'test.csv'
ITEM_CATEGORIES_FPATH = COMPETITION_DATA_DIRECTORY + 'item_categories.csv'

# Output of test_data_like_train_data.py
TEST_LIKE_SALES_FPATH = './preprocessed_data/train_with_zero.hdf'

TEST_LIKE_SALES_FKEY = 'df'
# Output of model_data.py
DATA_FPATH = './preprocessed_data/DATA.hdf'

# Text feature are saved to this file. Output of TextFeatures.ipynb
TEXT_FEATURE_FPATH = 'text_features.h5'

# Feature containing both text features and numeric features. Output of TextDataNumericDataCombined.ipynb
COMBINED_DATA_FPATH = 'DATA_WITH_TXT.h5'
