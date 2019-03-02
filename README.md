# kaggle_competitions
Code which was used in different kaggle competitions

Grid search:

{0,3,6,10}: is smoothing
{0.95, 0.9, 0.5}: is peak height min ratio
{20, 15, 10}: is peak threshold.

        Filename with param values                     CV score.
../input/dataprocessing/train_data_0_0.9_15.csv   0.6850970247064757
../input/dataprocessing/train_data_0_0.9_10.csv   0.6754162563014517
../input/dataprocessing/train_data_0_0.5_15.csv   0.6469644574254269


../input/dataprocessing/train_data_10_0.9_15.csv   0.7008880796526709
../input/dataprocessing/train_data_0_0.9_15.csv   0.6852698176650064
../input/dataprocessing/train_data_0_0.5_15.csv   0.6541918636550176
../input/dataprocessing/train_data_10_0.5_15.csv   0.6745023210537178


../input/dataprocessing/train_data_10_0.5_10.csv   0.6608575985942411
../input/dataprocessing/train_data_0_0.5_10.csv   0.6419327879665956
../input/dataprocessing/train_data_0_0.9_10.csv   0.6796122458136029
../input/dataprocessing/train_data_10_0.9_10.csv   0.6650492798003422

Inference:
    15 > 10
    smoothing with 10 > non smoothing
    0.9 > 0.5


../input/dataprocessing/train_data_6_0.95_15.csv   0.7109034354191525
../input/dataprocessing/train_data_6_0.95_20.csv   0.6854081316072402

../input/dataprocessing/train_data_3_0.95_15.csv   0.7334047856776442
../input/dataprocessing/train_data_3_0.95_20.csv   0.7027735392498863


../input/dataprocessing/train_data_3_0.9_15.csv   0.585794928323247
../input/dataprocessing/train_data_3_0.9_20.csv   0.6330471995521202


../input/dataprocessing/train_data_6_0.9_15.csv   0.7039673627815954
../input/dataprocessing/train_data_6_0.9_20.csv   0.7032481145313527

Inference:
    0.95 > 0.9
    if 0.95, then 3 > 6.
    if 0.95, then 15 > 20.
