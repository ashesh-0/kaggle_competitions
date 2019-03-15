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

no flip, Number of shifts in data augumentation.
../input/dataprocessing/train_data_3_0.95_15.csv   1   0.6932659614617658
../input/dataprocessing/train_data_3_0.95_15.csv   2   0.7126828631043923
../input/dataprocessing/train_data_3_0.95_15.csv   3   0.7420054133931855
../input/dataprocessing/train_data_3_0.95_15.csv   5   0.7332889767699805
../input/dataprocessing/train_data_3_0.95_15.csv   8   0.7441791916963365


without flip
../input/dataprocessing/train_data.csv   3   0.6822760564347564


no flip, number of shifts is 2. here we vary
    1. peak height min ratio [0.95, 0.98]
    2. corona_cleanup_distance [10, 50, 100]

../input/dataprocessing/train_data_0.98_10.csv   2   0.7234839640472763 (thresh: 0.79, train score: 1)
    (0.7, 0.12, stable)
    (0.6, 0.15, almost stable)
    (0.65, 0.15, almost stable)

../input/dataprocessing/train_data_0.95_10.csv   2   0.7285998586761844 ( thresh: 0.37, train score: 0.98)
    (0.6, 0.12, stable)
    (0.6, 0.16, almost stable)
    (0.68, 0.16, almost stable )

../input/dataprocessing/train_data_0.98_100.csv   2   0.6738666577808261 (thresh: 0.49, train score: 0.99)
    (0.58, 0.16, almost stable)
    (0.58, 0.16, almost stable)
    (0.58, 0.2, almost stable)

../input/dataprocessing/train_data_0.98_50.csv   2   0.6605525077983523 (thres: 0.4, train score: 0.80)
    (0.6,0.17, less stable)
    (0.58, 0.18, less stable)
    (0.6, 0.2, less stable)

../input/dataprocessing/train_data_0.95_50.csv   2   0.6119077395408501 (thres: 0.88, train score: 0.98)
    (0.7, 0.14, stable)
    (0.5, 0.2, not stable)
    (0.6, 0.2, less stable)

../input/dataprocessing/train_data_0.95_100.csv   2   0.6683054446658114 (thres: 0.44, train score: 0.96)
(0.58, 0.2, less stable)
(0.6, 0.2, less stable)
(0.6, 0.2, less stable)

Inference:
1. corona cleanup distance 10 is better.
2. 0.95 ~ 0.98 when looking at validation. however, 0.95 is better than 0.98 when looking at train data.

Repeating above experiment with data augumentation disabled.
no flip, no shifts. here we vary
    1. peak height min ratio [0.95, 0.98]
    2. corona_cleanup_distance [10, 50, 100]


../input/dataprocessing/train_data_0.98_10.csv   1   0.6244669709364603 (thresh: 0.12)
    (0.62, 0.2, almost stable)
    (0.6,0.2, stable)
    (0.7, 0.15, stable)

../input/dataprocessing/train_data_0.95_10.csv   1   0.710795182946097 (thresh: 0.25)
    (0.68, 0.15, stable)
    (0.6, 0.17, stable)
    (0.7, 0.17, stable)

../input/dataprocessing/train_data_0.98_100.csv   1   0.6207618409819141 (thresh: 0.7)
    (0.6, 0.2, not so stable)
    (0.58, 0.2, not so stable)
    (0.6, 0.15, not so stable)

../input/dataprocessing/train_data_0.98_50.csv   1   0.6660437333135646 (thresh: 0.4)
    (0.6, 0.16, not so stable)
    (0.6, 0.2, not stable)
    (0.68, 0.15, stable)

../input/dataprocessing/train_data_0.95_50.csv   1   0.6834901342978863 (thresh 0.68)
    (0.6, 0.17, stable)
    (0.5, 0.2, not so stable)
    (0.68, 0.15)

../input/dataprocessing/train_data_0.95_100.csv   1   0.6268047759323415 (thresh 0.83)
    (0.6, 0.15, stable)
    (0.6, 0.2, not so stable)
    (0.6, 0.2)

Inference:
    0.95 > 0.98
    10 is best in terms of stablility.
    We also need to look at how threshold is getting generated. It is very fluctuating across models.

When size is 4 times the original, in about 20 iterations, model gets to its best performace with lowest loss.
So 20 epoch is good enough for it.
Also, as far as threshold is concerned, I'll add a logic which will keep threshold to something around 0.5

Varying number of time steps. with 0.95 as peak height min ratio and 10 as corona_cleanup_distance:
    1. ../input/dataprocessing/train_data_0.95_10_50.csv   3   0.6657940642220486 (thresh: 0.5)
    2. ../input/dataprocessing/train_data_0.95_10_100.csv   3   0.6867873563067558 (thresh: 0.5)
    3. ../input/dataprocessing/train_data_0.95_10_200.csv   3   0.7242895326694354 (thresh: 0.5)


Here, on 200, we vary number of data_augumentation shifts. As before, we have 0.95, 10 set.
    ../input/dataprocessing/train_data_0.95_10_200.csv   1   0.7296492866635081 (threshold 0.5)
    (0.7, 0.15, stable, ~1)
    (0.7, 0.14, stable, ~1)
    (0.68, 0.16, stable, ~1)

    ../input/dataprocessing/train_data_0.95_10_200.csv   2   0.745963110142455 (threshold 0.5)
    (0.8, 0.12, stable, ~1)
    (0.6, 0.2, unstable, ~1)
    (0.7, 0,15, stable, ~1)

    Slightly Bizaare fitting threshold. It peaks at around 0.8. However, difference is of 0.03

    ../input/dataprocessing/train_data_0.95_10_200.csv   3   0.7132573898549575 (threshold 0.5)
    (0.7, 0.18, not so stable, ~1)
    (0.7, 0.15, almost stable, ~1)
    Similar pattern for peak: it peaks around 0.9 ?. Anything below 0.9 is bad.

    Need to repeat this with threshold tuning params so that it switches to something apart from 0.5

Repeating previous experiment( kaggle version 50):
    ../input/dataprocessing/train_data_0.95_10_200.csv   1   0.7269477621848228 (threshold : 0.5)
    (0.7, 0.15, stable)
    (0.7, 0.12, stable)
    (0.7, 0.12, stable)

    ../input/dataprocessing/train_data_0.95_10_200.csv   2   0.7084346924448323 (threshold: 0.5)
    (0.6, 0.18,  stable)
    (0.7, 0.15, stable)
    (0.7, 0.13, stable)

    ../input/dataprocessing/train_data_0.95_10_200.csv   3   0.723772752462884
    (0.6, 0.2, not stable)
    (0.7, 0.18, not stable)
    (0.68, 0.17, almost stable)

    Here, I dont see that increasing data is improving performance for validation data.

Dropout is placed between LSTM unit and dense layer.
Dropout is varied [0.3, 0.2, 0.1] with data_aug_num_times kept to 2:

    (dropout 0.3) ../input/dataprocessing/train_data_0.95_10_200.csv   2   0.6514688422958768 (threshold: 0.88)
    (0.65, 0.12, stable)
    (0.7, 0.1, very stable)
    (0.65, 0.12, stable)

    (dropout 0.2) ../input/dataprocessing/train_data_0.95_10_200.csv   2   0.716260474169257 (threshold: 0.5)
    (0.7, 0.18, stable)
    (0.7, 0.12, stable)
    (0.7, 0.13, stable)

    (dropout 0.1) ../input/dataprocessing/train_data_0.95_10_200.csv   2   0.7149732665597952 (threshold: 0.5)
    (0.68, 0.18, stable)
    (0.7, 0.12, stable)
    (0.7, 0.18, almost stable)

    Stablity is amazing with dropout. It is even more in terms of loss. In terms of metric in consideration as well it is more stable than naive case.

    One more point which I see that, running same configuration twice leads to improvements of about 0.3
    in magnitude. So one possible way of improving performance is to simply run it again and again.

Dropout is varied with data_aug_num_times kept to 1( kaggle version 51):
Dropout is varied with data_aug_num_times kept to 3:

(0.3) ../input/dataprocessing/train_data_0.95_10_200.csv   3   0.67 (threshold: 0.69)
    (0.7, 0.15, less stable)
    (0.7, 0.12, stable)
    (0.7, 0.14, less stable)

(0.2)
    (0.7, 0.15, less stable)

For lstm, it might make sense to keep dropout before lstm unit. Same dropout mask will be applied to all timestamps.

There were some features whose distribution (mean, quantiles) was very different in test when compared with train. That being so, few of those features were removed. (version 54 kaggle)
This version improved the score to 0.55 lb score(We removed few features. It could be fluke. Or it could be that it worked. This also shows that dropout just after input will also improve the performance)


In other public kernels, I see that the train score does not go to 1.That means that we have way too many features. Earlier, I had approached this problem as remove few features. However, now I feel that
we should not concatenate the three phases data naively. this will reduce the features by a factor of 3. also several factors out there are highly related. so it makes sense to not feed in all 3 phase features.

Varying dropouts, when dropout is placed after LSTM unit (0.3, 0.2, 0.1)
Now we have just 56 features, 200 timestamps. X shape (8712, 200, 56)
0.1 ../input/dataprocessing/train_data_0.95_10_200.csv   1   0.6264057474890197
    (0.6, 0.15, stable, train: 0.8)
    (0.65, 0.15, stable, train: 0.8)
    (0.6, 0.15, stable, train: 0.8)

(0.2) ../input/dataprocessing/train_data_0.95_10_200.csv   1   0.637695954120573 (threshold: 0.5)

    (0.6, 0.18, stable, train: 0.8)
    (0.7, 0.1, stable, train: 0.7)
    (0.5, 0.15, stable, train: 0.8)

(0.3) ../input/dataprocessing/train_data_0.95_10_200.csv   1   0.6405981443429222 (threshold: 0.4)

    (0.6, 0.18, stable, train: 0.6)
    (0.65, 0.5,, stable, train: 0.7)
    (0.6, 0.15, stable, train: 0.75)

Varying dropouts when dropout is placed just after input and after LSTM layer.
It is clear case of underfitting. with 0.1 as rate, results are as low as 0.2 on training data.


Phase data is being added now. I've picked a stage when performance on LB was good.
I see that the difference in signal is coming in 0.95, 0.99 percentiles and not in 0.5 percentile. I'm therefore tweaking it to be that.I also see that the corona removal doesn't seem to work. So I changed that a bit.

In the meanwhile, on the basis of the paper, I added a feature which tells which of the four segments does a timestamp belong to. In the paper, it is mentioned that in 2nd and 4th segment of the sinosoidal signal, partial discharge happens more often than not. There is an improvement on dev set performance with that change.
