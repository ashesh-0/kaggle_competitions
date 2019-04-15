Inference from papers:
    https://doi.org/10.1002/2017GL074677
    https://doi.org/10.1002/2017GL076708
    https://rdcu.be/bdG8Y

    1. Variance of accoustic signal is very important. (one can get R^2 > 0.8 for predicting time to failure.)
    2. Randomized forest works better. It works both for periodic as well as aperiodic events( by aperiodic, it means that successive events are not equally spaced out).
    3. First 10% of seismic cycle has information about whether a large or a small earthquake.
    4. The evolution of acoustic power over the seismic cycle exhibits a memory effect, such that fast events (with large stress drop and energy release) are followed by cycles that start with low acoustic power, whereas cycles following a slow event start at higher acoustic power.


Experiments:
1. Learning rate. (V7)
    [0.01, 0.001, 0.0001, 0.00001, 0.000005]
    With what features I've I'll go with 0.0001 as here training error seems to reduce the most and this is also stable.
    MAE hovers around 3

2. Try larger epoch. (100) (V8)
    there is clear improvement. Whats more, the curve seems to be improving at the end of epoch signalling that I should try to increase it even more.

3. Try even larger epoch (200) (V9)
    after 100, I see fluctuation. However, by looking at playground.tensorflow.org I saw that there are cases when after fluctuations, things stablize. therefore, I'm trying with even larger epoch.

4. Try even larger epoch(400) (V10)
    not much improvement.

5. Using model which performs best on validation set. (V11)
    this time results were not stable.

6. (V12) Increasing both ts_window from 50 to 100 and ts_size from 1000 to 1500. this makes sure that we are using all of 150K data points to predict. Here, performance on training data improved a lot, thereby indicating improvement. However, I don't see improvement in validation set. Also I see that performance saturates within epoch 75. To get a more general idea, in next experiment, I'm varying ts_window and ts_size systematically.

7. (V13) Here, I'm varying (ts_window, ts_size) =>  [(50, 3000), (100, 1500), (150, 1000)]
    (150,1000) is the best of the three. I saw that the improvement was still going on and so will try the longer epoch. This is by far the best model.

8. (V14) (ts_window, ts_size) => (150, 1000) with epoch being 200.

9. (V18) Here, we increased the number of features .however, model performance detoriated. apparently, it is not able to learn. That being the case, we need to do run it for longer time. Here I've applied l1_regularization to see if limiting
the weights improves the performance. [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]

I also saw that 0.00001 regularization was better than 0.0001 as the validation error has come to 2.4. With 9th experiment, I'm hoping to find best performance with no regularization. I'll try dropout as well.

10. (V19) l1 regularization is not helping. with it, training as well as validation performance detoriates.
11. (V20) With 150 epoch I'm trying to learn on 36 features.
12. (V21) With dropout, the performance on validation set improved to 0.24 from 0.3. This too happened when epoch was set to 150 in v19. In v21, epoch was set to 75.
13. (V22) Repeat 12 with more epoch. This is by far the best model.

14. (V23) However, I see that with dropout 0.2 is better than 0.1. So I'm increasing the dropout to [0.2, 0.3, 0.4]
    0.2 -> At 50 epoch, (train,val) => (1.6, 2.6), 100 epoch, (train, val) => (1.4, 2.4)
        -> train very stable.
    0.3 -> At 50 epoch (train,val) => (1.7, 2.7), 100 epoch (train,val)=> (1.4, 2.4)
    0.4 -> At 50 epoch, (train,val) => (1.6419,2.5), 100 epoch (train,val) => (1.4, 2.4)

    I don't see any difference. This means that there is way too much redundency in the network. Keeping dropout at
    0.2, I'll next try to reduce the number of nodes in the network.

