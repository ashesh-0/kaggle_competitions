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

8. (ts_window, ts_size) => (V14) (150, 1000) with epoch being 200.

