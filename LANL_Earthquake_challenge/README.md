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

2. Try longer epoch. (V8)
    there is clear improvement. Whats more, the curve seems to be improving at the end of epoch signalling that I should try to increase it even more.

3. Try even longer epochy (V9)
