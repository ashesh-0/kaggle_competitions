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


15. (V24) Here we vary the number of hidden nodes of Rnn and dense layers. RNN has num_nodes output dimension. Dense layer has num_nodes/2 size. [20, 40, 60]
    20-> At 50 epoch, (train,val) => (1.9, 2.7). At 100 epoch, (train, val) => (1.7, 2.5)
    40 -> At 50 epoch, (train, val)=> (1.7, 2.6). At 100 epoch, (train, val) => (1.5, 2.5)
    60 -> At 50 epoch, (train, val) => (1.5, 2.6). At 100 epoch, (train,val) => (1.4, 2.4)
    I don't see much difference.

16. (V25) Here, I kept the num_nodes to be 64. Dropout of 0.5 was used. Used batch normalization. Trained till 250 seconds. Very stable performance on validation set. However, performance numbers are high (train, val) => (3, 3.2)

17. (V26),(V27) I'll increase one more layer in the network. Idea is to increase the complexity. Dropout of 0.5 was too high. 0.3 looked better. However, results were still improving in 0.3 case.
18. (V28) Repeat (V27) (0.3 dropout) with larger epoch (300).
19. (V30), (V29) Was trying different network architectures. [RNN(40), Dense(20),Dense(20), Dense(10)]. deep architectures don't seem to converge. I'll try batch normalization
20. With batch normalization, performance is extremely stable.However, training loss does not come down below 2.9. So either there is some issue in deep architectures (bottleneck layer is too small) or the learning rate is too low.
21. I'm now trying to see with 10x learning rate (0.01) whether I'm able to get better train performance. Better performance is not observed.
22. (V31) Network is [RNN(64),Dense(40)]. This should get better train performance and hopefully better val performance. It has better performance. Validation performance reaches 2.41. PL score is 1.64. Dropout is 0.5. Normalization is disabled.
23. (V32) Network is [RNN(64),Dense(64)]. Validation performance reaches 2.41. PL score is 1.62. Epoch  250. Dropout is 0.3. Saturation reached around 100 epoch. It did not improve after that.
24. (V34) After fixing a bug which did not effect performance but underfed the number of nodes in dense layer. I'm running the network Network is [RNN(64),Dense(128)]

25. With LSTM, I'm getting better train accuracy. However, the validation score has degraded.

26. (V37) Here, I've added code to see prediction plot. Using the best model to run for 50 epoch and trying to find what prediction plot looks like.

27. I've also added support for TensorBoard. With it, one can see the gradients. I see that recurrent kernel gradients are too low for GRU. No node has gradient more than this. 0.00004. I'm trying LSTM to see how its recurrent gradients look like. kernel gradient has its max at around 0.01. Dense layer also has decent gradients in the range of 0.01.

28. RNN layer did not had an activation :(. Added it and made the commit to run 150 epoch.

