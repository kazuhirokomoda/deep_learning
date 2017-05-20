# Deep Learning

## Perceptron
Not at all "deep learning", but just as a starting point.

### sample output

```
[[-2]
 [ 5]
 [-4]]
9
10
9
3
5
4
4
2
2
4
1
4
3
0
[[-1.75917122]
 [ 4.06657637]
 [-3.52379369]]
# of correctly predicted data: 100
ratio:  1.0
```

The sample result of draw_graph function can be seen in [perceptron/perceptron.png](perceptron/perceptron.png).

### issues

- np.dot of horizontal vector (shape: (1,k)) and vertical vector (shape: (k,1)) does not return scalar
- assumes the number of classes is 2


## Multi Layer Perceptron
has the ability to solve some linear inseparable problems.

### sample output

```
epoch:     0, loss: 2.01165
epoch:  1000, loss: 1.47072
epoch:  2000, loss: 1.48494
epoch:  3000, loss: 1.34211
epoch:  4000, loss: 1.34242
epoch:  5000, loss: 1.33782
epoch:  6000, loss: 1.34714
epoch:  7000, loss: 1.35117
epoch:  8000, loss: 1.35228
epoch:  9000, loss: 1.36138
[0 0] 0 [[ 0.32544176  0.67456832]]
[0 1] 1 [[ 0.97572499  0.02427763]]
[1 0] 1 [[ 0.32518525  0.67472581]]
[1 1] 0 [[ 0.32544183  0.67456823]]
epoch:     0, loss: 3956.91455
epoch:  1000, loss: 1349.92425
epoch:  2000, loss: 739.92657
epoch:  3000, loss: 439.50536
epoch:  4000, loss: 358.56261
epoch:  5000, loss: 240.14291
epoch:  6000, loss: 243.83579
epoch:  7000, loss: 219.30163
epoch:  8000, loss: 170.61104
epoch:  9000, loss: 155.69228
[[11  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0]
 [ 0  0 18  0  0  0  0  0  0  0]
 [ 0  0  0 19  0  0  0  1  0  0]
 [ 0  0  0  0 18  0  0  1  0  0]
 [ 0  0  0  0  0 22  1  0  0  1]
 [ 0  0  0  0  0  0 18  0  0  0]
 [ 0  0  0  0  0  0  0 16  0  0]
 [ 0  1  0  0  0  0  1  0 18  0]
 [ 0  0  0  0  0  0  0  0  0 19]]
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        11
          1       0.94      1.00      0.97        15
          2       1.00      1.00      1.00        18
          3       1.00      0.95      0.97        20
          4       1.00      0.95      0.97        19
          5       1.00      0.92      0.96        24
          6       0.90      1.00      0.95        18
          7       0.89      1.00      0.94        16
          8       1.00      0.90      0.95        20
          9       0.95      1.00      0.97        19

avg / total       0.97      0.97      0.97       180
```

### issues

- ~~label is not yet expressed as 1-of-K expression~~



## references

### perceptron
- http://gihyo.jp/dev/serial/01/machine-learning/0017
- http://qiita.com/murataR/items/74a3a89ffcccb688d71f

### mlperceptron
- http://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
- http://hokuts.com/2016/10/09/bp2/
- http://aidiary.hatenablog.com/entry/20140201/1391218771
- http://btgr.hateblo.jp/entry/2016/05/17/092903
- https://github.com/yusugomori/DeepLearning/blob/master/python/MLP.py
