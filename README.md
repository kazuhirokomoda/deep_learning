# Deep Learning

# perceptron
Not at all "deep learning", but just as a starting point.

## sample output

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


## issues

- np.dot of horizontal vector (shape: (1,k)) and vertical vector (shape: (k,1)) does not return scalar
- assumes the number of classes is 2
