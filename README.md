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


## fast.ai
fast.ai (http://course.fast.ai/) provides us with "Practical Deep Learning For Coders, Part 1".
The following description in this section utilizes this MOOC course.

```
# conda virtual environment
conda info --envs
source activate py3tf

# after moving to the directory you like, for example
cd ~/git; pwd
git clone https://github.com/kazuhirokomoda/deep_learning.git
cd deep_learning/fast.ai; pwd

# or alternatively
cd ~/git/deep_learning/fast.ai; pwd
```

### a) local environment

under the virtual environment (py3tf), 
```
jupyter notebook

```

```
■ 仮想環境py3tfに足りなくてconda install (またはpip install)したモジュール
matplotlib
Theano
keras
Pillow (PIL)
scikit-learn
bcolz
h5py
```


### b) cloud environment (with GPU)
AWS p2 instances can be used by following [the course instruction](http://course.fast.ai/lessons/aws.html).

Meanwhile, mainly for my personal use, I list up commands to use [FloydHub](https://www.floydhub.com/): Heroku for Deep Learning.
Please refer to [this great repository](https://github.com/YuelongGuo/floydhub.fast.ai) for explanations because it shows an example of how to setup instance on floydhub to run lesson 1 of the course.

#### run with sample data

```
# check bcolz is included
cat floyd_requirements.txt

# initiate a cloud instance
floyd init your_favorate_task_name_e.g._neural_networks

# start a GPU instance with Jupyter notebook
# floyd run --mode jupyter --env theano:py2 --gpu
sh floyd-gpu.sh

# access jupyter notebook URL
# start a terminal (console) session in jupyter notebook. inside the terminal, run
sh setup.sh

# start running jupyter notebook of your choice
```

#### upload data via FloydHub and use it

```
cd ~/git/deep_learning/fast.ai; pwd

# download the dogs and cats dataset to an empty folder
mkdir floydhub.data.zipped
cd floydhub.data.zipped; pwd
wget http://files.fast.ai/data/dogscats.zip

# upload the zipped dataset to floydnet, and create a floydnet dataset
floyd data init dogscats.zipped
floyd data upload

# unzip the data on floydnet
# 1. the data ID should be the one you see from the above step
# 2. the mounted data is available in /input/ directory, and you need to direct the unzipped files to /output/ directory
mkdir floydhub.data.unzip
cd floydhub.data.unzip; pwd
floyd init dogscats.unzip
floyd run --gpu --data [DATA ID] "unzip /input/dogscats.zip -d /output"

# run
cd ~/git/deep_learning/fast.ai; pwd
mv data/ data.backup/
floyd init dogscats
floyd run --mode jupyter --data [DATA ID] --env theano:py2 --gpu
```

for example,
```
(py3tf)➜  floydhub.fast.ai.data.unzip git:(master) ✗ floyd run --gpu --data BSNUitG8kVN746BwWUuG8E "unzip /input/dogscats.zip -d /output"
Creating project run. Total upload size: 282.0B
Syncing code ...
Done=============================] 1031/1031 - 00:00:00
RUN ID                  NAME                        VERSION
----------------------  ------------------------  ---------
TQjERgHoT9TbjR3rQJrUyU  sunrize/dogscats.unzip:1          1


To view logs enter:
    floyd logs TQjERgHoT9TbjR3rQJrUyU

(py3tf)➜  floydhub.fast.ai.data.unzip git:(master) ✗ floyd logs TQjERgHoT9TbjR3rQJrUyU
```


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

### fast.ai
- http://course.fast.ai/
- [Anacondaのcondaコマンドによる仮想環境の使い方のまとめ](http://minus9d.hatenablog.com/entry/2016/01/29/235916)
- [nice troubleshooting for running lesson1 on floydhub](https://shaoanlu.wordpress.com/2017/03/06/floydhub/)
- [Fast.AI の Practical Deep Learning for Coders Part1 を受けた](http://futurismo.biz/archives/6440)
