## ロジスティック回帰による2クラス分類（備忘録）

ロジスティック回帰による2クラス分類を実装する。

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
```



### 1.データの生成

　クラス分類に使用するデータを生成する。今回は2次元平面上に一様乱数により生成した100個のデータを用意した。 i番目のデータが![$w_1 x_1^{(i)} + w_2 x_2^{(i)} + b &gt; 0$](https://render.githubusercontent.com/render/math?math=w_1%20x_1%5E%7B%28i%29%7D%20%2B%20w_2%20x_2%5E%7B%28i%29%7D%20%2B%20b%20%26gt%3B%200&mode=inline)ならば![$1$](https://render.githubusercontent.com/render/math?math=1&mode=inline)、そうでなければ![$0$](https://render.githubusercontent.com/render/math?math=0&mode=inline)に分類されているようなデータセットが得られていると仮定して、以下ではこの決定境界の推定を行う。

```python
w1 = 1.0
w2 = -2.0
b = 10
data_x = np.array([[10.0*np.random.uniform() for i in range(0,100)],[10.0*np.random.uniform() for i in range(0,100)]])
data_y = np.zeros(data_x.shape[1])

for i in range(data_x.shape[1]):
    if w1 * data_x[0][i] + w2 * data_x[1][i] + b >= 0.0 :
        data_y[i] = 1.0 
    else:
        data_y[i] = 0.0
```

データの形は

![$$ X.shape = (2,100) $$](https://render.githubusercontent.com/render/math?math=X.shape%20%3D%20%282%2C100%29&mode=display)

![$$ Y.shape = (100,) $$](https://render.githubusercontent.com/render/math?math=Y.shape%20%3D%20%28100%2C%29&mode=display)

となっている。

データの先頭5つはこんな感じ。

```python
df = pd.DataFrame(data_x.T,columns=["x1","x2"])
df['target'] = data_y
df.head()
```



|      | x1       | x2       | target |
| :--- | :------- | :------- | :----- |
| 0    | 7.824934 | 5.797468 | 1.0    |
| 1    | 1.229090 | 6.280158 | 0.0    |
| 2    | 6.371579 | 5.383301 | 1.0    |
| 3    | 5.896936 | 9.295513 | 0.0    |
| 4    | 4.066826 | 9.072754 | 0.0    |



ペアプロットはこんな感じ。

```python
sns.pairplot(df,hue="target")
data_y = data_y.reshape(1,data_y.shape[0])
```
![img](https://user-images.githubusercontent.com/54795218/100469851-6a769880-311a-11eb-9fa4-ce4bdda4d6b1.png)

後の都合のため、Yの形を以下の形に変形した。

![$$ Y.shape = (1,100) $$](https://render.githubusercontent.com/render/math?math=Y.shape%20%3D%20%281%2C100%29&mode=display)

### 2.ロジスティック回帰

　以下ではモデル部分の実装を行う。

まず、重みの初期化を行う部分を定義する。

```python
def initialize(dim):
    w = np.zeros((dim,1))
    b = 0.0
    return w,b
```

次にシグモイド関数を定義する:

![$$ \sigma(z) = \frac{1}{1+\exp{(-z)}} $$](https://render.githubusercontent.com/render/math?math=%5Csigma%28z%29%20%3D%20%5Cfrac%7B1%7D%7B1%2B%5Cexp%7B%28-z%29%7D%7D&mode=display)

ちなみに、シグモイド関数の微分は、

![$$ \sigma(z)' = (1-\sigma(z))\sigma(z) $$](https://render.githubusercontent.com/render/math?math=%5Csigma%28z%29%27%20%3D%20%281-%5Csigma%28z%29%29%5Csigma%28z%29&mode=display)

とかける。

```python
def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp( - x ) )
```

次に、コスト関数の勾配を計算する関数を定義する。 まず、活性化関数：

![$$ \vec{z}^{(i)} = \vec{w}^T\vec{x}^{(i)} + b $$](https://render.githubusercontent.com/render/math?math=%5Cvec%7Bz%7D%5E%7B%28i%29%7D%20%3D%20%5Cvec%7Bw%7D%5ET%5Cvec%7Bx%7D%5E%7B%28i%29%7D%20%2B%20b&mode=display)

![$$ a^{(i)} = \sigma(z^{(i)}) =  \sigma(\vec{w}^T\vec{x}^{(i)} + b) $$](https://render.githubusercontent.com/render/math?math=a%5E%7B%28i%29%7D%20%3D%20%5Csigma%28z%5E%7B%28i%29%7D%29%20%3D%20%20%5Csigma%28%5Cvec%7Bw%7D%5ET%5Cvec%7Bx%7D%5E%7B%28i%29%7D%20%2B%20b%29&mode=display)

を定義した時に、損失関数は以下の式で与えられ：

![$$ \mathcal{L}(a^{(i)},y^{(i)}) = -y^{(i)}\log{a^{(i)}}-(1-y^{(i)})\log{(1-a^{(i)})} $$](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BL%7D%28a%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29%20%3D%20-y%5E%7B%28i%29%7D%5Clog%7Ba%5E%7B%28i%29%7D%7D-%281-y%5E%7B%28i%29%7D%29%5Clog%7B%281-a%5E%7B%28i%29%7D%29%7D&mode=display)

コスト関数は損失関数の和：

![$$ J = \frac{1}{m}\Sigma_{i=1}^{m}\mathcal{L}(a^{(i)},y^{(i)})  $$](https://render.githubusercontent.com/render/math?math=J%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5CSigma_%7Bi%3D1%7D%5E%7Bm%7D%5Cmathcal%7BL%7D%28a%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29&mode=display)

として与えられる。



#### 2.1.Back Propagation

```python
def grad(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid( np.dot(w.T,X) + b )
    dw = 1.0 / m * np.dot(X,(A-Y).T)
    db = 1.0 / m * np.sum(A-Y)
#     print(- 1.0 / m * np.sum( Y * np.log(A) + ( 1 - Y ) * np.log( 1 - A ) ) )
    return dw,db
```

上式で用いられている勾配は、簡単な計算を行うことによって、

![$$ \frac{\partial J}{\partial \vec{w}} = \frac{1}{m}X(A-Y)^T $$](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Cvec%7Bw%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7DX%28A-Y%29%5ET&mode=display)

![$$ \frac{\partial J}{\partial b} = \frac{1}{m}\Sigma_{i=1}^{m}(A-Y) $$](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20b%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5CSigma_%7Bi%3D1%7D%5E%7Bm%7D%28A-Y%29&mode=display)

と与えられる。ここで、それぞれの変数の形を確認しておくと、

![$$ X.shape = (2,100) $$](https://render.githubusercontent.com/render/math?math=X.shape%20%3D%20%282%2C100%29&mode=display)

![$$ A.shape = (1,100) $$](https://render.githubusercontent.com/render/math?math=A.shape%20%3D%20%281%2C100%29&mode=display)

![$$ Y.shape = (1,100) $$](https://render.githubusercontent.com/render/math?math=Y.shape%20%3D%20%281%2C100%29&mode=display)

![$$ dw.shape = (2,100)\times(100,1) = (2,1) $$](https://render.githubusercontent.com/render/math?math=dw.shape%20%3D%20%282%2C100%29%5Ctimes%28100%2C1%29%20%3D%20%282%2C1%29&mode=display)

![$$ db.shape = np.sum(1,100) = (1,) $$](https://render.githubusercontent.com/render/math?math=db.shape%20%3D%20np.sum%281%2C100%29%20%3D%20%281%2C%29&mode=display)

となる。

#### 2.2.パラメータの更新

最後に、パラメーターである重み![$w_i$](https://render.githubusercontent.com/render/math?math=w_i&mode=inline)の更新を行う部分を実装する。勾配降下法によって、パラメーターを更新する：![$$ w_{i}^{(j+1)} = w_{i}^{(j)} - \eta \frac{\partial J}{\partial {w_{i}}} $$](https://render.githubusercontent.com/render/math?math=w_%7Bi%7D%5E%7B%28j%2B1%29%7D%20%3D%20w_%7Bi%7D%5E%7B%28j%29%7D%20-%20%5Ceta%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%7Bw_%7Bi%7D%7D%7D&mode=display)![$$ b^{(j+1)} = b^{(j)} - \eta \frac{\partial J}{\partial {b}} $$](https://render.githubusercontent.com/render/math?math=b%5E%7B%28j%2B1%29%7D%20%3D%20b%5E%7B%28j%29%7D%20-%20%5Ceta%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%7Bb%7D%7D&mode=display)

In [8]:

```python
def optimize(w,b,X,Y,learning_rate=0.1,iterate=1000):
    m = X.shape[1]
    for i in range(iterate):
        dw, db = grad(w,b,X,Y)
        w = w - dw * learning_rate
        b = b - db * learning_rate
    return w,b
```



### 3.メイン関数

```python
dim = data_x.shape[0]
w,b = initialize(dim)
w,b = optimize(w,b,data_x,data_y)
```

学習後のパラメータはこんな感じ

```python
print(w,b)
[[ 0.87850286]
 [-1.12473527]] 4.046088232542634
```

以下はグラフ絵画用のコード

```python
predict_y = np.zeros(data_x.shape[1])

for i in range(data_x.shape[1]):
    if w[0] * data_x[0][i] + w[1] * data_x[1][i] + b >= 0.0 :
        predict_y[i] = 1.0 
    else:
        predict_y[i] = 0.0
```

```python
df['predict'] = predict_y
df.head()
```



|      | x1       | x2       | target | predict |
| :--- | :------- | :------- | :----- | :------ |
| 0    | 7.824934 | 5.797468 | 1.0    | 1.0     |
| 1    | 1.229090 | 6.280158 | 0.0    | 0.0     |
| 2    | 6.371579 | 5.383301 | 1.0    | 1.0     |
| 3    | 5.896936 | 9.295513 | 0.0    | 0.0     |
| 4    | 4.066826 | 9.072754 | 0.0    | 0.0     |



```python
graph_x = np.array([0.1 * i for i in range(100)]) 
pred = - w[0]/w[1] * graph_x - b/w[1]
fig = plt.figure()
plt.scatter(data_x[0,:], data_x[1,:], marker='o', c=data_y,cmap=plt.cm.get_cmap('bwr'))
plt.plot(graph_x,pred,c="red")
plt.show()
```

![img2](https://user-images.githubusercontent.com/54795218/100469858-6cd8f280-311a-11eb-9d39-7628aa96bcb5.png)

線形分離できていることがわかる。

