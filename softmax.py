import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# MNISTデータの読み込み
mnist = fetch_mldata('MNIST original', data_home='./data/')
X, Y = mnist.data, mnist.target
X = X / 255.
Y = Y.astype("int")

# データセットの可視化
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i * 6500].reshape(28, 28), cmap='gray_r')
    plt.axis("off")
plt.show()

# データセットの分割
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=2)
train_y = np.eye(10)[train_y].astype(np.int32)
test_y = np.eye(10)[test_y].astype(np.int32)
train_n = train_x.shape[0]
test_n = test_x.shape[0]

# softmax関数
def softmax(x):
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return y

# 多クラスの交差エントロピー誤差の実装
def cross_entropy(y, t):
    L = -np.mean(np.sum(t*np.log(y), axis=1))
    return L

# ロジスティック回帰
class LogisticRegression:
    def __init__(self, n_in, n_out):
        self.W = np.random.uniform(0.08, -0.08, (n_in, n_out)) #勾配の初期化
        self.b = np.zeros(n_out) #バイアスの初期化

    def gradient_decent(self, X, Y, T, eps):
        batchsize = X.shape[0]
        delta = Y - T
        self.W = self.W - eps * np.dot(X.T, delta) / batchsize
        self.b = self.b - eps * np.sum(delta, axis=0) / batchsize
        

    def train(self, x, t, lr):
        y = softmax(np.dot(x, self.W) + self.b) #予測
        self.gradient_decent(x, y, t, lr) #パラメータの更新
        loss = cross_entropy(y, t) #ロスの算出
        return y, loss
    
    def test(self, x, t):
        y = softmax(np.dot(x, self.W) + self.b) #予測
        loss = cross_entropy(y, t) #ロスの算出
        return y, loss

# モデルの初期化
model = LogisticRegression(784, 10)
# ハイパーパラメタの設定
n_epoch = 20
batchsize = 100
lr = 1
# 学習
for epoch in range(n_epoch):
    print ('epoch %d |　' % epoch, end="")
    
    # Training
    sum_loss = 0
    pred_label = []
    perm = np.random.permutation(train_n) #ランダムに並び替える

    for i in range(0, train_n, batchsize): #ミニバッチごとに学習を行う
        x = train_x[perm[i:i+batchsize]]
        y = train_y[perm[i:i+batchsize]]
        
        pred, loss = model.train(x, y, lr)
        sum_loss += loss * x.shape[0]
        # pred には， (N, 10)の形で，画像が0~9の各数字のどれに分類されるか
        # の事後確率が入っている
        # そこで，最も大きい値をもつインデックスを取得することで，識別結果
        # を得ることができる
        pred_label.extend(pred.argmax(axis=1))

    loss = sum_loss / train_n
    # 正解率
    accu = accuracy_score(pred_label, np.argmax(train_y[perm], axis=1))
    print('Train loss %.3f, accuracy %.4f |　' %(loss, accu), end="")
    

    # Testing
    sum_loss = 0
    pred_label = []
    
    for i in range(0, test_n, batchsize):
        x = test_x[i: i+batchsize]
        y = test_y[i: i+batchsize]
        
        pred, loss = model.test(x, y)
        sum_loss += loss * x.shape[0]
        pred_label.extend(pred.argmax(axis=1))
        
    loss = sum_loss / test_n
        
    accu = accuracy_score(pred_label, np.argmax(test_y, axis=1))
    print('Test loss %.3f, accuracy %.4f' %(loss, accu) )
