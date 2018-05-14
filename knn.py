import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import fetch_mldata
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# MNISTデータの読み込み
mnist = fetch_mldata('MNIST original', data_home='./data/')
X, Y = mnist.data, mnist.target
X = np.array(X/255.0, dtype=np.float32)
Y = np.array(Y, dtype=np.uint8)
class_num = 10 #class数
print("X.shape: " + str(X.shape), ", Y.shape: " + str(Y.shape))

# データセットの可視化
for i in range(class_num):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i * 6500].reshape(28, 28), cmap='gray_r')
    plt.axis("off")
plt.show()

# データセットの削減
np.random.seed(100)
random_sample = np.arange(len(X))
np.random.shuffle(random_sample)
X = X[random_sample[:3000]]
Y = Y[random_sample[:3000]]

#テスト用データを分ける
train_X, test_x, train_Y, test_y = train_test_split(X, Y, test_size=0.2, random_state=2)
#学習用データと検証用データを分ける
train_x, val_x, train_y, val_y = train_test_split(train_X, train_Y, test_size=0.1, random_state=42)
print('train data:',train_x.shape,', train label:',train_y.shape)
print('val data:   ',val_x.shape,',    val label:   ',val_y.shape)
print('test data: ',test_x.shape,',   test label: ',test_y.shape)

# 距離関数の実装
def cosine_distance(X, Y):
    Z = np.zeros((X.shape[0], Y.shape[0]))
    for n in range(X.shape[0]):
        for m in range(Y.shape[0]):
            Z[n][m] = 1 - np.dot(X[n],Y[m]) / np.linalg.norm(X[n]) / np.linalg.norm(Y[m])
    return Z

def euclidean_distance(X, Y):
    Z = np.zeros((X.shape[0], Y.shape[0]))
    for n in range(X.shape[0]):
        for m in range(Y.shape[0]):
            Z[n][m] = np.linalg.norm(X[n] - Y[m])
    return Z

class KNN:
    def __init__(self, x, y, func=cosine_distance):
        self.train_x = x
        self.train_y = y
        self.distance_func = func

    #入力パターンに対して予測ラベルを返す
    def prediction(self, X, k):
        #1. 全ての入力パターンと全ての学習データとの距離を計算する．
        distance_matrix = self.distance_func(X, self.train_x)
        #2.  距離のに学習パターンをソートする
        #distance_matrixを昇順にソートするインデックスを返す
        sort_index = np.argsort(distance_matrix, axis=1)
        #3.  ソートした学習パターンの上位k個を取り上げ
        #    最も出現回数の多いカテゴリを出力する
        nearest_k = sort_index[:,:k] #上位k個のインデックスを取り出す
        labels = self.train_y[nearest_k] #上位k個のラベルを取り出す
        #上位k個のラベルに各ラベルが何個ずつ含まれるか調べる
        label_num = np.sum(np.eye(class_num)[labels], axis=1)
        #上位k個のラベルで最も多いラベルを調べる
        Y = np.argmax(label_num, axis=1)
        return Y

    #予測データと正解データを用いてaccuracyを計算する
    def get_accuracy(self, pred, real, eval_func=accuracy_score):
        accuracy = eval_func(pred, real)
        return accuracy
    
    # 最適なkを見つけるためにkを変化させて予測を行い，最も性能が高いkを返す
    def find_k(self, val_x, val_y, k_list):
        score_list = []
        for k in k_list:
            pred = self.prediction(val_x, k)
            accuracy = self.get_accuracy(pred, val_y)
            print('k：{0}, accuracy：{1:.5f}'.format(k,accuracy))
            score_list.append(accuracy)
            
        top_ind = np.argmax(score_list)
        best_k = k_list[top_ind]
        print('best k : {0}, val score : {1:.5f}'.format(best_k,score_list[top_ind]))
        return best_k

# k近傍法の実行（コサイン距離）
#インスタンス生成
knn = KNN(train_x, train_y, func = cosine_distance)
#検証用データval_xを用いて最適なkを算出する
k_list = np.arange(1,21,2)
best_k = knn.find_k(val_x, val_y, k_list)
#検証用データで算出したkを用いてテストデータのクラスを予測する
pred_y = knn.prediction(test_x, best_k)
#正解率の計算
result = knn.get_accuracy(pred_y, test_y)
print('test_accuracy :　{0:.5f}'.format(result))

# k近傍法の実行（ユークリッド距離）
knn = KNN(train_x, train_y, func = euclidean_distance)
#検証用データval_xを用いて最適なkを算出する
k_list = np.arange(1,21,2)
best_k = knn.find_k(val_x, val_y, k_list)
#検証用データで算出したkを用いてテストデータのクラスを予測する
pred_y = knn.prediction(test_x, best_k)
#正解率の計算
result = knn.get_accuracy(pred_y, test_y)
print('test_accuracy :　{0:.5f}'.format(result))
