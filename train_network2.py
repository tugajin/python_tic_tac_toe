# ====================
# パラメータ更新部
# ====================

# パッケージのインポート
from single_network import *
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random

# パラメータの準備
RN_EPOCHS = 30 # 学習回数
RN_BATCH_SIZE = 16 # バッチサイズ

# 学習データの読み込み
def load_data():
    history_path = sorted(Path('./data').glob('*.history2'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)
# 学習データの読み込み
def load_all_data():
    history_path = sorted(Path('./data').glob('*.history2'))
    ret = []
    for path in history_path:
        with path.open(mode='rb') as f:
            tmp = pickle.load(f)
            ret.extend(tmp)
    return ret
# ネットワークの学習
def train_network():
    # 学習データの読み込み
    history = load_data()
    xs, y_values, y_deep_values = zip(*history)

    # 学習のための入力データのシェイプの変換
    file, rank, channel = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), channel, file, rank)
    y_values = np.array(y_values)
    y_deep_values = np.array(y_deep_values)
    
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model = model.to(device)
    
    model.train()
    optimizer = optim.SGD(model.parameters(),lr=0.001, weight_decay=0.00001)
    
    print("len:" + str(len(xs)))
    
    indexs = [i for i in range(0,len(xs))]
    #indexs = [i for i in range(0,10)]
    
    for i in range(0,RN_EPOCHS):
        print("epoch:" + str(i),end="")
        random.shuffle(indexs)
        x = []
        yp = []
        yv = []
        yv2 = []
        sum_loss = 0.0
        sum_num = 0
        for j in indexs:

            x.append(xs[j])
            yv.append(y_values[j])
            yv2.append(y_deep_values[j])

            if len(x) == RN_BATCH_SIZE:
                x = torch.tensor(np.array(x),dtype=torch.float32)
                yv = torch.tensor(np.array(yv),dtype=torch.float32)
                yv2 = torch.tensor(np.array(yv2),dtype=torch.float32)

                x = x.to(device)
                yv = yv.to(device)
                yv2 = yv2.to(device)
                
                optimizer.zero_grad()
                outputs = model(x)
                outputs = torch.squeeze(outputs)
                loss_values = (0.0 * torch.sum((outputs - yv) ** 2)) + (1.0 * torch.sum((outputs - yv2) ** 2))
                loss = loss_values 

                loss.backward()
                optimizer.step()
                #print("loss" + str(loss.item()))
                x = []
                yv = []
                yv2 = []
                sum_loss += loss.item()
                sum_num += 1
        print(" avg loss " + str(sum_loss / sum_num))

    # 最新プレイヤーのモデルの保存
    torch.save(model.state_dict(), './model/latest_single.h5')

def check_train_data():
    # 学習データの読み込み
    history = load_all_data()
    print(type(history))
    for hist in history:
        x, y, y_deep = zip(hist)
        state = State(x[0][0],x[0][1])
        print(state)
        print(y)
        print(y_deep)
        print("------------------------")
    #print(type(y_deep_values))
    #print(y_deep_values)
    #print(type(y_values))
    #print(y_values)

# 動作確認
if __name__ == '__main__':
    #train_network()
    check_train_data()