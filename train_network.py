# ====================
# パラメータ更新部
# ====================

# パッケージのインポート
from dual_network import *
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random

# パラメータの準備
RN_EPOCHS = 30 # 学習回数
RN_BATCH_SIZE = 128 # バッチサイズ

# 学習データの読み込み
def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

# デュアルネットワークの学習
def train_network():
    # 学習データの読み込み
    history = load_data()
    xs, y_policies, y_values, y_deep_values = zip(*history)

    # 学習のための入力データのシェイプの変換
    file, rank, channel = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), channel, file, rank)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)
    y_deep_values = np.array(y_deep_values)
    
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualNet()
    model.load_state_dict(torch.load('./model/best.h5'))
    model = model.double()
    model = model.to(device)
    
    model.train()
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    
    print("len:" + str(len(xs)))
    
    indexs = [i for i in range(0,len(xs))]
    
    criterion_policies = nn.CrossEntropyLoss()
    criterion_values = nn.MSELoss()
    criterion_values2 = nn.MSELoss()
    
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
            yp.append(y_policies[j])
            yv.append(y_values[j])
            yv2.append(y_deep_values[j])

            if len(x) == RN_BATCH_SIZE:
                x = torch.tensor(x,dtype=torch.double)
                yp = np.array(yp)
                yp = yp.argmax(axis = 1)
                yp = torch.tensor(yp,dtype=torch.long)
                yv = torch.tensor(yv,dtype=torch.double)
                yv2 = torch.tensor(yv2,dtype=torch.double)
                
                x = x.to(device)
                yp = yp.to(device)
                yv = yv.to(device)
                yv2 = yv2.to(device)
                
                optimizer.zero_grad()
                outputs = model(x)
                output_policy = outputs[0]
                output_value = torch.squeeze(outputs[1])


                loss_policies = criterion_policies(output_policy,yp)
                loss_values = criterion_values(output_value,yv)
                loss_values2 = criterion_values2(output_value,yv2) 
                loss = loss_policies + loss_values + loss_values2
                #loss = loss_policies + loss_values 

                loss.backward()
                optimizer.step()
                #print("loss" + str(loss.item()))
                x = []
                yp = []
                yv = []
                yv2 = []
                sum_loss += loss.item()
                sum_num += 1
        print(" avg loss " + str(sum_loss / sum_num))

    # 最新プレイヤーのモデルの保存
    torch.save(model.state_dict(), './model/latest.h5')

# 動作確認
if __name__ == '__main__':
    train_network()
