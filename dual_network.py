# ====================
# デュアルネットワークの作成
# ====================

# パッケージのインポート
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from game import *
# パラメータの準備
DN_FILTERS  = 256 # 畳み込み層のカーネル数（本家は256）
DN_RESIDUAL_NUM =  16 # 残差ブロックの数（本家は19）
DN_INPUT_SHAPE = (3, 3, 2) # 入力シェイプ
DN_OUTPUT_SIZE = 9 # 配置先(3*3)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.l1 = nn.Linear(channel,channel//reduction)
        self.l2 = nn.Linear(channel//reduction,channel)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, (1, 1))
        y = y.view(x.shape[0],x.shape[1])
        y = F.relu(self.l1(y))
        y = torch.sigmoid(self.l2(y))
        y = y.view(x.shape[0],x.shape[1],1,1)
        x = x * y
        return x 
    def out(self, x):
        y = F.adaptive_avg_pool2d(x, (1, 1))
        y = y.view(x.shape[0],x.shape[1])
        y = F.relu(self.l1(y))
        y = torch.sigmoid(self.l2(y))
        y = y.view(x.shape[0],x.shape[1],1,1)
        print(y)

class DualNet(nn.Module):
    def __init__(self):
        super(DualNet,self).__init__()
        self.conv1 = nn.Conv2d(2,DN_FILTERS,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(DN_FILTERS,DN_FILTERS,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(DN_FILTERS,DN_FILTERS,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(DN_FILTERS,DN_FILTERS,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(DN_FILTERS,DN_FILTERS,kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(DN_FILTERS,DN_FILTERS,kernel_size=3,padding=1)
        self.conv7 = nn.Conv2d(DN_FILTERS,DN_FILTERS,kernel_size=3,padding=1)
        self.conv8 = nn.Conv2d(DN_FILTERS,DN_FILTERS,kernel_size=3,padding=1)
        self.conv9 = nn.Conv2d(DN_FILTERS,DN_FILTERS,kernel_size=3,padding=1)
        
        self.batch1 = nn.BatchNorm2d(DN_FILTERS)
        self.batch2 = nn.BatchNorm2d(DN_FILTERS)
        self.batch3 = nn.BatchNorm2d(DN_FILTERS)
        self.batch4 = nn.BatchNorm2d(DN_FILTERS)
        self.batch5 = nn.BatchNorm2d(DN_FILTERS)
        self.batch6 = nn.BatchNorm2d(DN_FILTERS)
        self.batch7 = nn.BatchNorm2d(DN_FILTERS)
        self.batch8 = nn.BatchNorm2d(DN_FILTERS)
        self.batch9 = nn.BatchNorm2d(DN_FILTERS)

        self.se3 = SELayer(DN_FILTERS)
        self.se5 = SELayer(DN_FILTERS)
        self.se7 = SELayer(DN_FILTERS)
        self.se9 = SELayer(DN_FILTERS)
        
        self.conv_p1 = nn.Conv2d(DN_FILTERS,2,kernel_size=1)
        self.batch_p1 = nn.BatchNorm2d(2)
        self.fc_p2 = nn.Linear(18,DN_OUTPUT_SIZE)
        
        self.conv_v1 = nn.Conv2d(DN_FILTERS,1,kernel_size=1)
        self.batch_v1 = nn.BatchNorm2d(1)
        self.fc_v2 = nn.Linear(9,DN_FILTERS)
        self.fc_v3 = nn.Linear(DN_FILTERS,1)
    
        
    def forward(self,x):

        h1 = F.relu(self.batch1(self.conv1(x)))
        
        h2 = F.relu(self.batch2(self.conv2(h1)))
        h3 = F.relu(self.se3(self.batch3(self.conv3(h1))) + h1)
        
        h4 = F.relu(self.batch4(self.conv4(h3)))
        h5 = F.relu(self.se5(self.batch5(self.conv5(h4))) + h3)
        
        h6 = F.relu(self.batch6(self.conv6(h5)))
        h7 = F.relu(self.se7(self.batch7(self.conv7(h6))) + h5)

        h8 = F.relu(self.batch8(self.conv8(h7)))
        h9 = F.relu(self.se9(self.batch9(self.conv9(h8))) + h7)

        
        #policy
        h_p1 = F.relu(self.batch_p1(self.conv_p1(h9)))

        h_p1 = h_p1.reshape(h_p1.shape[0],18)

        policy = self.fc_p2(h_p1)
        m = nn.Softmax(dim=1)
        policy = m(policy)
        
        #value
        
        h_v1 = F.relu(self.batch_v1(self.conv_v1(h9)))

        h_v1 = h_v1.reshape(h_v1.shape[0],9)
        
        h_v2 = F.relu(self.fc_v2(h_v1))
       
        value = torch.tanh(self.fc_v3(h_v2))
        
        return policy,value

    def out_se(self,x):

        print(x)
        h1 = F.relu(self.batch1(self.conv1(x)))
        
        h2 = F.relu(self.batch2(self.conv2(h1)))

        print("se3")
        self.se3.out(self.batch3(self.conv3(h1)))

        h3 = F.relu(self.se3(self.batch3(self.conv3(h1))) + h1)
        
        h4 = F.relu(self.batch4(self.conv4(h3)))
        h5 = F.relu(self.se5(self.batch5(self.conv5(h4))) + h3)
        
        print("se5")
        self.se5.out(self.batch5(self.conv5(h4)))

        h6 = F.relu(self.batch6(self.conv6(h5)))
        h7 = F.relu(self.se7(self.batch7(self.conv7(h6))) + h5)

        print("se7")
        self.se7.out(self.batch7(self.conv7(h6)))

        h8 = F.relu(self.batch8(self.conv8(h7)))
        h9 = F.relu(self.se9(self.batch9(self.conv9(h8))) + h7)

        print("se9")
        self.se9.out(self.batch9(self.conv9(h8)))

        
# デュアルネットワークの作成
def dual_network():
    # モデル作成済みの場合は無処理
    if os.path.exists('./model/best.h5'):
        return
    
    model = DualNet()


    # モデルの保存    
    torch.save(model.state_dict(), './model/best.h5')# ベストプレイヤーのモデル

def print_network():
    model = DualNet()
    model.load_state_dict(torch.load('./model/best.h5'))
    model = model.double()
    state = State()

    # 推論のための入力データのシェイプの変換
    file, rank, channel = DN_INPUT_SHAPE
    x = np.array([state.pieces, state.enemy_pieces])
    x = x.reshape(channel, file, rank)
    x = np.array([x])
    x = torch.tensor(x,dtype=torch.double)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = x.to(device)
    
    with torch.no_grad():
        # 推論
        model.out_se(x)


# 動作確認
if __name__ == '__main__':
    #dual_network()
    print_network()
