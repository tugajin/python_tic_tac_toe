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
# パラメータの準備
DN_FILTERS  = 256 # 畳み込み層のカーネル数（本家は256）
DN_RESIDUAL_NUM =  16 # 残差ブロックの数（本家は19）
DN_INPUT_SHAPE = (3, 3, 2) # 入力シェイプ
DN_OUTPUT_SIZE = 9 # 配置先(3*3)

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
        h3 = F.relu(self.batch3(self.conv3(h1)) + h1)
        
        h4 = F.relu(self.batch4(self.conv4(h3)))
        h5 = F.relu(self.batch5(self.conv5(h4)) + h3)
        
        h6 = F.relu(self.batch6(self.conv6(h5)))
        h7 = F.relu(self.batch7(self.conv7(h6)) + h5)

        h8 = F.relu(self.batch8(self.conv8(h7)))
        h9 = F.relu(self.batch9(self.conv9(h8)) + h7)

        
        #policy
        h_p1 = F.relu(self.batch_p1(self.conv_p1(h9)))

        h_p1 = h_p1.reshape(h_p1.shape[0],18)
        
        policy = self.fc_p2(h_p1)
        
        #value
        
        h_v1 = F.relu(self.batch_v1(self.conv_v1(h9)))

        h_v1 = h_v1.reshape(h_v1.shape[0],9)
        
        h_v2 = F.relu(self.fc_v2(h_v1))
       
        value = torch.tanh(self.fc_v3(h_v2))
        
        return policy,value
        
# デュアルネットワークの作成
def dual_network():
    # モデル作成済みの場合は無処理
    if os.path.exists('./model/best.h5'):
        return
    
    model = DualNet()

    # モデルの保存    
    torch.save(model.state_dict(), './model/best.h5')# ベストプレイヤーのモデル

# 動作確認
if __name__ == '__main__':
    dual_network()
