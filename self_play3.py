# ====================
# セルフプレイ部
# ====================

# パッケージのインポート
from game import *
from pv_ubfm import pv_ubfm_scores
from dual_network import DN_OUTPUT_SIZE
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import os
import torch
from single_network import *

# パラメータの準備
SP_GAME_COUNT = 500 # セルフプレイを行うゲーム数（本家は25000）
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

def load_data():
    with open('./base/base.pos',mode='rb') as f:
        return pickle.load(f)

# 先手プレイヤーの価値
def first_player_value(ended_state):
    # 1:先手勝利, -1:先手敗北, 0:引き分け
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# 学習データの保存
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True) # フォルダがない時は生成
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history3'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# セルフプレイ
def self_play():
    # 学習データ
    history = []

    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model = model.to(device)
    model.eval()

    pos_list = load_data()
    # 学習データ
    history = []

    # 状態の生成
    for i, pos in enumerate(pos_list):
        print('\rSelfPlay {}'.format(i+1), end='')
        #print("-----------------------------")
        state = State(pos[0], pos[1])
        if state.is_lose():
            #print("found lose")
            values = -1.0
        elif state.is_draw():
            #print("found draw")
            values = 0.0
        else:
            # 合法手の確率分布の取得
            scores, values = pv_ubfm_scores(model, state, SP_TEMPERATURE)
        # 学習データに状態と方策を追加
        #print(state)
        #print(values)
        history.append([[state.pieces, state.enemy_pieces], values])

    # 学習データの保存
    write_data(history)

 

# 動作確認
if __name__ == '__main__':
    self_play()
