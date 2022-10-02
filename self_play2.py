# ====================
# セルフプレイ部
# ====================

# パッケージのインポート
from game import *
from pv_ubfm import pv_ubfm_scores
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
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history2'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# 1ゲームの実行
def play(model, device):
    # 学習データ
    history = []

    # 状態の生成
    state = State()
    i = 0
    while True:
        append_pos_dict(state.hash_key())
        # ゲーム終了時
        if state.is_done():
            break

        # 合法手の確率分布の取得

        scores, values = pv_ubfm_scores(model, state, device, SP_TEMPERATURE)

        # 学習データに状態と方策を追加
        history.append([[state.pieces, state.enemy_pieces], None, values])
        # state2 = state
        # for i in range(3):
        #     state2 = state2.rotate45()
        #     state3 = state2.mirror()
        #     history.append([[state2.pieces, state2.enemy_pieces], None, values])
        #     history.append([[state3.pieces, state3.enemy_pieces], None, values])
        if random.random() < 0.4:
            action = np.random.choice(state.legal_actions())
        else:
            # 行動の取得
            action = state.legal_actions()[np.argmax(scores)]
            #print("--------------------------------")
            #print(state)
            #print("values:",values)
            #print("scores:",scores)
            #print("action:",action)

        # 次の状態の取得
        state = state.next(action)
        i += 1
    # 学習データに価値を追加
    value = first_player_value(state)
    #print(state)
    #print(value)
    for i in range(len(history)):
        history[i][1] = value
        value = -value
    return history

# セルフプレイ
def self_play():
    # 学習データ
    history = []

    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model = model.to(device)
    model.eval()

    # 複数回のゲームの実行
    for i in range(SP_GAME_COUNT):
        # 1ゲームの実行
        h = play(model, device)
        history.extend(h)

        # 出力
        print('\rSelfPlay {}/{} :pos {}'.format(i+1, SP_GAME_COUNT,len(pos_dict)), end='')
    print(f'all_pos:{len(pos_dict)} percent:{int((len(pos_dict) / ALL_POS_LEN) * 100)}')

    # 学習データの保存
    write_data(history)

 

# 動作確認
if __name__ == '__main__':
    self_play()
