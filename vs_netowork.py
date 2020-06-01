# ====================
# 新パラメータ評価部
# ====================

# パッケージのインポート
from game import State
from pv_mcts import pv_mcts_action
from pathlib import Path
from shutil import copy
import numpy as np
from dual_network import *

# パラメータの準備
EN_GAME_COUNT = 1000 # 1評価あたりのゲーム数（本家は400）
EN_TEMPERATURE = 1.0 # ボルツマン分布の温度

# 先手プレイヤーのポイント
def first_player_point(ended_state):
    # 1:先手勝利, 0:先手敗北, 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# 1ゲームの実行
def play(next_actions):
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break;

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)



# ネットワークの評価
def evaluate_network():
    # 最新プレイヤーのモデルの読み込み
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model0 = DualNet()
    #model0.load_state_dict(torch.load('./learned_param/elmo2_best.h5'))
    model0.load_state_dict(torch.load('./learned_param/normal_50/best.h5'))
    model0 = model0.double()
    model0 = model0.to(device)
    model0.eval()
    

    # ベストプレイヤーのモデルの読み込み
    model1 = DualNet()
    #model1.load_state_dict(torch.load('./learned_param/normal_best.h5'))
    model1.load_state_dict(torch.load('./learned_param/normal_25_tadashii/best.h5'))
    #model0.load_state_dict(torch.load('./learned_param/elmo2_best.h5'))
    model1 = model1.double()
    model1 = model1.to(device)
    model1.eval()

    
    # PV MCTSで行動選択を行う関数の生成
    next_action0 = pv_mcts_action(model0, EN_TEMPERATURE)
    next_action1 = pv_mcts_action(model1, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)

    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EN_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 出力
        print('\rEvaluate {}/{} {} {}'.format(i + 1, EN_GAME_COUNT, total_point, total_point/(i+1)), end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)
    print('Point', total_point)



# 動作確認
if __name__ == '__main__':
    evaluate_network()
