# ====================
# 新パラメータ評価部
# ====================

# パッケージのインポート
from game import State
from pv_mcts import *
from pathlib import Path
from shutil import copy
import numpy as np
from dual_network import *

# パラメータの準備
EN_GAME_COUNT = 10 # 1評価あたりのゲーム数（本家は400）
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

# ベストプレイヤーの交代
def update_best_player():
    copy('./model/latest.h5', './model/best.h5')
    print('Change BestPlayer')

# ネットワークの評価
def evaluate_network():
    # 最新プレイヤーのモデルの読み込み
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model0 = DualNet()
    model0.load_state_dict(torch.load('./model/latest.h5'))
    model0 = model0.double()
    model0 = model0.to(device)
    model0.eval()
    

    # ベストプレイヤーのモデルの読み込み
    model1 = DualNet()
    model1.load_state_dict(torch.load('./model/best.h5'))
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
        print('\rEvaluate {}/{}'.format(i + 1, EN_GAME_COUNT), end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)


    # ベストプレイヤーの交代
    if average_point > 0.5:
        update_best_player()
        evaluate_problem()
        return True
    else:
        return False

def evaluate_problem():
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cpu')
    model = DualNet()
    model.load_state_dict(torch.load('./model/best.h5'))
    model = model.double()
    model = model.to(device)
    model.eval()

    # 状態の生成
    state = State()
    print(state)
    score, values = pv_mcts_scores(model, state, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")
    state = state.next(2)
    print(state)
    moves = state.legal_actions()
    score, values = pv_mcts_scores(model, state, EN_TEMPERATURE) 
    for i in range(len(moves)):
        print(str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")

    state = state.next(1)
    print(state)
    score, values = pv_mcts_scores(model, state, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")

    state = state.next(4)
    print(state)
    score, values = pv_mcts_scores(model, state, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")

    state = state.next(6)
    print(state)
    score, values = pv_mcts_scores(model, state, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")

    state = State()
    state = state.next(2) 
    state = state.next(0) 
    state = state.next(4) 
    state = state.next(1) 
    print(state)
    score, values = pv_mcts_scores(model, state, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")

    state = State()
    state = state.next(2) 
    state = state.next(0) 
    state = state.next(5) 
    state = state.next(1) 
    print(state)
    score, values = pv_mcts_scores(model, state, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")

    state = State()
    state = state.next(0) 
    state = state.next(6) 
    state = state.next(1) 
    state = state.next(7) 
    print(state)
    score, values = pv_mcts_scores(model, state, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(str(moves[i]) + ":" + str(score[i]))
    print(values)

# 動作確認
if __name__ == '__main__':
    evaluate_network()
    evaluate_problem()
