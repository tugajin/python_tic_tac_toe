# ====================
# Descentの作成
# ====================

# パッケージのインポート
from math import sqrt
from game import *
from pv_ubfm import pv_ubfm_scores
from datetime import datetime
from pathlib import Path
from single_network import *

import random
import operator
import numpy as np
import pickle
import os
import torch
import copy

# パラメータの準備
PV_EVALUATE_COUNT = 50 # 1推論あたりのシミュレーション回数（本家は1600）
# パラメータの準備
SP_GAME_COUNT = 50 # セルフプレイを行うゲーム数（本家は25000）
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# 推論
def predict(model, node_list, device):

    #print("predict")
    #print(state)
    # 推論のための入力データのシェイプの変換
    file, rank, channel = DN_INPUT_SHAPE
    x = np.array([[node.state.pieces, node.state.enemy_pieces] for node in node_list])

    #print(x)

    x = x.reshape(len(node_list), channel, file, rank)
    x = np.array(x)
    x = torch.tensor(x,dtype=torch.float32)
   
    x = x.to(device)
    
    with torch.no_grad():
        # 推論
        y = model(x)
    # 価値の取得
    for i in range(len(node_list)):
        value = y[i][0].item()
        # 丸め
        value = (int(value * 10000))/10000
        if value >= 1:
            value = 0.9000
        elif value <= -1:
            value = -0.9000
        node_list[i].w = value
        node_list[i].n += 1
        if node_list[i].state.is_done():
            # 勝敗結果で価値を取得
            # より手数が少ない局面で勝ったほうが良いとする
            node_list[i].w = -0.9999 + (node_list[i].ply / 100) if node_list[i].state.is_lose() else 0
            node_list[i].completion = -1 if node_list[i].state.is_lose() else 0
            #node_list[i].w = node_list[i].completion = -1 if node_list[i].state.is_lose() else 0
            node_list[i].resolved = True

# ノードのリストを試行回数のリストに変換
def nodes_to_scores(nodes):
    scores = [1] * len(nodes)
    max_node = max(nodes,key=lambda n : n.n)
    for i, c in enumerate(nodes):
        if c.resolved:
            if c.completion == -1:
                n = max_node.n
                scores = [0] * len(nodes)
                scores[i] = max_node.n
                break
            elif c.completion == 0:
                n = c.n
            elif c.completion == 1:
                n = 0.1
            else:
                assert(False)
            scores[i] = n
        else:
            scores[i] = c.n - c.w
    return scores

# Descent木探索のスコアの取得
def pv_descent_scores(model, state, device, history, temperature):

    # モンテカルロ木探索のノードの定義
    class Node:
        # ノードの初期化
        def __init__(self, state, ply, action = -1):
            self.state = state # 状態
            self.w = -999 # 価値
            self.n = 0 # 試行回数
            self.ply = ply
            self.child_nodes = None  # 子ノード群
            self.action = action
            self.best_action = -1
            self.resolved = False
            self.completion = -99 # draw:0 win:1 lose:-1 unknown: -99

        def dump(self, only_root = False):
            print("-----------------start-------------------------")
            print(self.state)
            print("w:",self.w)
            print("n:",self.n)
            print("ply:",self.ply)
            print("action:",self.action)
            print("best_action",self.best_action)
            print("has_child:",not self.child_nodes is None)
            print("resoloved:",self.resolved)
            print("completion:",self.completion)
            if self.child_nodes:
                if not only_root:
                    best = self.next_child_node()
                    if best is None:
                        print("best mate is null")
                    else:
                        print("best_child2:", best.action)
                    
                for c in self.child_nodes:
                    print("  action:",c.action)
                    print("  child_w:",c.w)
                    print("  child_n:",c.n)
                    print("  child_ply:",c.ply)
                    print("  resoloved:",c.resolved)
                    print("  completion:",c.completion)
                    print("  ---------------------")
                    
            print("-----------------end-------------------------")
        # 局面の価値の計算
        def evaluate(self):
            assert(not self.resolved)
            assert(not self.state.is_done())
            # # ゲーム終了時
            # if self.state.is_done():
            #     # 勝敗結果で価値を取得
            #     self.w = self.completion = -1 if self.state.is_lose() else 0
            #     # 試行回数の更新
            #     self.n += 1
            #     self.resolved = True
            #     assert(False)
            #     return 

            # 子ノードが存在しない時
            if not self.child_nodes:
                # 子ノードの展開
                self.child_nodes = []
                for action in self.state.legal_actions():
                    self.child_nodes.append(Node(self.state.next(action),self.ply+1,action))
                # ニューラルネットワークの推論で価値を取得
                predict(model, self.child_nodes, device)
                # 価値と試行回数の更新
                self.update_node()
            if not self.resolved:
                # 評価値が最大の子ノードを取得
                next_node = self.next_child_node()
                assert(next_node is not None)
                next_node.evaluate()
                # 価値と試行回数の更新
                self.update_node()

        def next_child_node_debug(self):
            max_index = -1
            max_value = -9999
            min_num = self.n
            lose_num = 0
            draw_num = 0
            child_nodes_len = len(self.child_nodes)
            assert(child_nodes_len != 0)
            for i, child_node in enumerate(self.child_nodes):
                if child_node.resolved:
                    # 子供に負けを見つけた→つまり勝ちなので終わり
                    if child_node.completion == -1:
                        max_index = i
                        self.resolved = True
                        self.completion = 1
                        self.w = 1
                        self.dump(True)
                        assert(False)
                        return None
                    elif child_node.completion == 1:
                        lose_num += 1
                    else:
                        assert(child_node.completion == 0)
                        draw_num += 1
                else:
                    if -child_node.w == max_value:
                        if child_node.n < min_num:
                            max_index = i
                            max_value = -child_node.w
                            min_num = child_node.n
                    elif -child_node.w > max_value:
                        max_index = i
                        max_value = -child_node.w
                        min_num = child_node.n
            #子供が全部引き分け
            if child_nodes_len == draw_num:
                self.resolved = True
                self.completion = 0
                self.w = 0
                self.dump(True)
                assert(False)
                return None
            # 子供が全部勝ち→この局面は負け
            elif child_nodes_len == lose_num:
                self.resolved = True
                self.completion = -1
                self.w = -1
                self.dump(True)
                assert(False)
                return None
            # 子供に引き分けと負けが付与済→引き分け
            elif child_nodes_len == (draw_num + lose_num):
                assert(draw_num != 0)
                self.resolved = True
                self.completion = 0
                self.w = 0
                self.dump(True)
                assert(False)
                return None
            assert(not self.child_nodes[max_index].resolved)
            return self.child_nodes[max_index]
        
        # 評価値が最大の子ノードを取得
        def next_child_node(self):
            not_resolved = [child for child in self.child_nodes if not child.resolved]
            assert(len(not_resolved) != 0)
            max_child = max(not_resolved,key=lambda x: (-x.w, -x.n) )
            # debug_max_child = self.next_child_node_debug()
            # if max_child.action != debug_max_child.action:
            #     self.dump(True)
            #     print(max_child.action)
            #     print(debug_max_child.action)
            #     assert(False)
            return max_child

        # 現在のnodeの情報を更新
        def update_node(self):
            max_index = -1
            max_value = -9999
            max_num = -1
            lose_num = 0
            draw_num = 0
            child_nodes_len = len(self.child_nodes)
            assert(child_nodes_len != 0)
            for i, child_node in enumerate(self.child_nodes):
                if child_node.resolved:
                    # 子供に負けを見つけた→つまり勝ちなので終わり
                    if child_node.completion == -1:
                        max_index = i
                        self.resolved = True
                        self.completion = 1
                        self.w = -child_node.w
                        self.best_action = child_node.action
                        return 
                    elif child_node.completion == 1:
                        lose_num += 1
                        #負けの局面は選ばない
                        #continue
                    else:
                        assert(child_node.completion == 0)
                        draw_num += 1
                if -child_node.w == max_value:
                    if child_node.n > max_num:
                        max_index = i
                        max_value = -child_node.w
                        max_num = child_node.n
                elif -child_node.w > max_value:
                    max_index = i
                    max_value = -child_node.w
                    max_num = child_node.n
            self.n += 1
            #子供が全部引き分け
            if child_nodes_len == draw_num:
                self.resolved = True
                self.w = self.completion = 0
                return 
            # 子供が全部勝ち→この局面は負け
            elif child_nodes_len == lose_num:
                self.resolved = True
                self.completion = -1
                self.w = -self.child_nodes[max_index].w
                self.best_action = self.child_nodes[max_index].action
                return 
            # 子供に引き分けと負けが付与済→引き分け
            elif child_nodes_len == (draw_num + lose_num):
                assert(draw_num != 0)
                self.resolved = True
                self.w = self.completion = 0
                return 
            self.w = -self.child_nodes[max_index].w
            self.best_action = self.child_nodes[max_index].action

    # 現在の局面のノードの作成
    root_node = Node(state, 0)

    # 複数回の評価の実行
    #print("start simulation")
    for i in range(PV_EVALUATE_COUNT):
    #    print(f"try:{i}")
        if root_node.resolved:
            break
        root_node.evaluate()
    #    root_node.dump()
    #    print(f"result:{result}")
    #print("end simulation")

    # 合法手の確率分布
    scores = nodes_to_scores(root_node.child_nodes)

    # 探索木の情報をhistoryに登録
    add_history(history, root_node)

    n = root_node
    #n.dump(True)

    #while True:
    while False:
        n.dump()
        if not n.child_nodes:
            break
        best_child = n.next_child_node()
        if best_child is None:
            break
        n = best_child

    if temperature == 0: # 最大値のみ1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else: # ボルツマン分布でバラつき付加
        scores = boltzman(scores, temperature)
    return scores, root_node.w

# Descent木探索で行動選択
def pv_descent_action(model, device, history, temperature=0):
    def pv_descent_action(state):
        scores,values = pv_descent_scores(model, state, device, history, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_descent_action

# ボルツマン分布
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

# 探索木の情報を保存
def add_history(history, node):
    state = copy.deepcopy(node.state)
    history.append([[state.pieces, state.enemy_pieces], node.w, node.completion, node.resolved])
    # このノードの情報を格納
    if node.child_nodes is not None:
        for child in node.child_nodes:
            add_history(history,child)

# 学習データの保存
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True) # フォルダがない時は生成
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history4'.format(
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
        # ゲーム終了時
        if state.is_done():
            break

        # 合法手の確率分布の取得
        scores, values = pv_descent_scores(model, state, device, history, SP_TEMPERATURE)
        # 学習データに状態と方策を追加
        if random.random() < 0.4:
            action = np.random.choice(state.legal_actions())
        else:
            # 行動の取得
            action = state.legal_actions()[np.argmax(scores)]
        # 次の状態の取得
        state = state.next(action)
        i += 1
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
        print('\rSelfPlay {}/{}'.format(i+1, SP_GAME_COUNT), end='')

    print(f"history_len:{len(history)}")
    # 学習データの保存
    write_data(history)

# 動作確認
if __name__ == '__main__':
    self_play()