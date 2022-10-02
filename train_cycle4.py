# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from single_network import single_network
from pv_descent import *
from train_network4 import train_network
from evaluate_network2 import *
import multiprocessing as mp

if __name__ == '__main__':

    mp.set_start_method('spawn')

    # デュアルネットワークの作成
    single_network()

    for i in range(25):
        print('Train',i,'====================')
        # セルフプレイ部
        self_play()

        # パラメータ更新部
        train_network()

        # 新パラメータ評価部
        #evaluate_network()
        evaluate_problem()
        update_best_player()
