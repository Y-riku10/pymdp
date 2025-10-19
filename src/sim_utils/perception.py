# num_obs, 
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import vonmises
import seaborn as sns
from pymdp import utils
import jax.tree_util as jtu
from jax import jit
import jax.random as jr
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial
from pymdp.jax.agent import Agent as AIFAgent
from jax.tree_util import tree_map
from equinox import tree_at



import pinocchio as pin

# 観測のためのrobotArm環境を定義
class RobotArmEnv:
    def __init__(
            self,
            num_obs,
            time_steps,
            robot,
    ):
        self.num_obs = num_obs
        self.time_steps = time_steps
        self.robot = robot
        self.data = robot.createData()
        self.qs = None
        self.dqs = None
        self.ddqs = None
        self.obs_cont = None
        self.obs_onehot = None
        self.obs_idx = None


    def computeAllobs(self, qs, dt):
        self.qs = qs
        # 角度の軌跡から角速度と角加速度を計算
        dqs = np.diff(qs, n=1, axis=0)/dt
        self.dqs = dqs
        ddqs = np.diff(dqs, n=1, axis=0)/dt
        self.ddqs = ddqs

        # 用いるデータのインデックスを定義
        idx_qs = np.linspace(0, len(qs)-1, self.time_steps).astype(int)
        idx_dqs = np.linspace(0, len(dqs)-1, self.time_steps).astype(int)
        idx_ddqs = np.linspace(0, len(ddqs)-1, self.time_steps).astype(int)

        # qs, dqs, ddqs をそれぞれ間引き
        qs_sampled = qs[idx_qs]
        dqs_sampled = dqs[idx_dqs]
        ddqs_sampled = ddqs[idx_ddqs]

        # それらを横方向に結合（列方向に結合）
        self.obs_cont = np.concatenate([qs_sampled, dqs_sampled, ddqs_sampled], axis=1)

        # ワンホットエンコード
        # クリッピング範囲を定義
        qs_min, qs_max = self.robot.lowerPositionLimit, self.robot.upperPositionLimit
        velocity_limit = self.robot.velocityLimit
        dqs_min, dqs_max = -velocity_limit, velocity_limit
        effort_limit = self.robot.effortLimit
        ddqs_min, ddqs_max = -effort_limit, effort_limit

        # qs_onehot_list = []
        # dqs_onehot_list = []
        # ddqs_onehot_list = []

        # obs_onehot_list = np.empty(len(self.num_obs), dtype=object)

        num_obs = self.num_obs  # ex. [10,10,8,8,6,6] みたいなやつ
        nq = self.robot.nq
        num_total = nq * 3  # q, dq, ddq

        assert len(num_obs) == num_total, "num_obsの長さが自由度×3と一致しない"

        obs_onehot_list = []

        for m in range(num_total):
            joint = m // 3
            mod = m % 3  # 0: q, 1: dq, 2: ddq

            bin_num = num_obs[m]

            if mod == 0:
                values = (qs_sampled[:, joint:joint+1] % (2*np.pi)) - np.pi
                v_min, v_max = qs_min[joint], qs_max[joint]
            elif mod == 1:
                values = dqs_sampled[:, joint:joint+1]
                v_min, v_max = dqs_min[joint], dqs_max[joint]
            else:
                values = ddqs_sampled[:, joint:joint+1]
                v_min, v_max = ddqs_min[joint], ddqs_max[joint]

            onehot = quantize_and_onehot(values, bin_num, v_min, v_max)
            obs_onehot_list.append(onehot)  # onehot.shape = (time_steps, bin_num)
            # print(onehot)

        # self.obs_onehot = np.array(obs_onehot_list, dtype=object)  # (num_total,)のobject配列で中身 (T, bin数)
        self.obs_onehot = obs_onehot_list


        """
        for joint in range(nq):
            # q
            bin_q = num_obs[joint*3 + 0]
            onehot_q = quantize_and_onehot(
                qs_sampled[:, joint:joint+1] % (2*np.pi) - np.pi,
                bin_q,
                qs_min[joint],
                qs_max[joint]
            )
            obs_onehot_list.append(onehot_q)

            # dq
            bin_dq = num_obs[joint*3 + 1]
            onehot_dq = quantize_and_onehot(
                dqs_sampled[:, joint:joint+1],
                bin_dq,
                dqs_min[joint],
                dqs_max[joint]
            )
            obs_onehot_list.append(onehot_dq)

            # ddq
            bin_ddq = num_obs[joint*3 + 2]
            onehot_ddq = quantize_and_onehot(
                ddqs_sampled[:, joint:joint+1],
                bin_ddq,
                ddqs_min[joint],
                ddqs_max[joint]
            )
            obs_onehot_list.append(onehot_ddq)
            """

        # for joint in range(nq):
        #     # q
        #     bin_q = num_obs[joint*3 + 0]
        #     onehot_q = quantize_and_onehot(
        #         qs_sampled[:, joint:joint+1] % (2*np.pi) - np.pi,
        #         bin_q,
        #         qs_min[joint],
        #         qs_max[joint]
        #     )
        #     obs_onehot_list[joint*3 + 0] = np.squeeze(onehot_q)

        #     # dq
        #     bin_dq = num_obs[joint*3 + 1]
        #     onehot_dq = quantize_and_onehot(
        #         dqs_sampled[:, joint:joint+1],
        #         bin_dq,
        #         dqs_min[joint],
        #         dqs_max[joint]
        #     )
        #     obs_onehot_list[joint*3 + 1] = np.squeeze(onehot_dq)

        #     # ddq
        #     bin_ddq = num_obs[joint*3 + 2]
        #     onehot_ddq = quantize_and_onehot(
        #         ddqs_sampled[:, joint:joint+1],
        #         bin_ddq,
        #         ddqs_min[joint],
        #         ddqs_max[joint]
        #     )
        #     obs_onehot_list[joint*3 + 2] = np.squeeze(onehot_ddq)

        # # # 結合
        # self.obs_onehot = [[obs_onehot_list[m][t] for m in range(len(num_obs))] for t in range(self.time_steps)]
        # # self.obs_onehot = obs_onehot_list
        return self.obs_onehot


    def current_obs(self, time):
        if time >= self.time_steps:
            time = self.time_steps - 1
        obs = [self.obs_onehot[m][time] for m in range(len(self.num_obs))]
        return obs


# ユーティリティ関数
def quantize_and_onehot(values, num_bins, v_min, v_max):
    # 正規化 [0,1]
    norm_values = (values - v_min) / (v_max - v_min)
    norm_values = np.squeeze(np.clip(norm_values, 0, 1))
    # print(f"norm_value:{norm_values}")
    # インデックス化
    indices = np.squeeze((norm_values * (num_bins - 1)).astype(int))
    # print(f"index:{indices}")
    # one-hot化
    onehot = np.eye(num_bins)[indices]
    return onehot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_onehot_timeseries(obs_onehot, title="Onehot Observation History"):
    """
    One-hot化された観測のビン番号の時系列をプロット

    Parameters:
    - obs_onehot: (T, N) one-hot配列
    - target_indices: 可視化したい状態量のone-hotのインデックス範囲 (リスト or タプル)
                     例: joint0の角度なら (0, 10)
    - t_dense: 時刻リスト (長さT)
    - title: グラフタイトル
    """
    obs_onehot = np.array(obs_onehot)
    bins = obs_onehot.shape[1]

    # 各時刻で one-hot -> ビン番号に変換
    obs_idx = np.argmax(obs_onehot, axis=1)

    t_dense = np.arange(len(obs_onehot))

    plt.figure(figsize=(8, 4))
    plt.plot(t_dense, obs_idx, marker='o')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Bin Index")
    plt.ylim(-0.5, bins-0.5)
    plt.grid(True)
    plt.show()


def plot_obs_heatmap(obs_onehot, title="Onehot Observation History"):
    """
    Onehot化された観測のヒートマップを描画する  
    obs_onehot : (timesteps, num_bins)
    """
    plt.figure(figsize=(10, 4))
    ax = sns.heatmap(
        obs_onehot.T,  # 転置：時刻×ビン → ビン×時刻
        cmap="viridis",
        cbar=True,
        xticklabels=10,
        yticklabels=1
    )
    ax.invert_yaxis()    # 目盛りの間隔だけ指定
    # ax.yaxis.set_major_locator(MultipleLocator(5))
    # ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Bin index")
    plt.tight_layout()
    plt.show()


# エージェントを定義
class RobotPerceptor:
    """
    ロボットの認識を行うクラス
    """
    def __init__(
            self,
            num_obs,
            num_states=None,
            batch_size:int=1,
            num_history:int=16,
            noise:float=0.01,
            eps:float=10e-8,
            Avars=[1.0, 1.0, 1.0],
            Bvars=[1.0, 1.0, 1.0],
            Ainit='diagonal',
            Binit='diagonal',
            Dinit='sigmoid',
            modality_per_joint:int=None,
            Bdepends=False,
            sincos_encoding:bool=False
    ):
        # 生成モデルのパラメータ
        self.num_obs = num_obs
        self.num_modalities = len(num_obs)
        if num_states is None:
            num_states = num_obs
        else:
            self.num_states = num_states
        self.num_factors = len(num_states)
        self.num_controls = [1 for _ in range(self.num_factors)]

        # その他のパラメータ
        self.batch_size = batch_size # バッチサイズ
        self.num_history = num_history # 考慮する履歴の長さ
        self.modality_per_joint = modality_per_joint # 各関節のモダリティ数
        self.factor_per_joint = self.modality_per_joint # 各関節の状態数（モダリティ数と同じ）
        self.sincos_encoding = sincos_encoding # Trueならば角度にsincosエンコーディングを使用
        self.dof = len(self.num_obs) // self.modality_per_joint # ロボットの自由度
        # ロボットの自由度（DoF）と状態変数の個数の整合性チェック
        # 例：状態変数が6個、自由度が3ならばOK（3で割り切れる）
        assert self.num_factors % self.modality_per_joint == 0, "num_factors must be divisible by dof"

        # A,Bの依存関係を定義
        self.A_dependencies = [[f] for f in range(self.num_modalities)] # A行列の依存関係を定義

        # B行列の依存関係を定義
        if Bdepends:
            if sincos_encoding:
                assert self.modality_per_joint >= 2
                if self.modality_per_joint <= 2:
                    dep_joint
                    dep_joint = [[i-1, i] for i in range(1, self.modality_per_joint)]
                else:
                    dep_joint = [[0,1], [0,1], [0,1,2]] + [[i-1, i] for i in range(3, self.modality_per_joint)] if self.modality_per_joint > 2 else [[0,1], [0,1]]
            else:
                dep_joint = [[0]] + [[i-1, i] for i in range(1, self.modality_per_joint)]
            self.B_dependencies = [[d + self.modality_per_joint*j for d in subdep] for j in range(self.dof) for subdep in dep_joint]
        else:
            # B行列の依存関係を単純化（各状態変数が独立）
            self.B_dependencies = [[f] for f in range(self.num_factors)]

        # self.B_dependencies = [[i] for i in range(self.num_factors)] # agents.compute_expected_stateが複雑な依存関係に対応していないので、ここではNoneに設定 ← やっぱりいけるかも

        # ノイズ
        self.eps = eps
        # 尤度の分散
        if len(Avars) == self.factor_per_joint:
            self.Avars = Avars * self.dof
        elif len(Avars) == self.num_factors:
            self.Avars = Avars
        else:
            raise ValueError(
                f"Invalid length for Avars. Expected length: {self.factor_per_joint} or {self.num_factors}, "
                f"but got {len(Avars)}."
            )
        # 遷移の分散
        if len(Bvars) == self.factor_per_joint:
            self.Bvars = Bvars * self.dof
        elif len(Bvars) == self.num_factors:
            self.Bvars = Bvars
        else:
            raise ValueError(
                f"Invalid length for Bvars. Expected length: {self.factor_per_joint} or {self.num_factors}, "
                f"but got {len(Bvars)}."
            )

        # 生成モデルを初期化
        self.Ainit = Ainit
        self.Binit = Binit
        self.Dinit = Dinit
        self.initialize_A()
        self.initialize_B()
        self.initialize_C()
        self.initialize_D()
        self.initialize_pA()
        self.initialize_pB()

        # エージェントを初期化
        self.construct_agents()

        # 行動、観測、状態の初期化
        self.actions_t = 0 # 初期時刻では行動は未定義
        self.actions = None # 過去の行動
        self.observations = None # 過去の観測
        self.outcomes = None # 過去の観測

        # 推論のための引数（信念）の初期化
        self.infer_args = [self.agents.D, None] # [事前信念, 過去の状態(認識)]
        
        # 履歴を保存するリストの初期化
        self.vfe_hist = [] # 変分自由エネルギーの時系列データ
        self.kld_hist = [] # Kullback-Leibler divergenceの時系列データ
        self.bs_hist = [] # ベイジアンサプライズの結果の時系列データ
        self.un_hist = [] # 不確実性の時系列データ
        self.obs_hist = [] # 観測の時系列データ
        self.qs_hist = [] # 状態(認識)の時系列データ
        self.beliefs_hist = [] #信念の時系列データ
        self.actions_hist = [] # 行動の時系列データ
        self.A_hist = [] # A行列の時系列データ
        self.B_hist = [] # B行列の時系列データ

        self.FEparam = None # 変分自由エネルギーのパラメータを格納する辞書型データ

    def reset(self):
        """
        履歴を初期化
        """
        # 行動、観測、状態の初期化
        self.actions_t = 0 # 初期時刻では行動は未定義
        self.actions = None # 過去の行動
        self.observations = None # 過去の観測
        self.outcomes = None # 過去の観測

        # 推論のための引数（信念）の初期化
        self.infer_args = [self.agents.D, None] # [事前信念, 過去の状態(認識)]
        
        # 履歴を保存するリストの初期化
        self.vfe_hist = [] # 変分自由エネルギーの時系列データ
        self.kld_hist = [] # Kullback-Leibler divergenceの時系列データ
        self.bs_hist = [] # ベイジアンサプライズの結果の時系列データ
        self.un_hist = [] # 不確実性の時系列データ
        self.obs_hist = [] # 観測の時系列データ
        self.qs_hist = [] # 状態(認識)の時系列データ
        self.beliefs_hist = [] #信念の時系列データ
        self.actions_hist = [] # 行動の時系列データ
        self.A_hist = [] # A行列の時系列データ
        self.B_hist = [] # B行列の時系列データ

        self.FEparam = None # 変分自由エネルギーのパラメータを格納する辞書型データ


        
    def initialize_A(self):
        A_shapes = [[no] + [self.num_states[fidx] for _, fidx in enumerate(self.A_dependencies[m])] for m, no in enumerate(self.num_obs)]
        if self.Ainit == 'prior':
            # 尤度のプライアを作成
            # 位置：特になし、少しの分散
            # 速度：対数正規分布で分散は速度に比例
            # 加速度：観測の精度は悪めか？
            A_shapes = [[no] + [self.num_states[fidx] for _, fidx in enumerate(self.A_dependencies[m])] for m, no in enumerate(self.num_obs)]
            A = utils.obj_array_zeros(A_shapes)
            for m in range(self.num_modalities):
                no = self.num_obs[m]
                for dep_idx, fidx in enumerate(self.A_dependencies[m]):
                    ns = self.num_states[fidx]
                    for o in range(no):
                        for s in range(ns):
                            delta = np.abs(s - o)
                            sigma = self.Avars[m] * ns
                            # 位置：von Mises分布
                            if m % 3 == 0:
                                if delta > ns / 2:
                                    delta = ns - delta
                                kappa = 1 / sigma ** 2
                                A[m][o, s] = np.exp(kappa * np.cos(2 * np.pi * delta / ns)) + self.eps
                            
                            # 速度：分散が速度に比例する log-normal的分布
                            elif m % 3 == 1:
                                # 観測速度に応じた分散 (deltaも0ベース index なので中心はns/2)
                                obs_speed = np.abs(o - no // 2)
                                # 速度の絶対値によって分散が変わる
                                sigma = sigma * (1 + np.abs(obs_speed) / (ns // 2))
                                A[m][o, s] = np.exp(- (delta ** 2) / (2 * sigma ** 2)) + self.eps
                                
                                # #debug tmp
                                # if o == 0:
                                #     print(f"m: {m}, o: {o}, s: {s}, delta: {delta}, sigma: {sigma}")
                                #     print(f"A[{m}][{o}, {s}] = {A[m][o, s]}")
                                #     A[m][o, s] = 1

                            # 加速度：広めの正規分布
                            elif m % 3 == 2:
                                sigma = sigma
                                A[m][o, s] = np.exp(- (delta ** 2) / (2 * sigma ** 2)) + self.eps

        elif self.Ainit == 'flat':
            # 均一なAで初期化
            A = utils.obj_array_zeros(A_shapes)
            A = A + 1

        elif self.Ainit == 'diagonal':
            # 対角行列で初期化
            A = utils.obj_array_zeros(A_shapes)
            for m in range(self.num_modalities):
                no = self.num_obs[m]
                for dep_idx, fidx in enumerate(self.A_dependencies[m]):
                    ns = self.num_states[fidx]
                    for o in range(no):
                        for s in range(ns):
                            delta = np.abs(s-o)
                            if delta > ns/2:
                                delta = ns - delta
                            kappa = 2.0
                            A[m][o,s] = np.exp(kappa * np.cos(2 * np.pi * delta / ns)) + self.eps
        else:
            assert False, ("Ainit must be 'prior' or 'flat' or 'diagonal'")
        self.A = utils.norm_dist_obj_arr(A)
        self.A_jax = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), list(self.A))
        return self.A

    def initialize_B(self):
        if self.Binit == 'prior':
            # 遷移のプライアを作成（ここが単純に次の時刻の事前分布のようになる）
            # 位置：静止（変化しない、対角行列的）バイアス
            # 速度：低速中心の冪則的な形状、現在の速度を考慮するべきか
            # 加速度：０（中心）がピークのバイアス
            # self.B_dependencies = [[f] for f in range(self.num_factors)]
            # B = utils.initialize_empty_B(self.num_states, self.num_controls)
            B_shapes = [[ns] + [self.num_states[fidx] for _, fidx in enumerate(self.B_dependencies[f])] + [self.num_controls[f]] for f, ns in enumerate(self.num_states)]
            B = utils.obj_array_zeros(B_shapes)
            for f in range(self.num_factors):
                ns = self.num_states[f]
                for s1 in range(ns):
                    for s2 in range(ns):
                        delta = np.abs(s1 - s2)
                        sigma = self.Bvars[f] * ns
                            
                        # 位置：von Mises型
                        if f % 3 == 0:
                            if delta > ns / 2:
                                delta = ns - delta
                            kappa = 1 / sigma ** 2
                            B[f][s1, s2] = np.exp(kappa * np.cos(2 * np.pi * delta / ns))

                        # 速度：なめらか変化
                        elif f % 3 == 1:
                            sigma = sigma
                            B[f][s1, s2] = np.exp(- (delta ** 2) / (2 * sigma ** 2))

                        # 加速度：ゼロ中心の正規分布
                        elif f % 3 == 2:
                            sigma = sigma
                            B[f][s1, s2] = np.exp(- (delta ** 2) / (2 * sigma ** 2))

        elif self.Binit == 'flat':
            # 均一なBで初期化
            B_shapes = [[ns] + [self.num_states[fidx] for _, fidx in enumerate(self.B_dependencies[f])] + [self.num_controls[f]] for f, ns in enumerate(self.num_states)]
            B = utils.obj_array_zeros(B_shapes)
            B = B + 1
            B = utils.norm_dist_obj_arr(B)
        elif self.Binit == 'random':
            # ランダムなBで初期化
            B = utils.random_B_matrix(self.num_states, self.num_controls, self.B_dependencies)
        elif self.Binit == 'diagonal':
            # 対角行列で初期化
            # self.B_dependencies = [[f] for f in range(self.num_factors)]
            # B = utils.initialize_empty_B(self.num_states, self.num_controls)
            B_shapes = [[ns] + [self.num_states[fidx] for _, fidx in enumerate(self.B_dependencies[f])] + [self.num_controls[f]] for f, ns in enumerate(self.num_states)]
            B = utils.obj_array_zeros(B_shapes)
            for f in range(self.num_factors):
                ns = self.num_states[f]
                for s1 in range(ns):
                    for s2 in range(ns):
                        delta = abs(s1 - s2)

                        # von mises分布
                        # kappa = 2.0
                        # B[f][s1,s2] = np.exp(kappa * np.cos(2 * np.pi * delta / ns))

                        # 正規分布
                        sigma = 1.0  # 標準偏差は適宜調整
                        B[f][s1, s2] = np.exp(- (delta ** 2) / (2 * sigma ** 2))
        else:
            assert False, ("Binit must be 'prior' or 'flat' or 'random' or 'diagonal'")
        self.B = utils.norm_dist_obj_arr(B)
        self.B_jax = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), list(self.B))
        return self.B


    
    def initialize_D(self, gain=5.0):
        if self.Dinit == 'flat':
            # 均一なDで初期化
            D = utils.obj_array_ones([ns for ns in self.num_states])
        elif self.Dinit == 'sigmoid':
            # シグモイド型のDを定義
            D = utils.obj_array_ones([ns for ns in self.num_states])
            for i, _ in enumerate(self.num_states):
                x = np.linspace(-1, 1, len(D[i]))  # 入力範囲を[-1, 1]に設定
                D[i] = 1 / (1 + np.exp(-gain * x))  # シグモイド関数の適用
                D[i] /= np.sum(D[i])  # 正規化
        else:
            assert False, ("Dinit must be 'flat' or 'sigmoid'")
        self.D = D
        self.D_jax = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), list(self.D))
        return self.D
    
    def initialize_C(self):
        # 均一なCで初期化(選好なし)
        self.C = utils.obj_array_zeros([num_ob for num_ob in self.num_obs])
        self.C_jax = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), list(self.C))
        return self.C
    
    def initialize_pA(self):
        # Aと同じ形のpAをディリクレ分布で初期化
        self.pA = utils.dirichlet_like(self.A)
        self.pA_jax = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), list(self.pA))
        return self.pA
    
    def initialize_pB(self):
        # Bと同じ形のpBをディリクレ分布で初期化
        self.pB = utils.dirichlet_like(self.B)
        self.pB_jax = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), list(self.pB))
        return self.pB   

    def print_info(self):
        print("RobotPerceptor Info:")
        print(f"  dof: {self.dof}")
        print(f"  modality_per_joint: {self.modality_per_joint}")
        print(f"  factor_per_joint: {self.factor_per_joint}")
        print(f"  num_obs: {self.num_obs}")
        print(f"  num_states: {self.num_states}")
        print(f"  num_modalities: {self.num_modalities}")
        print(f"  num_factors: {self.num_factors}")
        print(f"  num_controls: {self.num_controls}")
        print(f"  eps: {self.eps}")
        print(f"  Avars: {self.Avars}")
        print(f"  Bvars: {self.Bvars}")
        print(f"  Ainit: {self.Ainit}")
        print(f"  Binit: {self.Binit}")
        print(f"  Dinit: {self.Dinit}")
        print(f"  A_dependencies: {self.A_dependencies}")
        print(f"  B_dependencies: {self.B_dependencies}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  num_history: {self.num_history}")
        print(f"  sincos_encoding: {self.sincos_encoding}")
    
    def construct_agents(self):
        self.agents = AIFAgent(
            A=self.A_jax,
            B=self.B_jax,
            C=self.C_jax,
            D=self.D_jax,
            E=None,
            pA=self.pA_jax,
            pB=self.pB_jax,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            onehot_obs=True,
            inference_algo="mmp"
        )

    # 最適化用の最低限の知覚
    def run_perception_for_optimize(self, timesteps, env):
        """
        vfe_hist, kld_hist, bs_hist, un_histのリストを返す
        """
        batch_keys = jr.split(jr.PRNGKey(0), self.batch_size)
        # 環境の時刻を初期化
        T = 0
        vfes = np.zeros(timesteps)
        FEparam = None
        actions_t = 0 # 初期時刻では行動は未定義
        actions = None # 過去の行動
        observations = None # 過去の観測
        outcomes = None # 過去の観測
        beliefs = None
        beliefs_orig = None

        # 推論のための引数（信念）の初期化
        infer_args = [self.agents.D, None] # [事前信念, 過去の状態(認識)]

        while True:
            # 環境から観測を取得
            obs = env.current_obs(T)
            # 履歴に記録
            self.obs_hist.append(obs)
            # バッチに合わせて観測を整形
            outcome_t = [jnp.expand_dims(jnp.expand_dims(o, 0), 0).astype(jnp.float32) for o in obs]
            if outcomes is None:
                outcomes = outcome_t
            else:
                outcomes = jtu.tree_map(lambda prev_o, new_o: jnp.concatenate([prev_o, new_o], 1), outcomes, outcome_t)

            # エージェントを更新
            (
                self.agents,
                outcomes,
                actions,
                infer_args,
                batch_keys,
                FEparam,
                beliefs,
                beliefs_orig
            ) = update_agent(
                self.agents,
                outcomes,
                actions,
                infer_args,
                batch_keys,
                batch_size=self.batch_size,
                num_history=self.num_history
            )

            # データの成形
            # データのidxは、hist[batch_idx][tau_idx]
            vfe = FEparam['vfe'][0][-1]  # 変分自由エネルギー
            # kld = self.FEparam['kld'][0][-1]  # Kullback-Leibler divergence
            # bs = self.FEparam['bs'][0][-1]    # ベイジアンサプライズ
            # un = self.FEparam['un'][0][-1]    # 不確実性

            # 履歴に保存
            vfes[T] = vfe

            # 時間ステップを更新
            T += 1

            if T == timesteps:
                break
        return vfes

    def run_perception(self, timesteps, env, savedir="./test_data", filename=None):
        """
        環境を観測し、認識を行う
        :param timesteps: 観測する時間ステップ数
        :param env: 環境オブジェクト
        :return: 認識結果のリスト
        """
        batch_keys = jr.split(jr.PRNGKey(0), self.batch_size)
        start_time = time.time()
        # 環境の時刻を初期化
        T = 0

        for _ in tqdm(range(timesteps), desc="Perception Loop", position=0, leave=False):
            # 環境から観測を取得
            obs = env.current_obs(T)
            # 履歴に記録
            self.obs_hist.append(obs)
            # バッチに合わせて観測を整形
            outcome_t = [jnp.expand_dims(jnp.expand_dims(o, 0), 0).astype(jnp.float32) for o in obs]
            if self.outcomes is None:
                self.outcomes = outcome_t
            else:
                self.outcomes = jtu.tree_map(lambda prev_o, new_o: jnp.concatenate([prev_o, new_o], 1), self.outcomes, outcome_t)

            # エージェントを更新
            (
                self.agents,
                self.outcomes,
                self.actions,
                self.infer_args,
                batch_keys,
                self.FEparam,
                self.beliefs_orig,
                self.beliefs
            ) = update_agent(
                self.agents,
                self.outcomes,
                self.actions,
                self.infer_args,
                batch_keys,
                batch_size=self.batch_size,
                num_history=self.num_history
            )

            # データの成形
            # データのidxは、hist[batch_idx][tau_idx]
            vfe = self.FEparam['vfe'][0][-1]  # 変分自由エネルギー
            kld = self.FEparam['kld'][0][-1]  # Kullback-Leibler divergence
            bs = self.FEparam['bs'][0][-1]    # ベイジアンサプライズ
            un = self.FEparam['un'][0][-1]    # 不確実性

            # 履歴に保存
            # vfe関係
            self.vfe_hist.append(vfe)
            self.kld_hist.append(kld)
            self.bs_hist.append(bs)
            self.un_hist.append(un)
            # 観測、状態、行動の履歴
            self.obs_hist.append(self.observations)
            self.qs_hist.append(self.infer_args[1][-1])
            self.actions_hist.append(self.actions)
            self.A_hist.append(self.agents.A)
            self.B_hist.append(self.agents.B)

            # 時間ステップを更新
            T += 1

            if T == timesteps:
                break
        
        # タスク終了時の処理
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Perception completed in {elapsed_time:.2f} seconds.")

        if T == timesteps:
            os.makedirs(savedir, exist_ok=True)
            if filename is None:
                # 現在の時刻を取得してファイル名に追加
                self.timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                self.filepath = savedir + f"test_{self.timestam}.pickle"
            else:
                self.filepath = savedir + filename
            # 認識結果を保存
            with open(self.filepath, mode='wb') as fo:
                pickle.dump(self, fo)
            print(f"\rMetadata saved to '{self.filepath}")
        return
    
    def plot_fe_all(
            self,
            filename=None,
            vfe=True,
            kld=True,
            bs=True,
            un=True,
            ):
        # データの取得
        # 保存したpickleファイルを読み込む
        if filename is not None:
            try:
                with open(filename, mode='rb') as fi:
                    loaded_data = pickle.load(fi)
                # データの取得
                vfes = loaded_data.get('vfes', [])
                kld_history = loaded_data.get('kld_history', [])
                bs_history = loaded_data.get('bs_history', [])
                un_history = loaded_data.get('un_history', [])
            except FileExistsError:
                print(f"file '{filename} not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
        # または現在保持しているデータ
        else:
            vfes = self.vfe_hist
            kld_history = self.kld_hist
            bs_history = self.bs_hist
            un_history = self.un_hist

        # # データのidxは、hist[t_idx][batch_idx][tau_idx]
        # vfes = vfes[:][0][-1]
        # kld_history = kld_history[:][0][-1]
        # bs_history = bs_history[:][0][-1]
        # un_history = un_history[:][0][-1]


                # 自由エネルギーのプロット
        if vfe or kld or bs or un:
            plt.figure(figsize=(10, 5))
            # vfeのプロット
            if vfe or kld or bs or un:
                if vfe:
                    plt.plot(range(len(vfes)), vfes, label="VFE", color='orange')
                if kld:
                    plt.plot(range(len(kld_history)), kld_history, label="KLD", linestyle='dashed', color='red')
                if bs:
                    plt.plot(range(len(bs_history)), bs_history, label="BS", linestyle='dashed', color='blue')
                if un:
                    plt.plot(range(len(un_history)), un_history, label="UN", linestyle='dashed', color='grey')

                plt.xlabel("Time Steps")
                plt.ylabel("VFE and Components")
                plt.title("VFE and Its Components Over Time")
                plt.legend()
                plt.grid()
                plt.show()


# エージェントの更新関数
@partial(jit, static_argnames=['batch_size', 'num_history'])
def update_agent(
    agents,
    observations,
    actions,
    infer_args,
    batch_keys,
    batch_size=1,
    num_history=16
):
    beliefs, err, vfe, kld2, bs, un = agents.infer_states_vfe(
        observations,
        infer_args[0],
        past_actions=actions,
        qs_hist=infer_args[1]
    )
    beliefs_orig = beliefs  # 元の信念を保存

    # vfe, bs, unをすべての状態因子について足し合わせる
    vfe = jnp.sum(jnp.stack(jtu.tree_leaves(vfe), axis=0), axis=0)
    bs = jnp.sum(jnp.stack(jtu.tree_leaves(bs), axis=0), axis=0)
    un = jnp.sum(jnp.stack(jtu.tree_leaves(un), axis=0), axis=0)

    # kldの計算
    # kldはすべての状態について足し合わせたものを得る
    # kld.shape = (batch_size, T(0<T<=16))
    kld = agents.calc_KLD_past_currentqs(infer_args[0], infer_args[1], beliefs_orig)

    # バッチサイズに基づいてランダムキーを分割
    batch_keys = jr.split(batch_keys[0], batch_size)
    # 行動をサンプリング
    dummy_q_pi, _neg_efe = agents.infer_policies(beliefs)
    next_action = jnp.zeros((dummy_q_pi.shape[0], len(agents.num_controls)), dtype=int)
    if actions is not None:
        actions = jnp.concatenate([actions, jnp.expand_dims(next_action, -2)], -2)
    else:
        actions = jnp.expand_dims(next_action, -2)

    # 観測、信念、行動の履歴を更新
    observations = tree_map(lambda x: x[:, -num_history:], observations)
    beliefs = tree_map(lambda x: x[:, -num_history:], beliefs)
    actions = tree_map(lambda x: x[:, -num_history:], actions)

    # Dの更新
    agents = tree_at(lambda x: x.D, agents, tree_map(lambda x: x[:, 0], beliefs))
    
    # 次の推論のために事前信念を更新
    if infer_args[1] is None:
        empirical_prior = infer_args[0]
    else:
        empirical_prior = agents.compute_expected_state(next_action, infer_args[1])
    infer_args[0] = empirical_prior
    infer_args[1] = beliefs

    return agents, observations, actions, infer_args, batch_keys, {
        'vfe': vfe,
        'kld': kld,
        'bs': bs,
        'un': un,
    }, beliefs_orig, beliefs


# 可視化関数
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import math

def plot_A_all(A, max_cols=3):
    """
    A行列またはA_jax行列を可視化する関数。

    Parameters:
        A_like : list of ndarray or DeviceArray
            A行列のリスト、もしくはA_jax行列のリスト
        max_cols : int
            1行あたりの最大列数
    """
    num_modalities = len(A)

    # A_jaxの場合（先頭次元が batch_size に等しい場合）→ 先頭次元を落とす
    if isinstance(A[0], (np.ndarray, jnp.ndarray)) and A[0].ndim >= 3:
        print("Detected A_jax format, converting to A format for visualization.")
        A = [np.array(a[0]) for a in A]  # jax→numpy化しつつ先頭次元除去

    # subplotの行数・列数を計算
    n_cols = min(max_cols, num_modalities)
    n_rows = math.ceil(num_modalities / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # axesを常に1次元list化
    if num_modalities == 1:
        axes = [axes]
    else:
        axes = np.ravel(axes).tolist()
    # fig, axes = plt.subplots(1, num_modalities, figsize=(5 * num_modalities, 4))

    # if num_modalities == 1:
    #     axes = [axes]

    for m in range(num_modalities):
        ax = axes[m]
        sns.heatmap(A[m], cmap='viridis', ax=ax)
        ax.set_title(f'Modality {m}')
        ax.set_xlabel('State')
        ax.set_ylabel('Observation')

    plt.tight_layout()
    plt.show()


def plot_B_all(B, max_cols=3):
    """
    B行列またはB_jax行列を可視化する関数。
    行動軸（最後の軸）は平均化して表示。

    Parameters:
        B : list of ndarray or DeviceArray
            B行列のリスト、もしくはB_jax行列のリスト
        max_cols : int
            1行あたりの最大列数
    """
    num_factors = len(B)

    # B_jaxの場合（先頭次元が batch_size に等しい場合）→ 先頭次元を落とす
    if isinstance(B[0], (np.ndarray, jnp.ndarray)) and B[0].ndim >= 4:
        print("Detected B_jax format, converting to B format for visualization.")
        B = [np.array(b[0]) for b in B]

    # subplotの行数・列数を計算
    n_cols = min(max_cols, num_factors)
    n_rows = math.ceil(num_factors / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # axesを常に1次元list化
    if num_factors == 1:
        axes = [axes]
    else:
        axes = np.ravel(axes).tolist()

    for f in range(num_factors):
        ax = axes[f]

        # 最後の軸（行動軸）を平均化
        B_mean = np.mean(B[f], axis=-1)

        sns.heatmap(B_mean, cmap='viridis', ax=ax)
        ax.set_title(f'Factor {f} (Transition)')
        ax.set_xlabel('Previous State')
        ax.set_ylabel('Next State')

    # 余ったsubplotを非表示に
    for i in range(num_factors, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()


# 使用例
# from optimize import *
# import pinocchio as pin
# import numpy as np

# # ロボットの自由度
# dof = 3
# # モデル・データ読み込み
# model = get_simple_arm(dof)  # 自分のURDFを読み込むなら buildModelFromUrdf
# data = model.createData()
# nq = model.nq

# # 初期姿勢
# q = pin.neutral(model)

# # エンドエフェクタのframe名とid
# frame_name = "dummy_link"
# frame_id = model.getFrameId(frame_name)

# # 例
# pi = np.pi
# tau = 2*pi
# time_steps = 500
# dt = 0.01
# qz = pin.neutral(model)
# # qs = random_qs_spline(model, time_steps, start, end) # 仮の軌跡

# # 目標位置と姿勢（SE3）
# theta = np.pi/4
# Rs = pin.utils.rotate("y", -3*theta)
# start_placement = pin.SE3(Rs, np.array([-2, 0, 0]))
# Re = pin.utils.rotate("y", 3*theta)
# end_placement = pin.SE3(Re, np.array([2, 0, 0]))

# start = ik_se3_solver(model, data, frame_id, start_placement)
# end = ik_se3_solver(model, data, frame_id, end_placement)

# limits = same_limits(model.nq)
# num_knots = 10

# # 感性評価準備
# # 観測の次元
# num_obs_per_joint = [10,30,5]
# num_obs = []
# for i in range(dof):
#     num_obs += num_obs_per_joint

# # エージェントの定義
# agent = RobotPerceptor(
#     num_obs=num_obs,  # 各関節の観測数
#     num_states=num_obs,  # 各関節の状態数
#     batch_size=1,
#     num_history=16,
#     Ainit='prior',
#     Binit='prior',
#     modality_per_joint=3,
#     sincos_encoding=False
# )

# opt = Optimizer(
#     robot=model,
#     agent=agent,
#     timesteps=time_steps,
#     dt=dt,
#     start=start,
#     end=end,
#     limits=limits,
# )

# opt.optimize(vfe=1.0, iters=1)
