# 最適化関連
from .robot import *
from .perception import *
from .utils import *
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
import pinocchio as pin
import pyswarms as ps
from functools import partial
# import jax.numpy as jnp
from jax import jit, vmap

# 保存用
import os
import pickle
import datetime
from pathlib import Path
from typing import Dict, Any

from scipy.optimize import minimize

from typing import Optional, Dict
from .utils import OptimizationResult # <--- 必要に応じてインポート
from tqdm.notebook import trange
import math

import copy

# H/M/S 形式に変換するヘルパー関数
def format_time(seconds: float) -> str:
    """秒数を H時間 M分 S.ss秒 の形式にフォーマットする。"""
    seconds = round(seconds, 2)
    h = math.floor(seconds / 3600)
    seconds -= h * 3600
    m = math.floor(seconds / 60)
    seconds -= m * 60
    
    if h > 0:
        return f"{h}h {m}m {seconds:.2f}s"
    elif m > 0:
        return f"{m}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"
    
class Optimizer:
    def __init__(
            self,
            robot,
            agent,
            timesteps,
            dt,
            start,
            end,
            limits=None,
            num_knots=9,
            compensate_grav=False
            ):
        
        # エージェントと環境を初期化
        if agent is not None:
            self.agent = agent
            self.env = RobotArmEnv(
                num_obs = self.agent.num_obs,
                time_steps = timesteps,
                robot = robot
            )
        
        # 信念の初期化
        self.const_beliefs = None
        self.const_beliefs_qs = None
        self.const_beliefs_result = None

        # モデルと軌跡の端点を初期化
        self.model = robot
        self.nq = self.model.nq
        self.data = self.model.createData()
        self.timesteps = timesteps
        self.dt = dt
        self.fps = int(1/self.dt)
        self.start = start
        self.end = end
        # 可動域の設定
        self.limits = [self.model.lowerPositionLimit, self.model.upperPositionLimit]
        if limits is not None:
            self.limits = limits
        # スプライン制御点と粒子の数
        self.num_knots = num_knots
        # self.n_particles = n_particles
        self.compensate_grav = compensate_grav

        # 結果を格納する変数の初期化
        self.best_cost = None
        self.best_particle = None
        self.best_qs = None

        # 保存フォルダパスを格納するメンバ変数を追加
        self.save_folder_path: Optional[Path] = None


    def _generate_folder_name(self) -> str:
        """
        Optimizerの主要な設定に基づき、一意で特徴的なフォルダ名を生成する。
        """

        # モデル名またはDOF
        robot_dof = self.model.nq

        # 現在のタイムスタンプ
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 特徴的な条件を組み合わせたフォルダ名
        folder_name = (
            f"SIM"
            f"_Dof{robot_dof}"                # DOF
            f"_T{self.timesteps}"           # タイムステップ
            f"_K{self.num_knots}"           # ノット点数
            # f"_P{self.n_particles}"         # 粒子数
            # f"_{belief_flag}" 
            f"_{timestamp}"
        )
        
        return folder_name

    def print_pyswarms_progress(self, i, iters, best_cost, bar_width=20, prefix="PSO"):
        """
        PySwarms純正風の進捗表示（printのみ）
        """
        progress = (i + 1) / iters
        filled = int(bar_width * progress)
        bar = "█" * filled + " " * (bar_width - filled)
        percent = int(progress * 100)

        msg = (
            f"{prefix}: "
            f"{percent:3d}%|{bar}| "
            f"{i+1}/{iters}, "
            f"best_cost={best_cost:.4e}"
        )

        print(msg, end="\r", flush=True)


    # def setup_save_directory(self, base_dir: str = "results") -> Path:
    #     """
    #     設定に基づいたフォルダ名を作成し、Optimizerの初期設定とconst_beliefsを保存する。

    #     Parameters
    #     ----------
    #     base_dir : str, default="results"
    #         結果を保存するベースディレクトリ。

    #     Returns
    #     -------
    #     Path
    #         実際に作成された保存フォルダのフルパス。
    #     """
    #     # 1. フォルダ名の生成とパスの作成
    #     folder_name = self._generate_folder_name()
    #     save_path = Path(base_dir) / folder_name
        
    #     # 2. フォルダの作成
    #     os.makedirs(save_path, exist_ok=True)
        
    #     # 3. Optimizer設定（変更のないパラメータ群）の保存
    #     config_data: Dict[str, Any] = {
    #         # ロボット情報
    #         'dof': self.nq,
    #         'model_name': getattr(self.model, 'name', f"PinocchioModel_DOF{self.nq}"),
    #         # 時間情報
    #         'timesteps': self.timesteps,
    #         'dt': self.dt,
    #         # 軌道情報
    #         'num_knots': self.num_knots,
    #         'start_config': self.start.tolist(), # ndarrayをリストに変換
    #         'end_config': self.end.tolist(),     # ndarrayをリストに変換
    #         'limits_low': self.limits[0].tolist(),
    #         'limits_high': self.limits[1].tolist(),
    #         # PSO情報
    #         # 'n_particles': self.n_particles,
    #         # エージェント情報
    #         'agent_params': self.agent.get_params() if hasattr(self, 'agent') and self.agent is not None else 'N/A',
    #         'const_beliefs_active': self.const_beliefs is not None,
    #     }
        
    #     config_file = save_path / "optimizer_config.pkl"
    #     with open(config_file, 'wb') as f:
    #         pickle.dump(config_data, f)
        
    #     # 4. const_beliefs の保存（**修正部分**）
    #     if self.const_beliefs_result is not None:
    #         # const_beliefs が save メソッドを持っていることを期待
    #         if hasattr(self.const_beliefs_result, 'save') and callable(self.const_beliefs_result.save):
    #             beliefs_folder_path = save_path / "const_beliefs" # .pklを付与してファイルとして扱う
    #             beliefs_file_path = beliefs_folder_path / "const_beliefs_result.pkl"
    #             try:
    #                 # saveメソッドを呼び出して、const_beliefs自身に保存処理を行わせる
    #                 # saveメソッドはファイルパス全体を受け取ることを想定
    #                 self.const_beliefs_result.save(str(beliefs_file_path)) 
    #                 plot_robot_motion(self.const_beliefs_result, folder_path=beliefs_folder_path, file_name="motion")
    #             except Exception as e:
    #                 print(f"Warning: const_beliefs.save() failed: {e}")
    #         else:
    #             # saveメソッドがない場合は、代わりにpickleで直接保存を試みる (フォールバック)
    #             print("Warning: const_beliefs object does not have a 'save' method. Falling back to direct pickle save.")
    #             beliefs_file = save_path / "const_beliefs_fallback.pkl"
    #             try:
    #                 with open(beliefs_file, 'wb') as f:
    #                     pickle.dump(self.const_beliefs_result, f)
    #             except Exception as e:
    #                 print(f"Warning: Failed to pickle const_beliefs (fallback): {e}")
    #     # 5. フォルダパスをメンバ変数に保存
    #     self.save_folder_path = save_path
        
    #     return save_path


    # def initialize_beliefs(self, jerk=0.0, energy=0.0, torque_change=0.0, compensate_grav=True, maxiter=1000):
    #     """
    #     ScipyのL-BFGS-Bを用いた勾配ベースの最適化によってジャークやエネルギー消費量を最小化し、
    #     事前分布として設定する。
    #     """
    #     best_cost, best_particle, best_qs = self.scipy_optimize(jerk=jerk, energy=energy, torque_change=torque_change, compensate_grav=compensate_grav, maxiter=maxiter)
    #     self.set_const_beliefs(best_qs)
    #     return best_cost, best_particle, best_qs

    def alltimes_scipy_optimize(self, jerk=0.0, energy=0.0, torque_change=0.0, compensate_grav=True, maxiter=1000, initial_seed=0):
        """
        ScipyのL-BFGS-Bを用いた勾配ベースの最適化。
        最適化変数として全時間ステップの姿勢 (qs) を使用する。
        """
        
        # 1. 初期解の生成と変数の設定
        # 初期姿勢 qs (timesteps, nq) を生成し、それをフラット化して最適化変数とする
        
        # 便宜的に、ここではlinear_base_particleの出力（中間ノット）から初期qsを生成すると仮定
        # 実際の初期qs生成ロジックに合わせて修正してください。
        initial_particle, initial_qs = self.linear_base_particle()
        
        # ★ 初期軌道 initial_qs を生成 (例: 線形補間 or スプライン)
        # ここでは、元のコードのようにスプラインから初期qsを生成していますが、
        # 実際のタスクに応じて、単純な線形補間など、他の方法でも構いません。
        # initial_qs = qs_from_particle(
        #     initial_particle, self.model, self.timesteps, self.start, 
        #     self.end, self.limits, num_knots=self.num_knots, free_end_knot=False)
        
        # ★ 全時間ステップの姿勢 (qs) を最適化変数とする
        initial_x = initial_qs.flatten()
        dimensions = initial_x.size # dimensions = timesteps * nq
        
        # 2. 可動域（ボックス制約）の設定
        bounds = []
        lowers = self.limits[0]
        uppers = self.limits[1]
        
        # ★ 全時間ステップの各関節の可動域をバインドとして設定
        # 制約: (timesteps) x (nq)
        for t in range(self.timesteps):
            for j in range(self.model.nq):
                # 始点 (t=0) と終点 (t=timesteps-1) は固定値として制約を設定（厳密には制約の再定義が必要）
                if t == 0:
                    # 始点姿勢を固定 (下限=上限)
                    bounds.append((self.start[j], self.start[j])) 
                elif t == self.timesteps - 1:
                    # 終点姿勢を固定 (下限=上限)
                    bounds.append((self.end[j], self.end[j]))
                else:
                    # 中間の姿勢は可動域で制限
                    bounds.append((lowers[j], uppers[j]))
        
        # 3. 目的関数の定義
        def objective_scipy_all_qs(x):
            # ★ 入力 x は (timesteps * nq) のフラットなベクトル
            # qsの形状に復元: (timesteps, nq)
            qs = x.reshape(self.timesteps, self.model.nq)
            dt = self.dt

            # ★ 速度、加速度、ジャークを差分（np.gradient）から計算
            # スプライン補間ではないため、生のqsから差分で計算する
            dqs = np.gradient(qs, dt, axis=0)      # 速度
            ddqs = np.gradient(dqs, dt, axis=0)    # 加速度
            dddqs = np.gradient(ddqs, dt, axis=0)  # ジャーク

            total_cost = 0.0
            # data = self.model.createData()
            data = self.data

            if jerk != 0.0:
                # compute_total_jerkは、ジャークの二乗積分を計算（上で確認済み）
                total_cost += jerk * compute_total_jerk(dddqs, self.dt)

            if energy != 0.0:
                # 重力補償ありで評価
                total_cost += energy * compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav)

            if torque_change != 0.0:
                total_cost += torque_change * compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt)
                
            # ★ 軌道の滑らかさを強制するために、速度・加速度のコストを追加することを強く推奨
            # 例: total_cost += 1e-4 * np.sum(dqs**2) * dt  
            #     total_cost += 1e-4 * np.sum(ddqs**2) * dt
                
            return total_cost

        # 4. 最適化の実行
        print(f"Total optimization variables (timesteps * nq): {dimensions}")
        print(f"Starting L-BFGS-B optimization...")
        
        result = minimize(
            objective_scipy_all_qs, # 新しい目的関数
            initial_x,             # 全姿勢をフラット化した初期値
            method='L-BFGS-B', 
            bounds=bounds,         # 全姿勢に対するボックス制約
            options={'maxiter': maxiter, 'disp': True, 'ftol': 1e-8}
        )
        
        # 5. 結果の格納
        best_cost = result.fun
        
        # ★ 最適化されたフラットなベクトルを qs (timesteps, nq) の形状に戻す
        best_x = result.x
        best_qs = best_x.reshape(self.timesteps, self.model.nq)
        
        print(f"\nOptimization Status: {result.message}")
        print(f"Final Cost: {best_cost:.4f}")

        # ★ 返り値は best_cost と best_qs のみ
        return best_cost, best_qs, initial_qs

    
    def scipy_optimize(self, jerk=0.0, energy=0.0, torque_change=0.0, compensate_grav=None, maxfun=15000, maxiter=1500, ftol=1e-8, initial_seed=0):
        """
        ScipyのL-BFGS-Bを用いた勾配ベースの最適化。
        ジャークやエネルギー消費量を最小化する。
        """
        # 0. 乱数シードの設定
        if initial_seed is not None:
            np.random.seed(initial_seed)

        if compensate_grav is None:
            compensate_grav = self.compensate_grav

        # 1. 初期解の生成
        # 線形補間から求めた粒子を初期値として使用
        initial_particle, _ = self.linear_base_particle()
        # dimensions = initial_particle.size
        
        # 2. 可動域（ボックス制約）の設定
        bounds = []
        lowers = self.limits[0]
        uppers = self.limits[1]
        
        # 中間ノット点の可動域をバインドとして設定
        # 変数の次元数 = (num_knots - 2) * nq
        for i in range(self.num_knots - 2):
            for j in range(self.model.nq):
                bounds.append((lowers[j], uppers[j]))
        
        repetitions = self.num_knots - 2 
        
        # lowers/uppers (nq次元) を repetitions 回繰り返して dimensions 次元にする
        full_lowers = np.tile(lowers, repetitions) 
        full_uppers = np.tile(uppers, repetitions) 
        
        # 正しい形状の low/high を使って乱数粒子を生成
        random_particle = np.random.uniform(low=full_lowers, high=full_uppers) 
        
        initial_particle = random_particle
        print(f"initial_particle: {initial_particle}")
        initial_qs = qs_from_particle(initial_particle, self.model, self.timesteps, self.start, 
            self.end, self.limits, num_knots=self.num_knots, free_end_knot=False)

        # 3. 目的関数の定義（PSOのobjectiveから再利用）
        # ここでは、簡略化のため、VFEなどの複雑な項は除外
        def objective_scipy(particle):
            # particle (x) はフラットなベクトル
            qs, grads = qs_from_particle(particle,
                                model=self.model,
                                time_steps=self.timesteps,
                                start=self.start,
                                end=self.end,
                                limits=self.limits,
                                num_knots=self.num_knots,
                                free_end_knot=False, # 終点固定を前提
                                grad1=True,
                                grad2=True,
                                grad3=True
                                )
            dqs_raw, ddqs_raw, dddqs_raw = grads
            dt = self.dt
            
            # 1. 速度 (dqs) の補正
            dqs = dqs_raw / dt 
            
            # 2. 加速度 (ddqs) の補正
            ddqs = ddqs_raw / (dt ** 2)
            
            # 3. ジャーク (dddqs) の補正
            dddqs = dddqs_raw / (dt ** 3)  # <--- dtの3乗で割る！

            total_cost = 0.0
            # data = self.model.createData()
            data = self.data

            if jerk != 0.0:
                total_cost += jerk * compute_total_jerk(dddqs, self.dt)

            if energy != 0.0:
                # 重力補償ありで評価
                total_cost += energy * compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav)

            if torque_change != 0.0:
                total_cost += torque_change * compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav)
                
            return total_cost

        # 4. 最適化の実行
        result = minimize(
            objective_scipy,
            initial_particle, # 初期姿勢
            method='L-BFGS-B', # 境界制約付きの最適化手法を選択
            bounds=bounds,     # ボックス制約
            options={'maxfun': maxfun, 'maxiter': maxiter, 'disp': True, 'ftol': ftol} # ログ表示
        )
        
        # 5. 結果の格納
        best_cost = result.fun
        best_particle = result.x
        best_qs, grads = qs_from_particle(
            best_particle, self.model, self.timesteps, self.start, 
            self.end, self.limits, num_knots=self.num_knots, free_end_knot=False,
            grad1=True, grad2=True, grad3=True)
        qs = best_qs
        dqs_raw, ddqs_raw, dddqs_raw = grads
        dt = self.dt
        dqs = dqs_raw / dt
        ddqs = ddqs_raw / (dt ** 2)
        dddqs = dddqs_raw / (dt ** 3)
        data = self.data
        final_metrics = {
            'jerk': compute_total_jerk(dddqs, self.dt),
            'energy': compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav),
            'torque_change': compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav)
        }

        print(f"\nOptimization Status: {result.message}")
        print(f"Final Cost: {best_cost:.4f}")
        

        # 6.3. ロボットモデルキーの特定
        robot_model_key = getattr(self.model, 'name', f"PinocchioModel_DOF{self.model.nq}")

        # 6.4. OptimizationResultのインスタンス化
        optimization_result = OptimizationResult(
            # 1. ロボットモデル関連
            dof=self.model.nq,
            robot_model_key=robot_model_key,
            robot_model=self.model,
            start=self.start,
            end=self.end,
            limits_low=self.limits[0],
            limits_high=self.limits[1],
            
            # 2. シミュレーション/可視化パラメータ
            dt=self.dt,
            time_steps=self.timesteps,
            fps=self.fps, # (Optimizerクラスにfps属性があればそれを使用)
            compensate_grav=compensate_grav,

            # 3. 最適化プロセスパラメータ
            optimizer_type='Scipy_L-BFGS-B',
            optimizer_params={"num_knots": self.num_knots, "maxiter": maxiter, "ftol": ftol, "random_seed": initial_seed},
            cost_func={"cost_func": f"jerk*{jerk} + energy*{energy} + torque_change*{torque_change} -> 0.0"},

            # 4. Perception Agentパラメータ
            # agent_params=agent_params,
            agent_params={},
            
            # 5. 最適化結果
            qs=best_qs,
            best_cost=best_cost,
            final_metrics=final_metrics,
            best_particle=best_particle,
            
            # 6. 履歴・ステータス (L-BFGS-Bの情報を格納)
            cost_history=None, 
            pos_history=None,
            optimization_summary={
                # 最終的な最適解
                "x": result.x, 
                "fun": result.fun,
                "jac": result.jac,
                
                # 計算量情報
                "nfev": result.nfev, # 目的関数評価回数
                "nit": result.nit,  # 反復回数
                
                # 終了ステータス情報
                "success": result.success,
                "status": result.status,
                "message": result.message,
                },
            )
        
        # OptimizationResultオブジェクトを返す
        return optimization_result
        

    def set_const_beliefs(self, result):
        self.const_beliefs_result = result
        self.const_beliefs_qs = result.qs
        self.const_beliefs = self.agent.generate_const_beliefs(self.env, self.const_beliefs_qs, self.dt)
        return
    


    def linear_base_particle(self):
        num_knots = self.num_knots
        start = self.start
        end = self.end
        timesteps = self.timesteps

        if num_knots < 2:
            base_particle = np.array([])
        else:
            # 制御点のインデックス（0からnum_knots-1）
            knot_indices = np.arange(num_knots)
            
            # 制御点の時間的な割合 u_i
            # 0.0 (始点) から 1.0 (終点) まで等間隔
            u = knot_indices / (num_knots - 1)
            
            # 全制御点（始点から終点まで）の値を線形補間で計算
            # q_knot[i] = (1 - u[i]) * start + u[i] * end
            # np.newaxisを使ってブロードキャストにより全関節を一括計算
            all_knot_values = (1 - u[:, np.newaxis]) * start + u[:, np.newaxis] * end
            
            # 中間制御点のみを抽出 (始点(0)と終点(-1)を除く)
            intermediate_knot_values = all_knot_values[1:-1]
            
            # PSOの入力形式に合わせるため、ベクトル化（flatten）して返却
            base_particle = intermediate_knot_values.flatten()

            t_norm = np.linspace(0.0, 1.0, num=timesteps)

            # 2. 差分ベクトル q_diff = q_end - q_start を計算
            q_diff = end - start

            # 3. 線形補間を実行
            # 各時刻 t_i における姿勢 qs[i] は、qs[i] = q_start + t_norm[i] * q_diff
            
            # t_norm を (timesteps, 1) の形状にし、q_diff と start をブロードキャスト
            # t_norm[:, np.newaxis] は (timesteps, 1)
            # start, q_diff は (nq,)
            base_qs = start + t_norm[:, np.newaxis] * q_diff
        return base_particle, base_qs

    

    def pso_minimize(self, jerk=0.0, energy=0.0, torque_change=0.0, vfe=0.0, kld=0.0, bs=0.0, un=0.0, vfe_var=0.0, iters=1, n_particles=30, ftol=1e-8, tolsteps=5, compensate_grav=None, end_penalty=0.0, temporal_approach_cost=0.0):
        """
        粒子群最適化（PSO: Particle Swarm Optimization）を用いて、
        ロボットアームの最適軌道を求める関数。

        各粒子は中間ノット点（姿勢）の集合を表し、目標コスト関数を最小化するよう探索を行う。
        コスト関数は、運動の滑らかさやエネルギー消費、トルク変化、
        さらには自由エネルギー原理に基づく指標（VFE, KLD, BSなど）を組み合わせて構成できる。

        Parameters
        ----------
        jerk : float, default=0.0
            ジャーク（二階微分の変化量）に基づく滑らかさのコスト係数。

        energy : float, default=0.0
            総消費エネルギー（トルク×角速度積分）のコスト係数。

        torque_change : float, default=0.0
            時間的なトルク変化（トルク微分）のコスト係数。

        vfe : float, default=0.0
            変分自由エネルギー（Variational Free Energy）の平均に基づくコスト係数。

        kld : float, default=0.0
            クルバック・ライブラー距離（Kullback–Leibler Divergence）のコスト係数（未使用だが拡張用）。

        bs : float, default=0.0
            BS項のコスト係数（未使用だが拡張用）。

        un : float, default=0.0
            Uncertainty（不確実性）項のコスト係数（未使用だが拡張用）。

        vfe_var : float, default=0.0
            VFEの分散に基づくコスト係数。VFEの時間変動が小さい安定した軌道を優先。

        iters : int, default=1
            粒子群最適化の反復回数。

        compensate_grav : bool, default=True
            Trueの場合、重力項を補償してトルクを評価（重力補償なしでの純粋な運動エネルギー評価も可能）。

        end_penalty : float, default=0.0
            終点到達誤差(||q(T) - q_end||^2)に基づくペナルティ係数。
            end_penalty > 0.0 の場合、終点ノットは自由変数となり、終点制約が緩和される。
            end_penalty = 0.0 の場合、終点ノットは固定される。

        Returns
        -------
        best_cost : float
            最小化されたコスト関数の値。

        best_particle : ndarray
            最適な粒子（ノット点パラメータ）の配列。

        best_qs : ndarray
            最適な関節角度列（タイムステップごとの姿勢）。

        Notes
        -----
        - 内部では PySwarms の `GlobalBestPSO` を利用している。
        - 各粒子は `(num_knots - 2) * model.nq` 次元のベクトルとして表現される。
        - 評価関数 `particle_objective()` は与えられた粒子を姿勢列 `qs` に変換し、
        指定されたコスト項の加重和を返す。
        - 自由エネルギー関連の項（VFEなど）は `self.agent` および `self.env` に依存して計算される。
        - JAXでのベクトル化（`vmap`）は一部未使用だが、並列計算を想定した設計。

        Examples
        --------
        >>> optimizer = TrajectoryOptimizer(model, env, agent, num_knots=10, n_particles=50)
        >>> best_cost, best_particle, best_qs = optimizer.optimize(
        ...     jerk=1.0, energy=0.5, torque_change=0.2, vfe=0.1, iters=100
        ... )
        >>> print("最適コスト:", best_cost)
        >>> visualize_trajectory(model, best_qs)
        """

        if compensate_grav is None:
            compensate_grav = self.compensate_grav
            
        # end_penaltyが有効な場合、終点ノットも最適化変数に含める
        if end_penalty > 0.0:
            # num_knots - 1 個の中間ノット + 終点ノット
            dimensions = (self.num_knots - 1) * self.model.nq 
            # 終点ノットを自由変数として扱うことを示すフラグ
            free_end_knot = True
        else:
            # 従来の通り、num_knots - 2 個の中間ノットのみ
            dimensions = (self.num_knots - 2) * self.model.nq
            free_end_knot = False

        dimensions = (self.num_knots - 2) * self.model.nq
        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options)

        # 目的関数の作成(vmap)
        def particle_objective(particle):
            qs, grads = qs_from_particle(
                particle, 
                model=self.model, 
                time_steps=self.timesteps,
                start=self.start, 
                end=self.end, 
                limits=self.limits, 
                num_knots=self.num_knots,
                free_end_knot=free_end_knot,
                grad1=True,
                grad2=True,
                grad3=True
                )
            dqs, ddqs, dddqs = grads
            dt = self.dt
            # 速度 (1階微分) を補正: dqs / dt
            dqs = dqs / dt
            # 加速度 (2階微分) を補正: ddqs / dt^2
            ddqs = ddqs / (dt ** 2)
            # ジャーク (3階微分) を補正: dddqs / dt^3
            dddqs = dddqs / (dt ** 3)
            data = self.data
            total_cost = 0.0
            if jerk != 0.0:
                # total_cost += jerk * compute_jerk_fn(qs)
                total_cost += jerk * compute_total_jerk(dddqs, self.dt)
            if energy != 0.0:
                # total_cost += energy * compute_energy_fn(qs)
                total_cost += energy * compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav)
            if torque_change != 0.0:
                total_cost += torque_change * compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav)
            if vfe != 0.0 or kld != 0.0 or bs != 0.0 or un != 0.0:
                vfes, klds, bss, uns = compute_total_vfe(self.agent, self.env, qs, self.dt, self.const_beliefs)
                total_vfe = sum(vfes)#/len(qs)
                total_kld = sum(klds)#/len(qs)
                total_bs = sum(bss)#/len(qs)
                total_un = sum(uns)#/len(qs)
                total_vfe_var = np.var(vfes)
                total_cost += vfe * total_vfe  + vfe_var * total_vfe_var + kld * total_kld + bs * total_bs + un * total_un
            if temporal_approach_cost != 0.0:
                # 1. 時間インデックスの配列 [0, 1, 2, ..., T_max]
                times = np.arange(self.timesteps)
                # 2. 時間依存の重み w(t) = (t / T_max)^2 を計算
                # T_maxが0の場合を避ける (通常 timesteps >= 2 のため問題なし)
                T_max = self.timesteps - 1
                weights = (times / T_max)**2 
                # 3. 各時刻での目標終点からの誤差 (qs(t) - q_end) を計算
                # self.end (q_end) はブロードキャストにより全時刻 qs から引かれる
                error_diff = qs - self.end 
                # 4. 誤差の二乗ノルム ||qs(t) - q_end||^2 を計算
                # np.sum(..., axis=1) で関節ごとの二乗和 (ノルム^2) を計算
                squared_error = np.sum(error_diff**2, axis=1) 
                # 5. 重み付けして合計し、総コストに加算
                # self.dt は積分近似のための時間刻み幅
                temporal_cost = np.sum(temporal_approach_cost * weights * squared_error * self.dt)
                total_cost += temporal_cost
            if end_penalty > 0.0:
                end_error = np.linalg.norm(qs[-1] - self.end)**2
                total_cost += end_penalty * end_error
            return total_cost
        
        # vmapで全粒子のコスト並列計算
        batched_objective = lambda particles: jnp.array([particle_objective(p) for p in particles])

        # オプティマイザ実行
        best_cost, best_particle = optimizer.optimize(batched_objective, iters=iters)
        best_qs, grads = qs_from_particle(
            best_particle, self.model, self.timesteps, self.start, 
            self.end, self.limits, num_knots=self.num_knots, free_end_knot=False,
            grad1=True, grad2=True, grad3=True)
        qs = best_qs
        dqs_raw, ddqs_raw, dddqs_raw = grads
        dt = self.dt
        dqs = dqs_raw / dt
        ddqs = ddqs_raw / (dt ** 2)
        dddqs = dddqs_raw / (dt ** 3)
        data = self.data
        metrics = {
            'jerk': compute_total_jerk(dddqs, self.dt),
            'energy': compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav),
            'torque_change': compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav),
        }
        if vfe != 0.0 or kld != 0.0 or bs != 0.0 or un != 0.0:
            vfes, klds, bss, uns = compute_total_vfe(self.agent, self.env, qs, self.dt, self.const_beliefs)
            total_vfe = sum(vfes)#/len(qs)
            total_kld = sum(klds)#/len(qs)
            total_bs = sum(bss)#/len(qs)
            total_un = sum(uns)#/len(qs)
            total_vfe_var = np.var(vfes)
            metrics.update({
                    'vfe': total_vfe,
                    'kld': total_kld,
                    'bs': total_bs,
                    'un': total_un,
                    'vfe_var': total_vfe_var,
                    'ig': total_kld + total_bs
                    })
            
        #コスト重みの収集
        cost_func: Dict[str, float] = {
            'jerk': jerk,
            'energy': energy,
            'torque_change': torque_change,
            'compensate_grav': compensate_grav, # 重力補償フラグも格納
            'vfe': vfe,
            'kld': kld,
            'bs': bs,
            'un': un,
            'vfe_var': vfe_var
        }
        
        # Agentパラメータの収集
        # self.agentが存在しない場合を考慮
        agent_params: Dict = self.agent.get_params() if hasattr(self, 'agent') and self.agent is not None else {}
        
        # 6.3. ロボットモデルキーの特定
        robot_model_key = getattr(self.model, 'name', f"PinocchioModel_DOF{self.model.nq}")

        # 6.4. OptimizationResultのインスタンス化
        # PSO特有の履歴とメトリクス
        pso_cost_history = optimizer.cost_history
        n_actual_iterations = len(pso_cost_history)

        optimization_result = OptimizationResult(
            # 1. ロボットモデル関連
            dof=self.model.nq,
            robot_model_key=robot_model_key,
            robot_model=self.model,
            start=self.start,
            end=self.end,
            limits_low=self.limits[0],
            limits_high=self.limits[1],
            
            # 2. シミュレーション/可視化パラメータ
            dt=self.dt,
            time_steps=self.timesteps,
            fps=self.fps, # (Optimizerクラスにfps属性があればそれを使用)
            
            # 3. 最適化プロセスパラメータ
            optimizer_type='GlobalBestPSO',
            num_knots=self.num_knots,
            n_particles=n_particles,
            max_iter=iters,
            cost_func=cost_func,
            random_seed=None,
            
            # 4. Perception Agentパラメータ
            agent_params=agent_params,
            
            # 5. 最適化結果
            qs=best_qs,
            best_cost=best_cost,
            final_metrics=metrics,
            best_particle=best_particle,
            
            # 6. 履歴・ステータス (L-BFGS-Bの情報を格納)
            cost_history=optimizer.cost_history,
            metric_history={
            'n_actual_iterations': n_actual_iterations,
            'best_cost_initial': pso_cost_history[0],
            },
        )
        
        # OptimizationResultオブジェクトを返す
        return optimization_result
    

    def pso_optimize(self, targets: dict, weight: dict = None, norm="L1", iters=1, n_particles=1, ftol=1e-8, ftol_iter=5, cost_threshold=1e-4, compensate_grav=None, options={'c1': 1.5, 'c2': 1.5, 'w': 0.9}, initial_pos=None):
        """
        粒子群最適化（PSO: Particle Swarm Optimization）を用いて、
        ロボットアームの最適軌道を求める関数。

        各粒子は中間ノット点（姿勢）の集合を表し、各指標と目標値との距離を最小化するよう探索を行う。
        コスト関数は、運動の滑らかさやエネルギー消費、トルク変化、
        さらには自由エネルギー原理に基づく指標（VFE, KLD, BSなど）を組み合わせて構成できる。

        Parameters
        ----------
        targets = {
        "ig": {"target": ig_target_value, "min": ig_min_range, "max": ig_max_range},
        "energy": {"target": energy_target_value, "min": energy_min_range, "max": energy_max_range},
        # ... 他の指標
    }
        
        iters : int, default=1
            粒子群最適化の反復回数。

        compensate_grav : bool, default=True
            Trueの場合、重力項を補償してトルクを評価（重力補償なしでの純粋な運動エネルギー評価も可能）。

        Returns
        -------
        best_cost : float
            最小化されたコスト関数の値。

        best_particle : ndarray
            最適な粒子（ノット点パラメータ）の配列。

        best_qs : ndarray
            最適な関節角度列（タイムステップごとの姿勢）。

        Notes
        -----
        - 内部では PySwarms の `GlobalBestPSO` を利用している。
        - 各粒子は `(num_knots - 2) * model.nq` 次元のベクトルとして表現される。
        - 評価関数 `particle_objective()` は与えられた粒子を姿勢列 `qs` に変換し、
        指定されたコスト項の加重和を返す。
        - 自由エネルギー関連の項（VFEなど）は `self.agent` および `self.env` に依存して計算される。
        - JAXでのベクトル化（`vmap`）は一部未使用だが、並列計算を想定した設計。

        """
        # print("ver4")

        if compensate_grav is None:
            compensate_grav = self.compensate_grav

        dimensions = (self.num_knots - 2) * self.model.nq
        free_end_knot = False

        dimensions = (self.num_knots - 2) * self.model.nq
        # options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, 
                                            dimensions=dimensions, 
                                            options=options)
        

        def particle_objective(particle):
            qs, grads = qs_from_particle(
                particle, 
                model=self.model, 
                time_steps=self.timesteps,
                start=self.start, 
                end=self.end, 
                limits=self.limits, 
                num_knots=self.num_knots,
                free_end_knot=free_end_knot,
                grad1=True,
                grad2=True,
                grad3=True
                )
            dqs, ddqs, dddqs = grads
            dt = self.dt
            # 速度 (1階微分) を補正: dqs / dt
            dqs = dqs / dt
            # 加速度 (2階微分) を補正: ddqs / dt^2
            ddqs = ddqs / (dt ** 2)
            # ジャーク (3階微分) を補正: dddqs / dt^3
            dddqs = dddqs / (dt ** 3)

            data = self.model.createData()
            total_cost = 0.0

            current_values = {
                "jerk": compute_total_jerk(dddqs, self.dt), 
                "energy": compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav), 
                "torque_change": compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav), 
                }

            # targetの値に収束させるための目的関数
            target_label_candidates = [
                "jerk", 
                "energy", 
                "torque_change", 
                "vfe", 
                "kld", 
                "bs", 
                "un", 
                "vfe_var",
                "ig"
            ]
            vfe_elements = [
                "vfe", 
                "kld", 
                "bs", 
                "un", 
                "vfe_var",
                "ig"
            ]
            
            
            # targets辞書のキーの集合を取得
            target_keys_set = set(targets.keys())
            # vfe_elements_listの集合を取得
            vfe_set = set(vfe_elements)

            # 共通部分を計算し、共通要素が存在するかを確認
            # 共通部分の要素数が 0 より大きければ True
            if bool(target_keys_set.intersection(vfe_set)):
                vfes, klds, bss, uns = compute_total_vfe(self.agent, self.env, qs, self.dt, self.const_beliefs)
                total_vfe = sum(vfes)#/len(qs)
                total_kld = sum(klds)#/len(qs)
                total_bs = sum(bss)#/len(qs)
                total_un = sum(uns)#/len(qs)
                total_vfe_var = np.var(vfes)
                total_ig = total_kld + total_bs

                vfe_metrics = {
                    "vfe": total_vfe, 
                    "kld": total_kld, 
                    "bs": total_bs, 
                    "un": total_un, 
                    "vfe_var": total_vfe_var,
                    "ig": total_ig,
                }
                current_values.update(vfe_metrics)

            
            # 目的関数構築
            for metric in targets:
                
                # 1. 最適化対象がターゲット条件に存在するか確認 (関係ないタグは自動的に無視)
                if metric not in target_label_candidates:
                    continue 
                
                # 2. ターゲット、現在値、正規化範囲を取得
                target_data = targets[metric]
                current_val = current_values.get(metric)
                
                # 必須キーの確認
                if not all(key in target_data for key in ['target', 'min', 'max']):
                    print(f"Warning: Target data for '{metric}' is incomplete. Skipping.")
                    continue
                if current_val is None:
                    continue
                    
                target_val = target_data['target']
                min_range = target_data['min']
                max_range = target_data['max']
                
                # 3. 正規化範囲の計算とゼロ除算回避
                metric_range = max_range - min_range
                
                if metric_range <= 1e-16:
                    continue
                    
                # 4. 正規化された偏差の計算
                # NormalizedDeviance = |current - target| / Range
                normalized_deviation = np.abs(current_val - target_val) / metric_range

                w = 1.0
                if weight is not None:
                    w = weight.get(metric, 1.0)

                # 5. 重み付きコストの加算
                if norm == "L1":
                    weighted_cost = w * normalized_deviation  # L1ノルム
                elif norm == "L2":
                    weighted_cost = w * (normalized_deviation ** 2)  # L2ノルム
                total_cost += weighted_cost

            return total_cost
        
        

        # vmapで全粒子のコスト並列計算
        batched_objective = lambda particles, **kwargs: np.array([particle_objective(p) for p in particles])
        # batched_objective = lambda particles: jnp.array([particle_objective(p) for p in particles])

        # オプティマイザ実行
        # best_cost, best_particle = optimizer.optimize(
        #     batched_objective, 
        #     iters=iters, 
        #     callback=check_stop_condition,
        #     ftol=ftol,
        #     ftol_iter=ftol_iter,
        #     n_processes=None,
        #     verbose=True)

        
        # pbar = trange(iters, desc="PSO", unit="iter")

        
        def print_pyswarms_progress(i, iters, best_cost, bar_width=20, prefix="PSO"):
            """
            PySwarms純正風の進捗表示（printのみ）
            """
            progress = (i + 1) / iters
            filled = int(bar_width * progress)
            bar = "█" * filled + " " * (bar_width - filled)
            percent = int(progress * 100)

            msg = (
                f"{prefix}: "
                f"{percent:3d}%|{bar}| "
                f"{i+1}/{iters}, "
                f"best_cost={best_cost:.4e}"
            )

            print(msg, end="\r", flush=True)


        # 最適化を実行
        start_time = time.time()

        print(f"PSO: Starting optimization (Max Iter: {iters}, Cost Target: {cost_threshold:.4e})...")

        # 改善停滞カウンタ（ftol_iterのためのもの）
        ftol_counter = 0
        stop_reason = ""
        stop_value = ""

        for i in range(iters):
            final_iter = i + 1
            best_cost, best_particle = optimizer.optimize(
                batched_objective,
                iters=1,
                n_processes=None,
                verbose=False,
                initial_pos=initial_pos if i==0 else None
            )

            history = optimizer.cost_history
            current_cost = history[-1]

            # PySwarms風進捗表示
            time_elapsed = time.time() - start_time
            prefix = f"PSO({format_time(time_elapsed)})"
            print_pyswarms_progress(i, iters, current_cost, prefix=prefix)

            # # Early stopping
            # if len(history) >= ftol_iter + 1:
            #     diffs = np.abs(np.diff(history[-(ftol_iter + 1):]))
            #     if np.all(diffs < ftol):
            #         print(f"\n[Early Stop] Iteration  : {i+1}, Best cost : {current_cost:.6e}, Diffs : {diffs}")
            #         break

            # A. 絶対コスト閾値による停止 (品質保証)
            if current_cost < cost_threshold:
                stop_reason = f"Absolute Cost Threshold Reached"
                stop_value = f"Cost < {cost_threshold:.4e}"
                break

            # B. ftol_iterによる停止 (改善停滞)
            if len(history) >= 2:
                diff = history[-2] - history[-1]
                
                if diff < ftol:
                    ftol_counter += 1
                else:
                    ftol_counter = 0

                if ftol_counter >= ftol_iter:
                    stop_reason = f"Relative Improvement Stalled"
                    stop_value = f"ΔCost < {ftol:.4e} for {ftol_iter} iters"
                    break
            else:
                # forループがitersまで回りきった場合
                stop_reason = f"Max Iterations Reached"
                stop_value = f"Max Iter: {iters}"

        end_time = time.time()
        elapsed_time = end_time - start_time
        final_iter = i + 1
        formatted_time = format_time(elapsed_time)

        # --- 1行目をクリア ---
        # \r で行頭に戻し、大量の空白で上書き後、再度 \r で行頭に戻す。
        # ここでの120はターミナル幅の目安。
        print("\r" + " " * 120, end="\r", flush=True)

        # --- 最終結果の1行目 ---
        print(f"Completed in {final_iter}/{iters} iters ({formatted_time}). Final Cost: {best_cost:.6e}")
        
        # --- 最終結果の2行目 ---
        print(f"Stop Reason: **{stop_reason}**. Stop Metric: {stop_value}.")
        
        # 複数の条件を連続実行する場合のために区切り線を追加
        print("-" * 80)

        best_qs, grads = qs_from_particle(
            best_particle, self.model, self.timesteps, self.start, 
            self.end, self.limits, num_knots=self.num_knots, free_end_knot=False,
            grad1=True, grad2=True, grad3=True)
        qs = best_qs
        dqs_raw, ddqs_raw, dddqs_raw = grads
        dt = self.dt
        dqs = dqs_raw / dt
        ddqs = ddqs_raw / (dt ** 2)
        dddqs = dddqs_raw / (dt ** 3)
        data = self.data
        vfes, klds, bss, uns = compute_total_vfe(self.agent, self.env, qs, self.dt, self.const_beliefs)
        total_vfe = sum(vfes)#/len(qs)
        total_kld = sum(klds)#/len(qs)
        total_bs = sum(bss)#/len(qs)
        total_un = sum(uns)#/len(qs)
        total_vfe_var = np.var(vfes)

        metrics = {
            'ig': total_kld + total_bs,
            'energy': compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav),
            'vfe': total_vfe,
            'jerk': compute_total_jerk(dddqs, self.dt),
            'torque_change': compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav),
            'kld': total_kld,
            'bs': total_bs,
            'un': total_un,
            'vfe_var': total_vfe_var
            }
        

        target_values_list = [] # ターゲット値リストも同時に作成
        for metric_name, data in targets.items():
            # ターゲット値のリスト項目（例: ig -> 0.5）
            target_values_list.append(f"{metric_name} -> {data['target']}")
        # 主要なターゲット値をまとめた文字列
        target_str = ", ".join(target_values_list)
        cost_func = {"targets": targets, "target_str": target_str} 

        # Agentパラメータの収集
        # self.agentが存在しない場合を考慮
        agent_params: Dict = self.agent.get_params() if hasattr(self, 'agent') and self.agent is not None else {}
        
        # 6.3. ロボットモデルキーの特定
        robot_model_key = getattr(self.model, 'name', f"PinocchioModel_DOF{self.model.nq}")

        # 6.4. OptimizationResultのインスタンス化
        # PSO特有の履歴とメトリクス
        pso_cost_history = optimizer.cost_history
        n_actual_iterations = len(pso_cost_history)

        optimization_result = OptimizationResult(
            # 1. ロボットモデル関連
            dof=self.model.nq,
            robot_model_key=robot_model_key,
            robot_model=self.model,
            start=self.start,
            end=self.end,
            limits_low=self.limits[0],
            limits_high=self.limits[1],
            
            # 2. シミュレーション/可視化パラメータ
            dt=self.dt,
            time_steps=self.timesteps,
            fps=self.fps, # (Optimizerクラスにfps属性があればそれを使用)
            compensate_grav=compensate_grav,

            # 3. 最適化プロセスパラメータ
            optimizer_type='GlobalBestPSO',
            optimizer_params={"num_kots": self.num_knots, 
                              "iters": iters, "n_particles": n_particles, 
                              "ftol": ftol, "tolstetps": ftol_iter},
            cost_func=cost_func,
            
            # 4. Perception Agentパラメータ
            agent_params=agent_params,
            
            # 5. 最適化結果
            qs=best_qs,
            best_cost=best_cost,
            final_metrics=metrics,
            best_particle=best_particle,
            
            # 6. 履歴・ステータス (L-BFGS-Bの情報を格納)
            cost_history=optimizer.cost_history,
            pos_history=optimizer.pos_history,
            optimization_summary={'nit': n_actual_iterations},
        )
        
        # OptimizationResultオブジェクトを返す
        return optimization_result
    

    def hybrid_optimize(self, targets: dict, weight: dict = None, norm="L1", 
                        pso_iters=1, n_particles=30, 
                        scipy_maxiter=500,
                        switch_cost_threshold=1e-3, 
                        ftol=1e-8, ftol_iter=5,
                        compensate_grav=None, 
                        initial_pos=None):
        """
        PSOで大域探索を行い、一定条件を満たした後にScipy(L-BFGS-B)で精密化するハイブリッド最適化。
        """
        if compensate_grav is None:
            compensate_grav = self.compensate_grav

        start_time = time.time()
        
        # ---------------------------------------------------------
        # 1. 共通の目的関数（ターゲット辞書対応）の定義
        # ---------------------------------------------------------
        def shared_objective(particle):
            """pso_optimizeのロジックを継承した正規化目的関数"""
            # particleの形状変換（Scipyは1次元、PSOは2次元で来るため対応）
            p = particle.flatten()
            
            qs, grads = qs_from_particle(
                p, self.model, self.timesteps, self.start, self.end, self.limits,
                num_knots=self.num_knots, free_end_knot=False,
                grad1=True, grad2=True, grad3=True
            )
            dqs, ddqs, dddqs = [g / (self.dt ** (i+1)) for i, g in enumerate(grads)]
            
            data = self.data # 高速化のため使い回し
            current_values = {
                "jerk": compute_total_jerk(dddqs, self.dt), 
                "energy": compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav), 
                "torque_change": compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav), 
            }

            # VFE関連の計算（targetsに含まれる場合のみ実行）
            vfe_keys = {"vfe", "kld", "bs", "un", "vfe_var", "ig"}
            if bool(set(targets.keys()).intersection(vfe_keys)):
                vfes, klds, bss, uns = compute_total_vfe(self.agent, self.env, qs, self.dt, self.const_beliefs)
                current_values.update({
                    "vfe": sum(vfes), "kld": sum(klds), "bs": sum(bss), "un": sum(uns),
                    "vfe_var": np.var(vfes), "ig": sum(klds) + sum(bss)
                })

            # 正規化コスト計算
            total_cost = 0.0
            for metric, target_data in targets.items():
                current_val = current_values.get(metric)
                if current_val is None: continue
                
                metric_range = target_data['max'] - target_data['min']
                if metric_range <= 1e-16: continue
                
                normalized_deviation = np.abs(current_val - target_data['target']) / metric_range
                w = weight.get(metric, 1.0) if weight else 1.0
                
                if norm == "L1":
                    total_cost += w * normalized_deviation
                else: # L2
                    total_cost += w * (normalized_deviation ** 2)
                    
            return total_cost

        # PSO用のバッチ処理ラッパー
        def batched_obj(particles, **kwargs):
            return np.array([shared_objective(p) for p in particles])

        # ---------------------------------------------------------
        # 2. 第1フェーズ：PSOによる大域探索
        # ---------------------------------------------------------
        dimensions = (self.num_knots - 2) * self.model.nq
        pso_options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
        pso_optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=pso_options)
        
        print(f"Phase 1: PSO (Max {pso_iters} iters, Switch Threshold: {switch_cost_threshold:.2e})")
        
        best_cost_pso = float('inf')
        best_particle_pso = None
        ftol_counter = 0

        # 切り替え理由の初期値（ループを完走した場合はこれになる）
        stop_reason = "Max PSO Iterations Reached"
        pso_actual_iters = pso_iters

        for i in range(pso_iters):
            cost, pos = pso_optimizer.optimize(batched_obj, iters=1, verbose=False, initial_pos=initial_pos if i==0 else None)
            best_cost_pso, best_particle_pso = cost, pos
            pso_actual_iters = i + 1
            
            # 進捗表示（print_pyswarms_progressは既存のものを使用）
            self.print_pyswarms_progress(i, pso_iters, best_cost_pso, prefix="PSO")

            # 切り替えトリガー1: コストが閾値を下回った
            if best_cost_pso < switch_cost_threshold:
                stop_reason = "Absolute Cost Threshold Reached"
                print(f"\n[Switch] Cost {best_cost_pso:.4e} < {switch_cost_threshold:.2e}. Switching to Scipy.")
                break
                
            # 切り替えトリガー2: 改善の停滞 (ftol)
            if len(pso_optimizer.cost_history) >= 2:
                diff = pso_optimizer.cost_history[-2] - pso_optimizer.cost_history[-1]
                if diff < ftol:
                    ftol_counter += 1
                else:
                    ftol_counter = 0
                if ftol_counter >= ftol_iter:
                    stop_reason = "Relative Improvement Stalled"
                    print(f"\n[Switch] {stop_reason}: ΔCost < {ftol:.2e} for {ftol_iter} iters.")
                    break

        combined_pos_history = list(pso_optimizer.pos_history)

        # ---------------------------------------------------------
        # 3. 第2フェーズ：Scipy (L-BFGS-B) による局所精緻化
        # ---------------------------------------------------------
        print(f"\nPhase 2: Scipy L-BFGS-B (Max {scipy_maxiter} iters)")
        
        scipy_trajectory = []

        def scipy_callback(xk):
            """Scipyの各反復終了時に呼ばれる関数"""
            # Scipyの粒子は1つなので、PSOの形 (n_particles, dims) に合わせるなら 
            # (1, dims) として保存しておくと後で扱いやすい
            scipy_trajectory.append(xk.copy().reshape(1, -1))

        # 境界制約の設定
        lowers, uppers = self.limits
        bounds = [(lowers[j], uppers[j]) for _ in range(self.num_knots - 2) for j in range(self.model.nq)]

        res = minimize(
            shared_objective,
            best_particle_pso, # PSOの結果を初期値にする
            method='L-BFGS-B',
            bounds=bounds,
            callback=scipy_callback,
            options={'maxiter': scipy_maxiter, 'ftol': ftol, 'disp': True}
        )

        # ---------------------------------------------------------
        # 4. 結果の統合とOptimizationResultの作成
        # ---------------------------------------------------------
        final_particle = res.x

        # 最終的なメトリクスを再計算
        final_qs, grads = qs_from_particle(
                final_particle, self.model, self.timesteps, self.start, self.end, self.limits,
                num_knots=self.num_knots, free_end_knot=False,
                grad1=True, grad2=True, grad3=True
            )
        dqs, ddqs, dddqs = [g / (self.dt ** (i+1)) for i, g in enumerate(grads)]
        vfes, klds, bss, uns = compute_total_vfe(self.agent, self.env, final_qs, self.dt, self.const_beliefs)
        total_vfe = sum(vfes)#/len(qs)
        total_kld = sum(klds)#/len(qs)
        total_bs = sum(bss)#/len(qs)
        total_un = sum(uns)#/len(qs)
        total_vfe_var = np.var(vfes)
        total_ig = total_kld + total_bs
        metrics = {
            'ig': total_ig,
            'energy': compute_total_energy(self.model, self.data, final_qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav),
            'vfe': total_vfe,
            'jerk': compute_total_jerk(dddqs, self.dt),
            'torque_change': compute_total_torque_change(self.model, self.data, final_qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav),
            'kld': total_kld,
            'bs': total_bs,
            'un': total_un,
            'vfe_var': total_vfe_var
            }
        
        # 記録
        # PSOの履歴の末尾に、Scipyの軌跡を繋げる
        combined_pos_history.extend(scipy_trajectory)
        
        target_values_list = [] # ターゲット値リストも同時に作成
        for metric_name, data in targets.items():
            # ターゲット値のリスト項目（例: ig -> 0.5）
            target_values_list.append(f"{metric_name} -> {data['target']}")
        # 主要なターゲット値をまとめた文字列
        target_str = ", ".join(target_values_list)
        cost_func = {"targets": targets, "target_str": target_str} 

        # Agentパラメータの収集
        # self.agentが存在しない場合を考慮
        agent_params: Dict = self.agent.get_params() if hasattr(self, 'agent') and self.agent is not None else {}
        

        result = OptimizationResult(
            # 1. ロボットモデル関連
            dof=self.model.nq,
            robot_model_key=getattr(self.model, 'name', f"PinocchioModel_DOF{self.model.nq}"),
            robot_model=self.model,
            start=self.start,
            end=self.end,
            limits_low=self.limits[0],
            limits_high=self.limits[1],
            
            # 2. シミュレーション/可視化パラメータ
            dt=self.dt,
            time_steps=self.timesteps,
            fps=self.fps, # (Optimizerクラスにfps属性があればそれを使用)
            compensate_grav=compensate_grav,

            # 3. 最適化プロセスパラメータ
            optimizer_type='Hybrid(PSO+L-BFGS-B)',
            optimizer_params = {
                # 共通設定
                "num_knots": self.num_knots,
                "norm": norm,
                "ftol": ftol,
                "ftol_iter": ftol_iter,
                # "cost_threshold": cost_threshold,
                
                # PSOフェーズの設定と結果
                "pso_max_iters": pso_iters,
                "pso_actual_iters": pso_actual_iters,
                "pso_n_particles": n_particles,
                "pso_switch_threshold": switch_cost_threshold,
                
                # Scipyフェーズの設定と結果
                "scipy_method": "L-BFGS-B",
                "scipy_maxiter": scipy_maxiter,
                "scipy_actual_nit": res.nit,    # Scipyが実際に要した反復数
                "scipy_nfev": res.nfev,         # 関数評価回数（計算負荷の指標）
                
                # ハイブリッド特有のステータス
                "switched_by": stop_reason,     # "threshold" か "stalled" か "max_pso_iter" か
            },
            cost_func=cost_func,
            
            # 4. Perception Agentパラメータ
            agent_params=agent_params,
            
            # 5. 最適化結果
            qs=final_qs,
            best_cost=res.fun,
            final_metrics=metrics,
            best_particle=final_particle,
            
            # 6. 履歴・ステータス (L-BFGS-Bの情報を格納)
            cost_history=pso_optimizer.cost_history + [res.fun],
            pos_history=combined_pos_history,
            optimization_summary={
                'pso_iters': len(pso_optimizer.cost_history),
                'scipy_iters': len(scipy_trajectory),
                'total_iters': len(combined_pos_history),
                'scipy_nit': res.nit,
                'success': res.success,
                'message': res.message
            }
        )
        
        print(f"Hybrid Optimization Completed. Final Cost: {res.fun:.6e}")
        return result
    

    def evaluate_particle(self, particle_vector, compensate_grav=None):
        """
        並列実行セーフな評価メソッド。
        各プロセスでエージェントをコピーすることで、状態汚染を防ぐ。
        """
        # 1. 状態のリセット（Deepcopy）
        # 2回目以降の0.1秒を信じて、毎回クリーンなコピーを作成
        local_agent = copy.deepcopy(self.agent)
        local_env = copy.deepcopy(self.env)
        
        # クラスのデフォルト設定を適用
        if compensate_grav is None:
            compensate_grav = self.compensate_grav

        # 2. 軌跡生成（スプライン）
        final_qs, grads = qs_from_particle(
            particle_vector, self.model, self.timesteps, self.start, self.end, self.limits,
            num_knots=self.num_knots, free_end_knot=False,
            grad1=True, grad2=True, grad3=True
        )
        dqs, ddqs, dddqs = [g / (self.dt ** (i+1)) for i, g in enumerate(grads)]

        # 3. VFE・情報の計算 (コピーした agent/env を使用)
        vfes, klds, bss, uns = compute_total_vfe(
            local_agent, local_env, final_qs, self.dt, self.const_beliefs
        )
        
        # 4. 物理メトリクスの計算
        # 辞書にまとめて返す
        metrics = {
            'ig': sum(klds) + sum(bss),
            'energy': compute_total_energy(self.model, self.data, final_qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav),
            'vfe': sum(vfes),
            'jerk': compute_total_jerk(dddqs, self.dt),
            'vfe_var': np.var(vfes),
            'pos': particle_vector  # 再現用にパラメータを保持
        }
        
        return metrics

        

# 以下は指標計算用関数
# 軌跡からサプライズ（収束したvfe）を計算する関数
def compute_total_vfe(agent, env, qs, dt, const_beliefs):
    env.computeAllobs(qs,dt)
    time_steps = len(qs)
    # vfes, klds, bss, uns = agent.run_perception_for_optimize(timesteps = time_steps, env = env)
    vfes, klds, bss, uns = agent.bayse_estimate(timesteps=time_steps, env=env, const_beliefs=const_beliefs)
    return vfes, klds, bss, uns


# 軌跡からジャークを計算する関数
def compute_total_jerk(dddqs, dt):
    # dqs = np.gradient(qs, dt, axis=0)
    # ddqs = np.gradient(dqs, dt, axis=0)
    # dddqs = np.gradient(ddqs, dt, axis=0)

    # dddqs = dddqs / dt**3
    jerk_cost = np.sum(dddqs**2) * dt
    return jerk_cost

# def make_compute_total_jerk_jax(dt):
#     @jit
#     def compute(qs):
#         dqs = jnp.gradient(qs, dt, axis=0)
#         ddqs = jnp.gradient(dqs, dt, axis=0)
#         dddq = jnp.gradient(ddqs, dt, axis=0)
#         return jnp.sum(dddq**2) * dt
#     return compute

# 軌跡からエネルギーを計算する関数
def compute_total_energy(model, data, qs, dqs, ddqs, dt, compensate_grav=False):
    total_energy = 0.0
    time_steps = len(qs)
    # # jnpは使うとダメ
    # dqs = np.gradient(qs, dt, axis=0)
    # ddqs = np.gradient(dqs, dt, axis=0)

    # dqs = dqs / dt
    # ddqs = ddqs / dt**2

    for q, dq, ddq in zip(qs, dqs, ddqs):
        # 全体のトルク
        pin.computeAllTerms(model, data, q, dq)
        tau_total = pin.rnea(model, data, q, dq, ddq)

        if compensate_grav:
            # 重力のみのトルク
            tau_gravity = pin.computeGeneralizedGravity(model, data, q)
            # 動的成分のみ取り出す
            tau = tau_total - tau_gravity
        else:
            tau = tau_total

        energy = np.sum(np.abs(tau * dq)) * dt
        total_energy += energy

    return total_energy

# 動力学トルクの変化
def compute_total_torque_change(model, data, qs, dqs, ddqs, dt, compensate_grav=False):
    taus = []
    # dqs = np.gradient(qs, dt, axis=0)
    # ddqs = np.gradient(dqs, dt, axis=0)

    # dqs = dqs / dt
    # ddqs = ddqs / dt**2

    for q, dq, ddq in zip(qs, dqs, ddqs):
        pin.computeAllTerms(model, data, q, dq)
        tau_total = pin.rnea(model, data, q, dq, ddq)
        if compensate_grav:
            tau_gravity = pin.computeGeneralizedGravity(model, data, q)
            tau_dynamic = tau_total - tau_gravity
        else:
            tau_dynamic = tau_total
        taus.append(tau_dynamic)

    taus = np.array(taus)
    dtaus = np.gradient(taus, dt, axis=0)
    # ddtaus = np.gradient(dtaus, dt, axis=0)
    torque_change_cost = np.sum(dtaus**2) * dt

    return torque_change_cost


