# 最適化関連
from .robot import *
from .perception import *
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
import pinocchio as pin
import pyswarms as ps
from functools import partial
# import jax.numpy as jnp
from jax import jit, vmap

from scipy.optimize import minimize

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
            num_knots=5,
            n_particles=30):
        
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

        # モデルと軌跡の端点を初期化
        self.model = robot
        self.nq = self.model.nq
        self.data = self.model.createData()
        self.timesteps = timesteps
        self.dt = dt
        self.start = start
        self.end = end
        # 可動域の設定
        self.limits = [self.model.lowerPositionLimit, self.model.upperPositionLimit]
        if limits is not None:
            self.limits = limits
        # スプライン制御点と粒子の数
        self.num_knots = num_knots
        self.n_particles = n_particles

        # 結果を格納する変数の初期化
        self.best_cost = None
        self.best_particle = None
        self.best_qs = None

    def initialize_beliefs(self, jerk=0.0, energy=0.0, torque_change=0.0, compensate_grav=True, maxiter=1000):
        """
        ScipyのL-BFGS-Bを用いた勾配ベースの最適化によってジャークやエネルギー消費量を最小化し、
        事前分布として設定する。
        """
        best_cost, best_particle, best_qs = self.scipy_optimize(jerk=jerk, energy=energy, torque_change=torque_change, compensate_grav=compensate_grav, maxiter=maxiter)
        self.set_const_beliefs(best_qs)
        return best_cost, best_particle, best_qs

    def scipy_optimize(self, jerk=0.0, energy=0.0, torque_change=0.0, compensate_grav=True, maxiter=1000):
        """
        ScipyのL-BFGS-Bを用いた勾配ベースの最適化。
        ジャークやエネルギー消費量を最小化する。
        """
        # 1. 初期解の生成
        # 線形補間から求めた粒子を初期値として使用
        initial_particle, _ = self.linear_base_particle()
        dimensions = initial_particle.size
        
        # 2. 可動域（ボックス制約）の設定
        bounds = []
        lowers = self.limits[0]
        uppers = self.limits[1]
        
        # 中間ノット点の可動域をバインドとして設定
        # 変数の次元数 = (num_knots - 2) * nq
        for i in range(self.num_knots - 2):
            for j in range(self.model.nq):
                bounds.append((lowers[j], uppers[j]))
        
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

            dqs, ddqs, dddqs = grads
            total_cost = 0.0
            data = self.model.createData()

            if jerk != 0.0:
                total_cost += jerk * compute_total_jerk(dddqs, self.dt)

            if energy != 0.0:
                # 重力補償ありで評価
                total_cost += energy * compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav)

            if torque_change != 0.0:
                total_cost += torque_change * compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt)
                
            return total_cost

        # 4. 最適化の実行
        result = minimize(
            objective_scipy,
            initial_particle, # 初期姿勢
            method='L-BFGS-B', # 境界制約付きの最適化手法を選択
            bounds=bounds,     # ボックス制約
            options={'maxiter': maxiter, 'disp': True} # ログ表示
        )
        
        # 5. 結果の格納
        best_cost = result.fun
        best_particle = result.x
        best_qs = qs_from_particle(
            best_particle, self.model, self.timesteps, self.start, 
            self.end, self.limits, num_knots=self.num_knots, free_end_knot=False)
        
        print(f"\nOptimization Status: {result.message}")
        print(f"Final Cost: {best_cost:.4f}")

        return best_cost, best_particle, best_qs
    

    def set_const_beliefs(self, qs):
        self.const_beliefs_qs = qs
        self.const_beliefs = self.agent.generate_const_beliefs(self.env, qs, self.dt)
        return


    # def optimize_torque_change(self):
    #     dimensions = (self.num_knots - 2) * self.model.nq
    #     options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
    #     optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=dimensions, options=options)

    #     def objective(particles, model, data, start, end, limits, time_steps, dt, num_knots):
    #         # particles.shape = (n_particles, dimensions)
    #         costs = []
    #         for particle in particles:
    #             # particle.shape = ((num_knots - 2)*nq,)
    #             # フラットなベクトルを(time_steps, nq)に変換
    #             qs = qs_from_particle(particle,
    #                                 model=model,
    #                                 time_steps=time_steps,
    #                                 start=start,
    #                                 end=end,
    #                                 limits=limits,
    #                                 num_knots=num_knots)
    #             cost = compute_total_torque_change(model, data, qs, dt)
    #             costs.append(cost)
    #         return np.array(costs)

    #     # partialでほかの引数をバインド
    #     obj_func = partial(objective,
    #                     model=self.model,
    #                     data=self.data,
    #                     start=self.start,
    #                     end=self.end,
    #                     limits=self.limits,
    #                     time_steps=self.timesteps,
    #                     dt=self.dt,
    #                     num_knots=self.num_knots)
        
    #     self.best_cost, self.best_particle = optimizer.optimize(obj_func, iters=100)
    #     self.best_qs = qs_from_particle(self.best_particle, self.model, self.timesteps, self.start, self.end, self.limits, num_knots=self.num_knots)
        
    #     return self.best_cost, self.best_particle, self.best_qs
    
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

    
    # def optimize_old(self, jerk=0.0, energy=0.0, torque_change=0.0, vfe=0.0, kld=0.0, bs=0.0, un=0.0, vfe_var=0.0, iters=1):
    #     """
    #     複数のコスト指標を重み付け線形和として最適化する。

    #     Parameters:
    #         jerk (float): 見かけ上のジャークの重み
    #         energy (float): トルクエネルギー消費の重み
    #         torque_change (float): トルク変化量の重み
    #         vfe (float): 収束後の変分自由エネルギーの重み
    #         iters (int): 最適化反復回数

    #     Returns:
    #         best_cost (float): 最適コスト値
    #         best_particle (np.ndarray): 最適パラメータ
    #         best_qs (np.ndarray): 最適軌道
    #     """
    #     dimensions = (self.num_knots - 2) * self.model.nq
    #     options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
    #     optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=dimensions, options=options)

    #     def objective(particles, model, agent, env, data, start, end, limits, time_steps, dt, num_knots,
    #                 jerk_weight, energy_weight, torque_change_weight, vfe_weight, kld_weight, bs_weight, un_weight, vfe_var_weight):
    #         costs = []
    #         for particle in particles:
    #             qs = qs_from_particle(particle, model=model, time_steps=time_steps,
    #                                 start=start, end=end, limits=limits, num_knots=num_knots)
                
    #             total_cost = 0.0

    #             if jerk_weight != 0.0:
    #                 jerk_cost = compute_total_jerk(qs, dt)
    #                 total_cost += jerk_weight * jerk_cost

    #             if energy_weight != 0.0:
    #                 energy_cost = compute_total_energy(model, data, qs, dt)
    #                 total_cost += energy_weight * energy_cost

    #             if torque_change_weight != 0.0:
    #                 torque_change_cost = compute_total_torque_change(model, data, qs, dt)
    #                 total_cost += torque_change_weight * torque_change_cost
                
    #             if vfe != 0.0 or kld != 0.0 or bs != 0.0 or un != 0.0:
    #                 vfes, klds, bss, uns = compute_total_vfe(agent, env, qs, dt)
    #                 total_vfe = sum(vfes)
    #                 total_kld = sum(klds)
    #                 total_bs = sum(bss)
    #                 total_un = sum(uns)
    #                 total_vfe_var = np.var(vfes)
    #                 total_cost += vfe_weight * total_vfe + kld_weight * total_kld + bs_weight * total_bs + un_weight * total_un + vfe_var_weight*total_vfe_var

    #             costs.append(total_cost)

    #         return np.array(costs)

    #     obj_func = partial(objective,
    #                     model=self.model,
    #                     agent=self.agent,
    #                     env=self.env,
    #                     data=self.data,
    #                     start=self.start,
    #                     end=self.end,
    #                     limits=self.limits,
    #                     time_steps=self.timesteps,
    #                     dt=self.dt,
    #                     num_knots=self.num_knots,
    #                     jerk_weight=jerk,
    #                     energy_weight=energy,
    #                     torque_change_weight=torque_change,
    #                     vfe_weight=vfe,
    #                     kld_weight=kld,
    #                     bs_weight=bs,
    #                     un_weight=un,
    #                     vfe_var_weight=vfe_var
    #                     )

    #     self.best_cost, self.best_particle = optimizer.optimize(obj_func, iters=iters)
    #     self.best_qs = qs_from_particle(self.best_particle, self.model, self.timesteps, self.start,
    #                                     self.end, self.limits, num_knots=self.num_knots)

    #     return self.best_cost, self.best_particle, self.best_qs
    

    def optimize(self, targets=None, jerk=0.0, energy=0.0, torque_change=0.0, vfe=0.0, kld=0.0, bs=0.0, un=0.0, vfe_var=0.0, iters=1, compensate_grav=True, end_penalty=0.0, temporal_approach_cost=0.0, MIN_RANGE_DENOMINATOR=1e-6):
        """
    粒子群最適化（PSO: Particle Swarm Optimization）を用いて、
    ロボットアームの最適軌道を求める関数。

    各粒子は中間ノット点（姿勢）の集合を表し、目標コスト関数を最小化するよう探索を行う。
    コスト関数は、運動の滑らかさやエネルギー消費、トルク変化、
    さらには自由エネルギー原理に基づく指標（VFE, KLD, BSなど）を組み合わせて構成できる。

    Parameters
    ----------
    targets = {
    "ig": {"target": ig_target_value, "min": ig_min_range, "max": ig_max_range},
    "energy": {"target": energy_target_value, "min": energy_min_range, "max": energy_max_range},
    # ... 他の指標
}
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
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=dimensions, options=options)

        # # JAX版計算関数を生成
        # compute_jerk_fn = make_compute_total_jerk_jax(self.dt)
        # compute_energy_fn = make_compute_total_energy_jax(self.model, self.data, self.dt)
        # compute_torque_change_fn = make_compute_total_torque_change_jax(self.model, self.data, self.dt) pinocchioがc++使ってるからだめ


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

            data = self.model.createData()
            total_cost = 0.0

            if jerk != 0.0:
                # total_cost += jerk * compute_jerk_fn(qs)
                total_cost += jerk * compute_total_jerk(dddqs, self.dt)

            if energy != 0.0:
                # total_cost += energy * compute_energy_fn(qs)
                total_cost += energy * compute_total_energy(self.model, data, qs, dqs, ddqs, self.dt, compensate_grav=compensate_grav)

            if torque_change != 0.0:
                total_cost += torque_change * compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt)

            if vfe != 0.0 or kld != 0.0 or bs != 0.0 or un != 0.0:
                vfes, klds, bss, uns = compute_total_vfe(self.agent, self.env, qs, self.dt, self.const_beliefs)
                total_vfe = sum(vfes)/len(qs)
                total_kld = sum(klds)/len(qs)
                total_bs = sum(bss)/len(qs)
                total_un = sum(uns)/len(qs)
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

            # targetの値に収束させるための目的関数
            if targets:
                TARGET_METRICS = targets
                target_label_candidates = [
                    "jerk", 
                    "energy", 
                    "torque_change", 
                    "vfe", 
                    "kld", 
                    "bs", 
                    "un", 
                    "vfe_var"
                ]
                vfe_elements = [
                    "vfe", 
                    "kld", 
                    "bs", 
                    "un", 
                    "vfe_var"
                ]
                
                # 共通部分を計算し、共通要素が存在するかを確認
                # 共通部分の要素数が 0 より大きければ True
                # if bool(target_keys_set.intersection(vfe_set)):

                # # targets辞書のキーの集合を取得
                # target_keys_set = set(targets.keys())
                
                # # vfe_elements_listの集合を取得
                # vfe_set = set(vfe_elements)
                
                vfes, klds, bss, uns = compute_total_vfe(self.agent, self.env, qs, self.dt, self.const_beliefs)
                total_vfe = sum(vfes)/len(qs)
                total_kld = sum(klds)/len(qs)
                total_bs = sum(bss)/len(qs)
                total_un = sum(uns)/len(qs)
                total_vfe_var = np.var(vfes)

                current_values = {
                "jerk": compute_total_jerk(dddqs, self.dt), 
                "energy": compute_total_energy(self.model, data, qs, dqs, ddqs,self.dt, compensate_grav=compensate_grav), 
                "torque_change": compute_total_torque_change(self.model, data, qs, dqs, ddqs, self.dt), 
                "vfe": total_vfe, 
                "kld": total_kld, 
                "bs": total_bs, 
                "un": total_un, 
                "vfe_var": total_vfe_var
                }

                
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
                    
                    if metric_range <= MIN_RANGE_DENOMINATOR:
                        continue
                        
                    # 4. 正規化された偏差の計算
                    # NormalizedDeviance = |current - target| / Range
                    normalized_deviation = np.abs(current_val - target_val) / metric_range
                    
                    # 5. 重み付きコストの加算
                    weighted_cost = normalized_deviation
                    total_cost += weighted_cost

            return total_cost

        # vmapで全粒子のコスト並列計算
        batched_objective = lambda particles: jnp.array([particle_objective(p) for p in particles])

        # オプティマイザ実行
        self.best_cost, self.best_particle = optimizer.optimize(batched_objective, iters=iters)
        self.best_qs = qs_from_particle(
            self.best_particle, 
            self.model, 
            self.timesteps, 
            self.start,
            self.end, 
            self.limits, 
            num_knots=self.num_knots,
            free_end_knot=free_end_knot
            )

        return self.best_cost, self.best_particle, self.best_qs

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
def compute_total_energy(model, data, qs, dqs, ddqs, dt, compensate_grav=True):
    total_energy = 0.0
    time_steps = len(qs)
    # # jnpは使うとダメ
    # dqs = np.gradient(qs, dt, axis=0)
    # ddqs = np.gradient(dqs, dt, axis=0)

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

# def make_compute_total_energy_jax(model, data, dt):
#     @jit
#     def compute(qs):
#         total_energy = 0.0
#         dqs = jnp.gradient(qs, dt, axis=0)
#         ddqs = jnp.gradient(dqs, dt, axis=0)

#         for q, dq, ddq in zip(qs, dqs, ddqs):
#             # 全体のトルク
#             pin.computeAllTerms(model, data, q, dq)
#             tau_total = pin.rnea(model, data, q, dq, ddq)
#             # 重力のみのトルク
#             tau_gravity = pin.computeGeneralizedGravity(model, data, q)
#             # 動的成分のみ取り出す
#             tau = tau_total - tau_gravity
#             energy = jnp.sum(jnp.abs(tau * dq)) * dt
#             total_energy += energy
#             return total_energy
#     return compute

# 動力学トルクの変化
def compute_total_torque_change(model, data, qs, dqs, ddqs, dt):
    taus = []
    # dqs = np.gradient(qs, dt, axis=0)
    # ddqs = np.gradient(dqs, dt, axis=0)

    for q, dq, ddq in zip(qs, dqs, ddqs):
        pin.computeAllTerms(model, data, q, dq)
        tau_total = pin.rnea(model, data, q, dq, ddq)
        tau_gravity = pin.computeGeneralizedGravity(model, data, q)
        tau_dynamic = tau_total - tau_gravity
        taus.append(tau_dynamic)

    taus = np.array(taus)
    dtaus = np.gradient(taus, dt, axis=0)
    # ddtaus = np.gradient(dtaus, dt, axis=0)
    torque_change_cost = np.sum(dtaus**2) * dt

    return torque_change_cost

# def make_compute_total_torque_change_jax(model, data, dt):
#     @jit
#     def compute(qs):
#         taus = []
#         for q in qs:
#             dq = jnp.zeros_like(q)
#             pin.computeAllTerms(model, data, q, dq)
#             tau_total = pin.rnea(model, data, q, dq, jnp.zeros_like(dq))
#             tau_gravity = pin.computeGeneralizedGravity(model, data, q)
#             taus.append(tau_total - tau_gravity)
#         taus = jnp.stack(taus)
#         dtau = jnp.diff(taus, n=1, axis=0) / dt
#         ddtau = jnp.diff(dtau, n=1, axis=0) / dt
#         return jnp.sum(ddtau**2) * dt
#     return compute


