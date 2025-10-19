import pinocchio as pin

# モデル構築関数

def generate_simple_arm_urdf(dof:int, filepath=None):
    assert dof > 0, f"無効な自由度：{dof}"

    if filepath is None:
        save_dir = "./robots"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"simple_arm_dof{dof}.urdf")

    urdf_str = f'<robot name="simple_arm_dof{dof}">\n  <link name="universe"/>\n\n'

    # リンクとジョイントの定義
    for i in range(1, dof+1):
        parent_link = "universe" if i == 1 else f"link{i-1}"
        child_link = f"link{i}"

        # ジョイント
        urdf_str += f'''  <joint name="joint{i}" type="revolute">
    <parent link="{parent_link}"/>
    <child link="{child_link}"/>
    <origin xyz="0 0 {1.0 if i>1 else 0}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="10.0" lower="-3.14" upper="3.14" velocity="5.0"/>
  </joint>\n'''

        # リンク（inertial + visual）
        urdf_str += f'''  <link name="{child_link}">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="1.0" radius="0.05"/>
      </geometry>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
    </visual>
  </link>\n'''

    # エンドエフェクタ
    urdf_str += '''  <joint name="ee_tip" type="fixed">
    <parent link="link{0}"/>
    <child link="ee_link"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
  </joint>\n'''.format(dof)

    urdf_str += '''  <link name="ee_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.02"/>
      </geometry>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
    </visual>
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>\n'''

    urdf_str += '</robot>\n'

    with open(filepath, "w") as f:
        f.write(urdf_str)

    print(f"URDFファイルを {filepath} に保存しました。自由度: {dof}")
    return filepath

# 古いurdf生成関数（残しておく）
def old_generate_simple_arm_urdf(dof:int, filepath=None):

    assert dof > 0, f"無効な自由度：{dof}"
    
    if filepath is None:
        save_dir = "./robots"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"simple_arm_dof{dof}.urdf")
        filepath = f"./robots/simple_arm_dof{dof}.urdf"

    urdf_str = f'<robot name="simple_arm_dof{dof}">\n\n  <link name="universe"/>\n\n'

    # 各ジョイントとリンクの定義
    if dof != 1:
        for i in range(1, dof):
            parent_link = "universe" if i == 1 else f"link{i-1}"
            child_link = f"link{i}"

            urdf_str += f'''  <joint name="joint{i}" type="revolute">
        <parent link="{parent_link}"/>
        <child link="{child_link}"/>
        <origin xyz="0 0 {0 if i==1 else 1.0}" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit effort="10.0" lower="-3.14" upper="3.14" velocity="5.0"/>
    </joint>\n\n'''

            urdf_str += f'''  <link name="{child_link}">
        <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 0.5"/>
        <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>\n\n'''
            
    # dof = 1 のときエンドエフェクタのみ
    parent_link = f"link{dof-1}"
    if dof == 1:
        parent_link = "universe"

    # エンドエフェクタリンクとジョイント
    urdf_str += f'''  <joint name="joint{dof}" type="revolute">
    <parent link="{parent_link}"/>
    <child link="ee_link"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="10.0" lower="-3.14" upper="3.14" velocity="5.0"/>
</joint>\n\n'''

    urdf_str += '''  <link name="ee_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.02"/>
      </geometry>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
    </visual>
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>\n\n'''

    # エンドエフェクタの固定ジョイント
    urdf_str += '''  <joint name="ee_tip" type="fixed">
    <parent link="ee_link"/>
    <child link="dummy_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>\n\n'''

    urdf_str += '  <link name="dummy_link"/>\n\n</robot>\n'

    # ファイルへ書き込み
    with open(filepath, "w") as f:
        f.write(urdf_str)

    print(f"URDFファイルを {filepath} に保存しました。自由度: {dof}")
    return filepath

def get_simple_arm(dof=2):
    urdf = generate_simple_arm_urdf(dof=dof)
    return pin.buildModelFromUrdf(generate_simple_arm_urdf(dof=dof))



# 可視化関連
import os
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

import matplotlib.pyplot as plt

def create_ax_with_scope(scope, base_height=5, equal_aspect=True):
    """
    scope に基づいて適切なサイズの Figure/Axes を生成する。

    Parameters
    ----------
    scope : dict
        {'x': (xmin, xmax), 'y': (ymin, ymax)} の形式で指定。
    base_height : float, optional
        図全体の高さ（inch単位）。幅はscopeの比に合わせて自動計算。
    equal_aspect : bool, optional
        True の場合、スケールを等しくする（デフォルト: True）。
    show_grid : bool, optional
        True の場合、グリッド線を表示（デフォルト: False）。

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    # 範囲と比率を計算
    x_range = scope['x'][1] - scope['x'][0]
    y_range = scope['y'][1] - scope['y'][0]
    aspect_ratio = x_range / y_range

    # 図の生成（サイズをscope比に合わせる）
    fig_width = base_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, base_height))

    # 軸設定
    ax.set_xlim(scope['x'])
    ax.set_ylim(scope['y'])
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')

    return fig, ax

def plot_robot_2D(model, q, ax=None, show=True, detail=False, grid=False, scope=None,
                  title='Robot Arm 2D', plane='xz', max_range=None,
                  draw_base=True, draw_gripper=True,
                  base_size=(1.0, 2.0), gripper_size=0.2):
    """
    Pinocchioモデルの現在姿勢をmatplotで2Dプロットする関数
    """
    # 描画のパラメータ
    link_lw = 5  # ロボットリンクの線幅
    link_color = 'black'  # ロボットリンクの色
    joint_size = 10  # ジョイントの点の大きさ
    joint_face_color = 'white'  # ジョイントの面の色
    joint_edge_color = 'black'  # ジョイントの縁の色
    ee_size = 30  # エンドエフェクタの点の大きさ
    ee_color = 'red'  # エンドエフェクタの色
    ee_lw = 5  # エンドエフェクタの線幅
    base_size = base_size  # 土台のサイズ (幅, 高さ)
    base_color = 'gray'  # 土台の色
    base_alpha = 1.0  # 土台の透明度
    gripper_size = gripper_size  # グリッパのサイズ
    gripper_color = 'red'  # グリッパの色
    gripper_lw = 3  # グリッパの線幅


    # 順運動学を計算
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)

    # プロット用意
    if ax is None:
        fig, ax = create_ax_with_scope(scope)

    # 各ジョイントの位置を取得
    positions = [np.array([0.0, 0.0])]  # ワールド原点

    # 平面選択
    if plane in ('xy', 'yx'):
        axis_idx = (0, 1)
    elif plane in ('yz', 'zy'):
        axis_idx = (1, 2)
    elif plane in ('zx', 'xz'):
        axis_idx = (0, 2)
    else:
        raise ValueError(f"Unknown plane: {plane}")
    axis_idx = np.array(axis_idx)

    for i in range(1, model.njoints):
        pos = data.oMi[i].translation
        pos = pos[axis_idx]
        positions.append(pos)
        if detail:
            ax.text(pos[0],pos[1], model.names[i])
    positions = np.array(positions)

  
    # --- 土台を描画 ---
    if draw_base:
        base_w, base_h = base_size
        base = plt.Rectangle((-base_w/2, -base_h), base_w, base_h, color=base_color, alpha=base_alpha)
        ax.add_patch(base)

    # --- グリッパを描画 ---
    for frame_id, frame in enumerate(model.frames):
        if frame.name == "ee_tip":
            ee_pos = data.oMf[frame_id].translation[axis_idx]
            parent_id = frame.parentJoint
            parent_pos = data.oMi[parent_id].translation[axis_idx]
            xs, ys = zip(parent_pos, ee_pos)
            

            if draw_gripper:
                # エンドエフェクタの向き
                v = ee_pos - parent_pos
                v = v / np.linalg.norm(v)
                n = np.array([-v[1], v[0]])# 法線ベクトル

                # グリッパのパラメータ設定
                base_len = gripper_size * 0.5   # 爪の根元部分の長さ
                tip_len = gripper_size * 0.5    # 折れた先の部分の長さ
                bend_angle = np.deg2rad(60)     # 折れる角度 [deg]
                open_angle = np.deg2rad(60)     # 爪の根元開き角度 [deg]

                def R(angle):
                    return np.array([
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]
                    ])

                # 爪の描画
                for side in [+1, -1]:
                    # sideごとの方向を決定
                    open_dir = R(open_angle*side) @ v
                    bend_dir = R(bend_angle*(-side)) @ open_dir
                    
                    # 爪の根元線
                    base_start = ee_pos
                    base_end = base_start + open_dir * base_len

                    # 爪の先端線（折れ部分）
                    tip_end = base_end + bend_dir * tip_len

                    gripper_color = 'red'
                    gripper_lw = 3

                    # 描画
                    ax.plot([base_start[0], base_end[0]], [base_start[1], base_end[1]], 
                            c=gripper_color, lw=gripper_lw)
                    ax.plot([base_end[0], tip_end[0]], [base_end[1], tip_end[1]], 
                            c=gripper_color, lw=gripper_lw)
                if detail:
                    ax.text(pos[0], pos[1], frame.name)

            # エンドエフェクタのリンクを描画
            ax.plot(xs, ys, c=link_color, linewidth=ee_lw)
            ax.scatter(ee_pos[0], ee_pos[1], c=ee_color, s=ee_size, marker='o')

    
    # --- ロボットのリンクとジョイントを描画 ---
    ax.plot(positions[:, 0], positions[:, 1], 'o-', linewidth=link_lw, c=link_color,
            markersize=joint_size, markerfacecolor=joint_face_color, markeredgecolor=joint_edge_color)
    
    # 軸設定
    # 範囲を計算
    if scope is None:
        if max_range is None:
            total_length = np.sum([np.linalg.norm(data.oMi[i].translation - data.oMi[i-1].translation)
                                for i in range(1, model.njoints)]) + 1.0
        else:
            total_length = max_range*1.3
        xlim = (-total_length, total_length)
        ylim = (-total_length, total_length)
    else:
        xlim, ylim = scope['x'], scope['y']

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(grid)
    if detail:
        ax.set_title(title)
        ax.set_xlabel('X' if axis_idx[0]==0 else 'Y')
        ax.set_ylabel('Y' if axis_idx[1]==1 else 'Z')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if show:
        plt.show()

    return ax


# ロボットのスケルトンを描画する関数(3D)
def plot_robot_3D(model, q, ax=None, show=True, detail=False, title='Robot Arm 3D', max_range=None):
    # データオブジェクト生成
    data = model.createData()
    
    # 順運動学計算
    pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)

    if ax is None:
        # 3Dプロットセットアップ
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


    # ジョイントの位置記録用
    joint_positions = []

    for joint_id in range(1, model.njoints):
        oMi = data.oMi[joint_id]  # ワールド座標系での姿勢
        pos = oMi.translation
        joint_positions.append(pos)
        # 親ジョイントとの接続線を引く
        parent_id = model.parents[joint_id]
        if parent_id > 0:  # universe以外
            parent_pos = data.oMi[parent_id].translation
            xs, ys, zs = zip(parent_pos, pos)
            ax.plot(xs, ys, zs, c='b', linewidth=5)

        # ジョイントの位置に点を打つ
        ax.scatter(pos[0], pos[1], pos[2], c='gold', s=50, marker='o')
        if detail:
            ax.text(pos[0], pos[1], pos[2], model.names[joint_id])
      
    # エンドエフェクタも描画
    for frame_id, frame in enumerate(model.frames):
        if frame.name=="ee_tip":
            parent_id = frame.parentJoint
            pos = data.oMf[frame_id].translation
            parent_pos = data.oMi[parent_id].translation
            xs, ys, zs = zip(parent_pos, pos)
            ax.plot(xs, ys, zs, c='r',linewidth=3)
            ax.scatter(pos[0], pos[1], pos[2], c='r', s=30, marker='o')
            if detail:
                ax.text(pos[0], pos[1], pos[2], frame.name)

    # 描画範囲設定
    if max_range is not None:
        max_range *= 0.7
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

    # グラフ調整
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot Skeleton')
    ax.set_box_aspect([1,1,1])  # アスペクト比固定
    if not detail:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])  # 3Dなら

    if show:
        plt.show()
    return ax

# 呼び出し例（モデルと状態ベクトルqが必要）
# plot_robot_skeleton(model, q)

def plot_robot_motion(model, qs, dt=0.1, movie=None, savedir='./tmp_movies/', grid=False,
                      title='', is3d=False, detail=False, plane='xz', scope=None,
                      draw_base=True, draw_gripper=True,
                      base_size=(0.5, 0.1), gripper_size=0.2):
    """
    ロボットの動きをmatplotでアニメーション表示・保存する関数

    Parameters
    ----------
    model : pinocchio.Model
        ロボットモデル
    data : pinocchio.Data
        ロボットデータ
    qs : ndarray (T, n)
        各時刻の関節角度列
    dt : float
        シミュレーションの時間刻み [s]
        0.1  = 10fps(default)
        0.05 = 20fps
        0.033= 30fps
    movie : str or None
        保存するファイル名。拡張子付き。NoneならJupyter内で表示。
    """
    # animationの時間幅に変換
    interval = dt * 1000 # dt(sec) → interval(msec)
    # 軌跡の長さ
    time_steps = qs.shape[0]
    # fpsを計算
    fps = 1/dt
    # 長さを計算
    duration = time_steps * dt

    # robotの最大リーチを計算
    max_range = 0.0

    for joint_id in range(1, model.njoints):
        # この関節の親座標系からの位置ベクトル（トランスレーション）
        length_vec = model.jointPlacements[joint_id].translation
        # 選択した軸の長さ成分のノルム（もしくは単純な距離でもOK）
        link_length = np.linalg.norm(length_vec)
        max_range += link_length

    for frame_id, frame in enumerate(model.frames):
        if frame.name=="ee_tip":
            length_vec = model.frames[frame_id].placement.translation
            link_length = np.linalg.norm(length_vec)
            max_range += link_length

    # プロットセットアップ
    if is3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(6,6))

    # アニメーション更新関数
    def update(frame):
        ax.cla()
        q = qs[frame]
        time = frame * dt
        if is3d:
            plot_robot_3D(model, q, ax=ax, show=False, detail=detail, max_range=max_range)
            # 軸範囲の取得
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            # 軸範囲の右上あたりに表示
            ax.text(
                xlim[1]*0.7, ylim[1]*0.9, zlim[1]*0.9,  # 表示位置
                f"Time: {time:.2f} s / {duration:.2f} s\nFrame: {frame} / {time_steps}\nFPS: {fps:.1f}",  
                ha='left', va='top',
                fontsize=7, color='black',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.8)
                )
        else:
            plot_robot_2D(model, q, ax=ax, show=False, detail=detail, grid=grid, scope=scope,
                          title=title, plane=plane, max_range=max_range,
                          draw_base=draw_base, draw_gripper=draw_gripper,
                          base_size=base_size, gripper_size=gripper_size)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if detail:
                ax.text(
                    xlim[1]*0.7, ylim[1]*0.9, 
                    f"Time: {time:.2f} s / {duration:.2f} s\nFrame: {frame} / {time_steps}\nFPS: {fps:.1f}", 
                    ha='left', va='top',
                    fontsize=7, color='black',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.8)
                    )
        return ax,

    ani = animation.FuncAnimation(
        fig, update, frames=time_steps, interval=interval, blit=False
    )

    if movie is None:
        print("Now preparing html without saving")
        display(HTML(ani.to_jshtml()))
    
    if movie is not None:
        #movieの拡張子を判別してそれにあった処理をする
        # movieの拡張子を取得
        base, ext = os.path.splitext(movie)
        os.makedirs(savedir, exist_ok=True)

        if ext == '':
            # 拡張子がなければ .gif を付けて保存
            ext = '.gif'
            writer='pillow'
        elif ext.lower() in ['.gif']:
            writer='pillow'
        elif ext.lower() in ['.mp4', '.mov', '.avi']:
            writer='ffmpeg'
        else:
            # それ以外なら test_motion.gif にする
            ext = '.gif'
            movie = 'test_motion'
            writer = 'pillow'
            print("Unsupported extension — saving as 'test_motion.gif'...")
        # パスを構成
        filepath = savedir + movie + ext
        print(f"saving as '{filepath}'...")
        ani.save(filepath, writer=writer, fps=fps)
        print(f"saved as {movie}")
    return




# スプライン関連
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
import pinocchio as pin
import pyswarms as ps

# qs_flat(T * nq)をqs_traj(T, nq)の形状を持つ軌跡データに成形
def reshape_qs_from_flat(qs_flat, time_steps, nq):
    assert qs_flat.shape[0] == time_steps * nq, "qs_flat must have shape (T * nq,)"
    return qs_flat.reshape((time_steps, nq))

# 0~1のランダムな値をモデルの関節の可動域に拡大
def expand_random_to_joint_limits(qs_norm, model, limits=None):
    nq = model.nq
    qs_expanded = np.zeros_like(qs_norm)
    if limits is None:
        lowers, uppers = model.lowerPositionLimit, model.upperPositionLimit
    else:
        lowers, uppers = limits[0], limits[1]
    for i in range(nq):
        lower, upper = lowers[i], uppers[i]
        qs_expanded[:, i] = lower + (upper - lower) * qs_norm[:, i]
    return qs_expanded

def random_q(model):
    nq = model.nq
    qs = np.random.rand(nq)
    return qs

# 毎ステップでランダムな関節角度を生成（使わない）
def random_qs(model, time_steps, limits=None, seed=0):
    np.random.seed(seed)
    nq = model.nq
    qs_flat = np.random.rand(time_steps * nq)
    qs_norm = reshape_qs_from_flat(qs_flat, time_steps, nq)
    return expand_random_to_joint_limits(qs_norm, model, limits=limits)

# 全関節で共通の可動域
def same_limits(nq, lower=-np.pi, upper=np.pi):
    lowers = [lower for _ in range(nq)]
    uppers = [upper for _ in range(nq)]
    limits = [lowers, uppers]
    return np.array(limits)

def random_qs_spline(model, time_steps, start, end, limits=None, num_knots=5, seed=0):
    # 始点と終点の形状を確認
    q_shape = pin.neutral(model).shape
    assert start.shape == q_shape, f"start.shape {start.shape} and end.shape {end.shape} must match model's q_shape {q_shape}"
    assert end.shape == q_shape, f"start.shape {start.shape} and end.shape {end.shape} must match model's q_shape {q_shape}"

    # ランダムシード初期化
    np.random.seed(seed)

    nq = model.nq

    # 可動域の設定
    if limits is None:
        lowers = model.lowerPositionLimit
        uppers = model.upperPositionLimit
    else:
        if len(limits) == 1:
            limits = same_limits(nq, lower=limits[0], upper=limits[1])
        lowers, uppers = limits
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    # 始点と終点の範囲を確認
    # startのすべての関節角度に対して、それぞれが可動域に収まっていなければassertする。
    # endも同様

    # スプラインの制御点の時間軸
    control_times = np.linspace(0, time_steps-1, num=num_knots)
    times = np.arange(time_steps)

    # 関節ごとの軌跡を格納する配列
    qs = np.zeros((time_steps, nq))
    
    for j in range(nq):
        # 始点と終点を固定
        control_values = np.zeros(num_knots)
        control_values[0] = start[j]
        control_values[-1] = end[j]

        # 中間の制御点をランダムに生成
        control_values[1:-1] = np.random.uniform(lowers[j], uppers[j], num_knots-2)

        # CubicSpline補完
        # cs = PchipInterpolator(control_times, control_values)
        cs = CubicSpline(control_times, control_values, bc_type='clamped')

        # 時系列データ生成
        q_traj = cs(times)
        qs[:, j] = q_traj
    return qs

def qs_from_particle(particle, model, time_steps, start, end, limits, num_knots=5):
    """
    len(particle) = (num_knots - 2) * nq
    """
    # 始点と終点の形状を確認
    q_shape = pin.neutral(model).shape
    assert start.shape == q_shape, f"start.shape {start.shape} and end.shape {end.shape} must match model's q_shape {q_shape}"
    assert end.shape == q_shape, f"start.shape {start.shape} and end.shape {end.shape} must match model's q_shape {q_shape}"

    nq = model.nq

    # 可動域の設定
    if len(limits) == 1:
        limits = same_limits(nq, lower=limits[0], upper=limits[1])
    lowers, uppers = limits
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    # 始点と終点の範囲を確認
    # startのすべての関節角度に対して、それぞれが可動域に収まっていなければassertする。
    # endも同様

    # スプラインの制御点の時間軸
    control_times = np.linspace(0, time_steps-1, num=num_knots)
    times = np.arange(time_steps)

    # 関節ごとの軌跡を格納する配列
    qs = np.zeros((time_steps, nq))
    
    for j in range(nq):
        # 始点と終点を固定
        control_values = np.zeros(num_knots)
        control_values[0] = start[j]
        control_values[-1] = end[j]

        # particleを中間の制御点に設定
        control_values[1:-1] = particle[(num_knots-2)*j:(num_knots-2)*(j+1)]

        # CubicSpline補完
        cs = PchipInterpolator(control_times, control_values)
        # cs = CubicSpline(control_times, control_values, bc_type='clamped')

        # 時系列データ生成
        q_traj = cs(times)
        qs[:, j] = q_traj
    return qs


# 特定のフレームの位置から全体の姿勢を求める関数
def ik_se3_solver(model, data, frame_id, target_placement):
    # 反復計算用パラメータ
    max_iter = 100
    eps = 1e-6
    alpha = 0.5  # ステップサイズ

    # 初期姿勢
    q = pin.neutral(model)

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_placement = data.oMf[frame_id]

        # 誤差を計算 (SE3.log)
        err = pin.log(target_placement.inverse() * current_placement).vector

        if np.linalg.norm(err) < eps:
            print(f"収束しました！ iteration: {i}")
            break

        # ヤコビアン計算
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)

        # 関節変位の計算 (擬似逆行列で解く)
        v = -alpha * np.linalg.pinv(J) @ err

        # 姿勢の更新
        q = pin.integrate(model, q, v)
    
    return q


# # 最適化関連
# from robot import *
# from perception import *
# import numpy as np
# from scipy.interpolate import PchipInterpolator, CubicSpline
# import pinocchio as pin
# import pyswarms as ps
# from functools import partial

# class Optimizer:
#     def __init__(
#             self,
#             model,
#             timesteps,
#             dt,
#             start,
#             end,
#             limits=None,
#             num_knots=5,
#             n_particles=30):
        
#         # モデルと軌跡の端点を初期化
#         self.model = model
#         self.nq = model.nq
#         self.data = self.model.createData()
#         self.timesteps = timesteps
#         self.dt = dt
#         self.start = start
#         self.end = end
#         # 可動域の設定
#         self.limits = [model.lowerPositionLimit, model.upperPositionLimit]
#         if limits is not None:
#             self.limits = limits
#         # スプライン制御点と粒子の数
#         self.num_knots = num_knots
#         self.n_particles = n_particles

#         # 結果を格納する変数の初期化
#         self.best_cost = None
#         self.best_particle = None
#         self.best_qs = None


#     def optimize_torque_change(self):
#         dimensions = (self.num_knots - 2) * self.model.nq
#         options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
#         optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=dimensions, options=options)

#         def objective(particles, model, data, start, end, limits, time_steps, dt, num_knots):
#             # particles.shape = (n_particles, dimensions)
#             costs = []
#             for particle in particles:
#                 # particle.shape = ((num_knots - 2)*nq,)
#                 # フラットなベクトルを(time_steps, nq)に変換
#                 qs = qs_from_particle(particle,
#                                     model=model,
#                                     time_steps=time_steps,
#                                     start=start,
#                                     end=end,
#                                     limits=limits,
#                                     num_knots=num_knots)
#                 cost = compute_total_torque_change(model, data, qs, dt)
#                 costs.append(cost)
#             return np.array(costs)

#         # partialでほかの引数をバインド
#         obj_func = partial(objective,
#                         model=self.model,
#                         data=self.data,
#                         start=self.start,
#                         end=self.end,
#                         limits=self.limits,
#                         time_steps=self.timesteps,
#                         dt=self.dt,
#                         num_knots=self.num_knots)
        
#         self.best_cost, self.best_particle = optimizer.optimize(obj_func, iters=100)
#         self.best_qs = qs_from_particle(self.best_particle, self.model, self.timesteps, self.start, self.end, self.limits, num_knots=self.num_knots)
        
#         return self.best_cost, self.best_particle, self.best_qs
    

    
#     def optimize(self, jerk=0.0, energy=0.0, torque_change=0.0, iters=100):
#         """
#         複数のコスト指標を重み付け線形和として最適化する。

#         Parameters:
#             jerk (float): 見かけ上のジャークの重み
#             energy (float): トルクエネルギー消費の重み
#             torque_change (float): トルク変化量の重み
#             iters (int): 最適化反復回数

#         Returns:
#             best_cost (float): 最適コスト値
#             best_particle (np.ndarray): 最適パラメータ
#             best_qs (np.ndarray): 最適軌道
#         """
#         dimensions = (self.num_knots - 2) * self.model.nq
#         options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
#         optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=dimensions, options=options)

#         def objective(particles, model, data, start, end, limits, time_steps, dt, num_knots,
#                     jerk_weight, energy_weight, torque_change_weight):
#             costs = []
#             for particle in particles:
#                 qs = qs_from_particle(particle, model=model, time_steps=time_steps,
#                                     start=start, end=end, limits=limits, num_knots=num_knots)
                
#                 total_cost = 0.0

#                 if jerk_weight != 0.0:
#                     jerk_cost = compute_total_jerk(qs, dt)
#                     total_cost += jerk_weight * jerk_cost

#                 if energy_weight != 0.0:
#                     energy_cost = compute_total_energy(model, data, qs, dt)
#                     total_cost += energy_weight * energy_cost

#                 if torque_change_weight != 0.0:
#                     torque_change_cost = compute_total_torque_change(model, data, qs, dt)
#                     total_cost += torque_change_weight * torque_change_cost

#                 costs.append(total_cost)

#             return np.array(costs)

#         obj_func = partial(objective,
#                         model=self.model,
#                         data=self.data,
#                         start=self.start,
#                         end=self.end,
#                         limits=self.limits,
#                         time_steps=self.timesteps,
#                         dt=self.dt,
#                         num_knots=self.num_knots,
#                         jerk_weight=jerk,
#                         energy_weight=energy,
#                         torque_change_weight=torque_change)

#         self.best_cost, self.best_particle = optimizer.optimize(obj_func, iters=iters)
#         self.best_qs = qs_from_particle(self.best_particle, self.model, self.timesteps, self.start,
#                                         self.end, self.limits, num_knots=self.num_knots)

#         return self.best_cost, self.best_particle, self.best_qs


# # 軌跡からジャークを計算する関数
# def compute_total_jerk(qs, dt):
#     ddq = np.diff(qs, n=2, axis=0) / (dt**2)
#     dddq = np.diff(ddq, n=1, axis=0) / dt
#     jerk_cost = np.sum(dddq**2) * dt
#     return jerk_cost

# # 軌跡からエネルギーを計算する関数
# def compute_total_energy(model, data, qs, dt, gravity=True):
#     total_energy = 0.0
#     time_steps = len(qs)

#     for t in range(time_steps-1):
#         q_current = qs[t]
#         q_next = qs[t+1]
#         dq = (q_next - q_current) / dt

#         # 全体のトルク
#         pin.computeAllTerms(model, data, q_current, dq)
#         tau_total = pin.rnea(model, data, q_current, dq, np.zeros_like(dq))

#         if gravity:
#             # 重力のみのトルク
#             tau_gravity = pin.computeGeneralizedGravity(model, data, q_current)
#             # 動的成分のみ取り出す
#             tau = tau_total - tau_gravity
#         else:
#             tau = tau_total

#         energy = np.sum(np.abs(tau * dq)) * dt
#         total_energy += energy

#     return total_energy

# # 動力学トルクの変化(重力を除いたジャーク)
# def compute_total_torque_change(model, data, qs, dt):
#     taus = []

#     for q in qs:
#         dq = np.zeros_like(q)
#         pin.computeAllTerms(model, data, q, dq)
#         tau_total = pin.rnea(model, data, q, dq, np.zeros_like(dq))
#         tau_gravity = pin.computeGeneralizedGravity(model, data, q)
#         tau_dynamic = tau_total - tau_gravity
#         taus.append(tau_dynamic)

#     taus = np.array(taus)
#     dtau = np.diff(taus, n=1, axis=0) / dt
#     ddtau = np.diff(dtau, n=1, axis=0) / dt
#     torque_change_cost = np.sum(ddtau**2) * dt

#     return torque_change_cost



from PIL import Image
import os

def get_keyframes(filepath, num_samples, output_dir=None):
    """
    
    """
    # GIFファイルのパス
    gif_path = filepath
    gif_dir = os.path.dirname(gif_path)
    gif_name = os.path.splitext(os.path.basename(gif_path))[0]

     # 出力先フォルダのパスを作成
    output_dir = os.path.join(gif_dir, f"{gif_name}_keyframes")
    os.makedirs(output_dir, exist_ok=True)

    # GIFを開く
    img = Image.open(gif_path)

    # 全フレーム数の取得
    total_frames = img.n_frames
    print(f'全フレーム数: {total_frames}')

    # 等間隔のフレーム番号を計算（はじめと終わり含む）
    frame_indices = [round(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)]
    print(f'取り出すフレーム番号: {frame_indices}')

    # 指定したフレームを保存
    for i, frame_index in enumerate(frame_indices):
        img.seek(frame_index)  # 指定フレームへ
        frame = img.convert("RGB")  # RGB変換（必要なら）
        frame.save(os.path.join(output_dir, f'frame_{i}.png'))
    print('完了しました。')