# 結果の保存とかいろいろ
from .robot import *
from .perception import *
# from .optimize import *

import numpy as np
from pathlib import Path
from PIL import Image
import os

import os
from pathlib import Path
from PIL import Image
from typing import Optional, List

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union
import numpy as np
import pinocchio as pin

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union
import numpy as np
import pinocchio as pin

from dataclasses import dataclass
import numpy as np
import pickle
import pinocchio as pin

import json
import os
from pathlib import Path
from datetime import datetime


import pickle
from pathlib import Path
from typing import Union, Any, Optional

def load_object_from_pickle(
    path: Optional[Union[str, Path]] = None,
    folder_path: Optional[Union[str, Path]] = None,
    file_name: Optional[str] = None
) -> Any:
    """
    指定されたパスからpickleファイルを読み込み、格納されているオブジェクトを返します。
    
    Parameters
    ----------
    path : Optional[Union[str, Path]]
        pickleファイルへの完全なパス。
    folder_path : Optional[Union[str, Path]]
        pickleファイルが格納されているフォルダのパス。
    file_name : Optional[str]
        読み込むファイル名（例: 'data.pkl'）。
        
    Returns
    -------
    Any
        pickleファイルから読み込まれたオブジェクト。
        
    Raises
    ------
    ValueError
        必要なパス情報が不足している場合。
    FileNotFoundError
        指定されたファイルが見つからない場合。
    """
    
    # 1. パスの決定と結合
    if path:
        # 完全なパスが与えられた場合
        file_path = Path(path)
    elif folder_path and file_name:
        # フォルダパスとファイル名が与えられた場合、結合する
        file_path = Path(folder_path) / file_name
    else:
        # 必要な情報が不足している場合
        raise ValueError(
            "パスを指定するか、folder_pathとfile_nameの両方を指定してください。"
        )

    # 2. ファイルの存在確認
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
    # 3. pickleファイルの読み込み
    try:
        # 'rb' (read binary) モードでファイルを開く
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except pickle.UnpicklingError as e:
        print(f"⚠️ pickleファイルの読み込み中にエラーが発生しました（ファイル破損の可能性）: {e}")
        raise
    except Exception as e:
        print(f"⚠️ ファイル処理中に予期せぬエラーが発生しました: {e}")
        raise




@dataclass
class OptimizationResult:
    """最適化の結果と、それを再現・分析するための全てのパラメータを保持するクラス"""
    
    # ------------------ 1. ロボットモデル関連 (再現用) ------------------
    dof: int                           # 自由度
    robot_model_key: str               # ロボットモデルを特定するキー (例: 'simple_arm_dof3')
    robot_model: Any
    start: np.ndarray                  # 初期姿勢 (nq,)
    end: np.ndarray                    # 目標姿勢 (nq,)
    limits_low: np.ndarray             # 関節可動域 下限
    limits_high: np.ndarray            # 関節可動域 上限

    # ------------------ 2. シミュレーション/可視化パラメータ ------------------
    dt: float                          # 時間刻み幅 [s]
    time_steps: int                    # 総時間ステップ数 T
    fps: int                           # 可視化時のフレームレート (デフォルト値)
    compensate_grav: bool              # 重力補償の有無

    # ------------------ 3. 最適化プロセスパラメータ ------------------
    optimizer_type: str                # 使用したオプティマイザ ('PSO', 'Scipy_Spline'など)
    # num_knots: int                     # スプライン制御点の数
    # n_particles: int                   # 粒子の数 (PSOの場合)
    # max_iter: int                      # 最大イテレーション数
    # cost_weights: dict                 # コスト関数の重み (例: {'jerk': 1.0, 'energy': 0.1, 'vfe': 0.0, ...})
    # random_seed: Optional[int]         # 乱数シードを保持
    optimizer_params: dict
    cost_func: dict


    # ------------------ 4. Perception Agentパラメータ ------------------
    agent_params: Optional[dict]                 # Agentクラスの初期化パラメータを全て格納 (num_obs, eps, Avars, Bvars, num_iterなど),またはNone

    
    # ------------------ 5. 最適化結果 ------------------
    qs: np.ndarray                     # 最適化された関節角度軌跡 (T, nq)
    best_cost: float                   # 最終的な最小コスト
    final_metrics: dict                # 最終的な各指標の値
    best_particle: np.ndarray          # 最適化された中間ノット/制御点
    
    cost_history: Optional[list]       # コストのイテレーション履歴
    pos_history: Optional[dict]     # VFE, KLDなどの時系列データ履歴　例：
    optimization_summary: Dict[str, Union[float, int, str]]          # 最適化ステータス　例：{'ftol': ftol, 'status': result.status, 'message': result.message, 'nfev': result.nfev}

    # ------------------ 6. 保存/ロード/描画機能 ------------------
        
    def save(self, folder_path: str, file_name: str):
        """
        結果をファイルに保存する（推奨はpickle）。
        同時に、info()のサマリーをテキストファイルとして同フォルダに保存する。
        """
        import pickle
        import os
        from pathlib import Path

        filepath = os.path.join(folder_path, file_name)

        # 1. pickleファイルとして保存
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Error: Failed to save OptimizationResult object to {filepath}: {e}")
            return

        # 2. info()のサマリーをテキストファイルとして保存
        try:
            # info()を呼び出し、表示はせず文字列としてサマリーを取得
            summary_text = self.info(print_output=False) 
            
            # pickleファイルのパスから拡張子を .txt に変更したパスを作成
            save_path = Path(filepath)
            txt_filepath = save_path.with_suffix('.txt')
            
            # テキストファイルとして保存
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(summary_text)

            print(f"Success: Optimization result saved to {save_path.name} and summary to {txt_filepath.name}")

        except Exception as e:
            print(f"Warning: Failed to save info() summary to text file: {e}")

    @classmethod
    def load(cls, filepath: str) -> 'OptimizationResult':
        """ファイルから結果をロードする"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    # ------------------ 7. 情報表示機能 ------------------

    def __info(self, print_output: bool = True) -> str:
        """
        OptimizationResultの主要な情報と最適化結果を整形して表示/取得する。
        
        Parameters
        ----------
        print_output : bool, default=True
            Trueの場合、結果を標準出力に表示する。Falseの場合、文字列として返すのみ。
        
        Returns
        -------
        str
            整形された結果のサマリーテキスト。
        """
        
        # --- 既存のヘルパー関数（info()内に定義されていると仮定）のロジックを再利用 ---
        
        def format_array_stats(arr: Optional[np.ndarray], label: str) -> str:
            # ... (既存のロジック) ...
            if arr is None:
                return f"{label}: None"
            if arr.size == 0:
                return f"{label}: (Empty Array)"
            if arr.ndim == 1:
                return f"{label}: [Min: {arr.min():.4f}, Max: {arr.max():.4f}] (Shape: {arr.shape})"
            return f"{label}: [Min: {arr.min():.4f}, Max: {arr.max():.4f}] (Shape: {arr.shape})"

        def format_agent_params(params: Dict[str, Any]) -> str:
            # ... (既存のロジック) ...
            if not params:
                return " - Perception Agent: 未使用\n"
            
            lines = [
                f" - Perception Agent: 使用 ({params.get('num_iter', 1)} 反復)",
                f"   - DOF/Modality/Factors: {params.get('dof', '?')} / {len(params.get('num_obs', []))} / {params.get('num_factors', '?')}",
                f"   - Params (eps, hist): {params.get('eps', '?')} / {params.get('num_history', '?')}",
                f"   - Avars/Bvars (Len): {len(params.get('Avars_validated', []))} / {len(params.get('Bvars_validated', []))}"
            ]
            return '\n'.join(lines) + '\n'

        # --- 標準出力の代わりに文字列を構築 ---
        output_lines = []
        
        def safe_append(text):
            output_lines.append(text)

        # 既存の print() の内容を safe_append() に置き換える
        
        safe_append("=" * 60)
        safe_append("           <<< Optimization Result Summary >>>             ")
        safe_append("=" * 60)
        
        # 1. 最適化結果の概要
        safe_append("\n[Optimization Result]")
        safe_append(f"  Best Cost: {self.best_cost:.6f}")
        safe_append(f"  Optimizer Type: {self.optimizer_type} (Seed: {self.random_seed})")
        
        # 2. ロボットとタスク
        safe_append("\n[Robot & Task Definition]")
        safe_append(f"  Robot Model: {self.robot_model_key} (DOF: {self.dof})")
        safe_append(f"  Time: {self.time_steps} steps (fps: {self.fps} , dt: {self.dt:.4f} s, Total: {self.time_steps * self.dt:.2f} s)")
        safe_append(f"  Start Q: {self.start[:min(self.dof, 5)]}...")
        safe_append(f"  End Q: {self.end[:min(self.dof, 5)]}...")
        safe_append(f"  Limits Low/High (DOF {self.dof}): [Min: {self.limits_low.min():.2f}, Max: {self.limits_high.max():.2f}]")
        
        # 3. 最適化パラメータ
        safe_append("\n[Optimization Parameters]")
        safe_append(f"  Max Iterations: {self.max_iter}")
        safe_append(f"  Knot Points: {self.num_knots}")
        if self.n_particles > 0:
            safe_append(f"  Particles: {self.n_particles}")
            
        safe_append("  Cost Weights:")
        for key, value in self.cost_weights.items():
            if isinstance(value, float) and value != 0.0:
                safe_append(f"    - {key.ljust(15)}: {value:.4f}")
            elif isinstance(value, (int, bool)) and value:
                safe_append(f"    - {key.ljust(15)}: {value}")
        
        # 4. Perception Agent
        safe_append(format_agent_params(self.agent_params).strip()) # 末尾の改行を削除
        
        # 5. 最終的な指標 (Final Metrics)
        safe_append("\n[Final Computed Metrics]")
        
        for key, value in self.final_metrics.items():
            if isinstance(value, float):
                safe_append(f"  - {key.ljust(15)}: {value:.6f}")
            else:
                safe_append(f"  - {key.ljust(15)}: {value}")
        
        # 6. 格納データ統計
        safe_append("\n[Stored Data Statistics]")
        safe_append(format_array_stats(self.qs, "  Trajectory (qs)"))
        safe_append(format_array_stats(self.best_particle, "  Best Particle"))

        if self.cost_history is not None:
            safe_append(f"  Cost History: {len(self.cost_history)} points (Initial: {self.cost_history[0]:.4f})")
            
        if self.metric_history:
            safe_append(f"  Optimizer Info: {', '.join(self.metric_history.keys())}")
            if self.optimizer_type.startswith('Scipy'):
                safe_append(f"    - Scipy Status: {self.metric_history.get('status')} ({self.metric_history.get('message', 'N/A')})")
        
        safe_append("=" * 60)
        
        summary_text = '\n'.join(output_lines)
        
        if print_output:
            print(summary_text)

        return summary_text
    

    def info(self, print_output: bool = True) -> str:
        """最適化の結果とパラメータを整形して出力します。"""
        
        # 軌道長さを計算
        trajectory_length = self.time_steps * self.dt
        
        # PSOとScipyで異なる収束情報を取得
        n_iters = self.optimization_summary.get('nit', self.optimization_summary.get('n_actual_iterations', 'N/A'))
        n_fev = self.optimization_summary.get('nfev', 'N/A')
        
        # cost_historyの傾向
        cost_tendency_str = "なし"
        if self.cost_history and len(self.cost_history) > 1:
            first_cost = self.cost_history[0]
            last_cost = self.cost_history[-1]
            cost_tendency_str = f"初期: {first_cost:.4f} -> 最終: {last_cost:.4f}"
        
        output = [
            "==================================================",
            f"🚀 OPTIMIZATION RESULT PREVIEW ({self.optimizer_type})",
            "==================================================",
            
            "## 1. 最終結果とステータス (Quick Look)",
            f"   - 最小コスト (Best Cost): {self.best_cost:.6f}",
            f"   - 収束ステータス: {self.optimization_summary.get('success', 'N/A')}",
            f"   - 終了メッセージ: {self.optimization_summary.get('message', 'N/A')}",
            f"   - 最終メトリクス: {_format_dict(self.final_metrics, max_items=6, precision=4)}",
            f"   - 最適軌道 (qs) サイズ: {self.qs.shape}",

            "\n## 2. 最適化設定",
            f"   - コスト関数 (Cost Func): {self.cost_func.get('target_str', 'N/A')}",
            f"   - オプティマイザ: {self.optimizer_type}",
            f"   - プロセス設定 (Params): {_format_dict(self.optimizer_params, max_items=5)}",

            "\n## 3. 計算量と履歴",
            f"   - 反復回数 (nit): {n_iters}",
            f"   - 関数評価回数 (nfev): {n_fev}",
            f"   - コスト履歴: {cost_tendency_str} ({len(self.cost_history) if self.cost_history else 0} points)",
            
            "\n## 4. ロボットと軌道パラメータ (再現用)",
            f"   - 自由度 (DOF): {self.dof}",
            f"   - モデルキー: {self.robot_model_key}",
            f"   - 総時間: {trajectory_length:.2f} s ({self.time_steps} steps)",
            f"   - 重力補償: {self.compensate_grav}",
            f"   - 初期/目標姿勢: Start: {_format_array(self.start)}, End: {_format_array(self.end)}",

            "\n## 5. Perception Agentパラメータ",
            f"   - Agent設定: {'有効' if self.agent_params else '無効'}",
            f"   - Agent Params: {_format_dict(self.agent_params, max_items=3)}",
        ]

        summary_text = "\n".join(output)
        if print_output:
            print(summary_text)
        return summary_text
    
    
# ユーティリティ関数
def _format_array(arr: Optional[np.ndarray], max_len=5, precision=3) -> str:
    """NumPy配列を簡潔な文字列にフォーマットするヘルパー関数"""
    if arr is None:
        return "None"
    
    # 配列が非常に長い場合は、最初のmax_len個の要素のみを表示
    if arr.size > max_len:
        summary = f"Shape {arr.shape}, Dtype {arr.dtype}, Max: {arr.max():.{precision}f}, Min: {arr.min():.{precision}f}"
        return summary
    else:
        # 要素数が少ない場合は全て表示
        return f"Shape {arr.shape}, {arr.tolist()}"

def _format_dict(d: Optional[dict], max_items=4, precision=4) -> str:
    """辞書を簡潔な文字列にフォーマットするヘルパー関数"""
    if not d:
        return "{}"
    
    items = []
    # 辞書のキーと値のペアを文字列化
    for i, (k, v) in enumerate(d.items()):
        if i >= max_items and len(d) > max_items:
            items.append(f"... (+{len(d) - max_items} more)")
            break
            
        if isinstance(v, (float, np.floating)):
            v_str = f"{v:.{precision}f}"
        elif isinstance(v, np.ndarray):
             v_str = f"Array {v.shape}"
        elif isinstance(v, str) and len(v) > 30:
            v_str = f'"{v[:30]}..."'
        else:
            v_str = str(v)
            
        items.append(f"'{k}': {v_str}")
        
    return f"{{ {', '.join(items)} }}"


# class ExperimentManager:
#     def __init__(self, root_dir: str = "results"):
#         self.root_dir = Path(root_dir)

#     def create_group_folder(self, robot_key: str, opt_type: str, common_config: dict) -> Path:
#         """実験グループフォルダを作成し、共通設定を保存する"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         group_name = f"{timestamp}_{robot_key}_{opt_type}"
#         group_path = self.root_dir / group_name
        
#         group_path.mkdir(parents=True, exist_ok=False) # 既に存在する場合はエラー
        
#         # 共通設定の保存
#         with open(group_path / "common_config.json", 'w') as f:
#             json.dump(common_config, f, indent=4) # NumPy配列を扱う場合はカスタムエンコーダが必要

#         return group_path

#     def save_trial_result(self, group_path: Path, result: OptimizationResult, trial_index: int, trial_params: dict):
#         """個別試行の結果を保存する"""
        
#         # フォルダ名の決定
#         trial_name = f"{trial_index:03d}_cost{result.best_cost:.3f}"
#         trial_path = group_path / trial_name
#         trial_path.mkdir(exist_ok=True)

#         # 1. result.pkl の保存
#         result.save(trial_path / "result.pkl")

#         # 2. trajectory.npy の保存
#         np.save(trial_path / "trajectory.npy", result.qs)
        
#         # 3. trial_params.json の保存
#         with open(trial_path / "trial_params.json", 'w') as f:
#             json.dump(trial_params, f, indent=4)
        
#         # 4. metrics_history の保存 (result.metric_historyをJSON/NPYで保存)
#         if result.cost_history is not None:
#              # 例としてコスト履歴をJSONで保存
#             with open(trial_path / "metrics_history.json", 'w') as f:
#                 json.dump({"cost_history": result.cost_history.tolist()}, f, indent=4)

#         print(f"結果を {trial_path} に保存しました。")
        
#     def load_result(self, trial_path: Path) -> OptimizationResult:
#         """result.pklをロードする"""
#         return OptimizationResult.load(trial_path / "result.pkl")



# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML, display

# 型エイリアス
Array = np.ndarray
ScopeType = Dict[str, Tuple[float, float]]
DrawParamsType = Dict[str, Union[float, int, str, bool, Tuple[float, float]]]

# 描画パラメータのデフォルト値
DEFAULT_DRAW_PARAMS: DrawParamsType = {
    'link_lw': 5,
    'link_color': 'black',
    'joint_size': 10,
    'joint_face_color': 'white',
    'joint_edge_color': 'black',
    'ee_size': 30,
    'ee_color': 'red',
    'ee_lw': 5,
    'base_color': 'gray',
    'base_alpha': 1.0,
    'gripper_color': 'red',
    'gripper_lw': 3,
    'draw_base': True,           # 描画のON/OFF
    'draw_gripper': True,        # 描画のON/OFF
    'base_width': 0.5,           # 土台の幅
    'base_height': 0.1,          # 土台の高さ
    'gripper_size': 0.3,
    'gripper_angle_bend': 60,    # グリッパの折れ角度 [deg]
    'gripper_angle_open': 60,    # グリッパの開き角度 [deg]
}


def plot_robot_motion(result: OptimizationResult = None, qs=None, opt=None, 
                    #   model: pin.Model, qs: Array, dt: float,
                      folder_path: str = './tmp_movies/', file_name: Optional[str] = None, 
                      grid: bool = False, title: str = '', is3d: bool = False, 
                      detail: bool = False, plane: str = 'xz', 
                      scope: Optional[ScopeType] = None,
                      draw_params: Optional[DrawParamsType] = None) -> None:
    """
    ロボットの動きをmatplotでアニメーション表示・保存する関数

    Parameters
    ----------
    model : pin.Model
        ロボットモデル (必須)
    qs : ndarray (T, n)
        各時刻の関節角度列 (必須)
    dt : float
        シミュレーションの時間刻み [s] (必須)
    folder_path : str
        ムービーの保存先ディレクトリ。
    file_name : str or None
        保存するファイル名（拡張子付き）。NoneならJupyter内で表示。
    ... (その他の描画・設定パラメータ)
    draw_params : Dict
        ロボットの**描画に関するすべてのカスタムパラメータ**を格納した辞書。
        以下のキーを使用して、リンクの色、線幅、ジョイントやエンドエフェクタのサイズなどをカスタマイズできる。
        指定しないパラメータにはデフォルト値が適用される。
        
        **利用可能なキーとデフォルト値:**
        
        | キー名 | デフォルト値 | 説明 |
        | :--- | :--- | :--- |
        | `link_lw` | `5` | リンクの線幅。 |
        | `link_color` | `'black'` | リンクの色。 |
        | `joint_size` | `10` | ジョイントの点のサイズ。 |
        | `joint_face_color` | `'white'` | ジョイントの塗りつぶしの色。 |
        | `joint_edge_color` | `'black'` | ジョイントの枠線の色。 |
        | `ee_size` | `30` | エンドエフェクタの点のサイズ。 |
        | `ee_color` | `'red'` | エンドエフェクタの色。 |
        | `ee_lw` | `5` | エンドエフェクタの枠線の太さ。 |
        | `draw_base` | `True` | 土台（ベース）の描画ON/OFF。 |
        | `base_width` | `0.5` | 土台の幅（X軸方向）。 |
        | `base_height` | `0.1` | 土台の高さ（Z軸方向）。 |
        | `base_color` | `'gray'` | 土台の色。 |
        | `base_alpha` | `1.0` | 土台の透明度。 |
        | `draw_gripper` | `True` | グリッパの描画ON/OFF。 |
        | `gripper_size` | `0.3` | グリッパのサイズ係数。 |
        | `gripper_angle_bend`| `60` | グリッパの折れ曲がる角度 [deg]。|
        | `gripper_angle_open`| `60` | グリッパの開き角度 [deg]。|
        | `gripper_color` | `'red'` | グリッパの色。 |
        | `gripper_lw` | `3` | グリッパの線幅。 |

    DEFAULT_DRAW_PARAMS: DrawParamsType = {
        'link_lw': 5,
        'link_color': 'black',
        'joint_size': 10,
        'joint_face_color': 'white',
        'joint_edge_color': 'black',
        'ee_size': 30,
        'ee_color': 'red',
        'ee_lw': 5,
        'base_color': 'gray',
        'base_alpha': 1.0,
        'gripper_color': 'red',
        'gripper_lw': 3,
        'draw_base': True,           # 描画のON/OFF
        'draw_gripper': True,        # 描画のON/OFF
        'base_width': 0.5,           # 土台の幅
        'base_height': 0.1,          # 土台の高さ
        'gripper_size': 0.3,
        'gripper_angle_bend': 60,    # グリッパの折れ角度 [deg]
        'gripper_angle_open': 60,    # グリッパの開き角度 [deg]
    }
    """

    if result:
        model = result.robot_model
        qs = result.qs
        dt = result.dt
        fps = result.fps
        interval = 1000/fps
        time_steps = result.time_steps
        duration = time_steps * dt
        if fps == 0 or fps is None:
            interval = result.dt * 1000
    else:
        model = opt.model
        qs = qs
        dt = opt.dt
        fps = 1.0/dt
        interval = dt * 1000  # dt(sec) → interval(msec)
        time_steps = opt.timesteps
        duration = time_steps * dt

    
    # 描画パラメータの統合
    p = DEFAULT_DRAW_PARAMS.copy()
    if draw_params:
        p.update(draw_params)



    # ロボットの最大リーチを計算
    max_range = np.sum([np.linalg.norm(model.jointPlacements[i].translation) for i in range(1, model.njoints)])
    ee_frame_id = model.getFrameId("ee_tip")
    if ee_frame_id < model.nframes:
        max_range += np.linalg.norm(model.frames[ee_frame_id].placement.translation)

    # プロットセットアップ
    if is3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
    else:
        # 2Dの場合、scopeが指定されていないなら、最大リーチに基づいてaxを作成
        if scope is None:
            # 簡素化された create_ax_with_scope の代替
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            max_range_ext = max_range * 1.1
            ax.set_xlim(-max_range_ext, max_range_ext)
            ax.set_ylim(-max_range_ext, max_range_ext)
        else:
            fig, ax = create_ax_with_scope(scope, is3d=False)


    # アニメーション更新関数
    def update(frame):
        ax.cla()
        q = qs[frame]
        time = frame * dt
        
        current_title = f"{title} (Time: {time:.2f} s / {duration:.2f} s)"
        
        if is3d:
            plot_robot_3D(model, q, ax=ax, show=False, detail=detail, max_range=max_range, title=current_title, draw_params=p)
            if detail:
                 ax.set_title(current_title, fontsize=10)
        else:
            plot_robot_2D(model, q, ax=ax, show=False, detail=detail, grid=grid, scope=scope,
                          title=current_title, plane=plane, max_range=max_range,
                          draw_params=p) # 描画パラメータを渡す
            if detail:
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                ax.text(xlim[1]*0.7, ylim[1]*0.9, 
                        f"Frame: {frame} / {time_steps}\nFPS: {fps:.1f}", 
                        ha='left', va='top', fontsize=7, 
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.8))
        return ax,

    ani = animation.FuncAnimation(
        fig, update, frames=time_steps, interval=interval, blit=False
    )
    plt.close(fig) # Jupyterでの重複表示を防ぐために一時的に閉じる

    if file_name is None:
        print("Now preparing HTML display...")
        # Jupyter環境での表示
        display(HTML(ani.to_jshtml()))
    
    else:
        # パス処理のモダン化と統一 (folder_path / file_name)
        save_path = Path(folder_path)
        save_path.mkdir(parents=True, exist_ok=True) # フォルダを作成
        
        movie_path = Path(file_name)
        base = movie_path.stem
        ext = movie_path.suffix.lower()

        # 拡張子に基づいたWriterの決定
        writer = 'pillow' # デフォルトはGIF
        if ext in ['.mp4', '.mov', '.avi']:
            writer = 'ffmpeg'
        elif ext == '':
            ext = '.mp4'
            writer = 'ffmpeg'
        elif ext not in ['.gif', '.mp4', '.mov', '.avi']:
            print(f"Unsupported extension '{ext}' — saving as '{base}.mp4'...")
            ext = '.mp4'

        final_filepath = save_path / f"{base}{ext}"
        print(f"Saving animation as '{final_filepath}'...")
        
        ani.save(final_filepath, writer=writer, fps=fps)

        print("Saved successfully.")

def plot_robot_3D(model: pin.Model, q: Array, ax: Optional[plt.Axes] = None, 
                  show: bool = True, detail: bool = False, title: str = 'Robot Arm 3D', 
                  max_range: Optional[float] = None, draw_params: Optional[DrawParamsType] = None) -> plt.Axes:
    """
    Pinocchioモデルの現在姿勢をmatplotで3Dプロットする関数
    """
    
    # 描画パラメータの統合
    p = DEFAULT_DRAW_PARAMS.copy()
    if draw_params:
        p.update(draw_params)
        
    data = model.createData()
    # ... (順運動学の計算、描画ロジックは前回の実装と同じ。p['...']でアクセス) ...
    pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])

    # --- リンクとジョイントの描画 ---
    for joint_id in range(1, model.njoints):
        pos = data.oMi[joint_id].translation
        parent_id = model.parents[joint_id]
        
        if parent_id >= 0:
            parent_pos = data.oMi[parent_id].translation
            xs, ys, zs = zip(parent_pos, pos)
            ax.plot(xs, ys, zs, c=p['link_color'], linewidth=p['link_lw'])

        ax.scatter(pos[0], pos[1], pos[2], c=p['joint_edge_color'], s=p['joint_size']*5, marker='o')
        if detail:
            ax.text(pos[0], pos[1], pos[2], model.names[joint_id], fontsize=7)
      
    # --- エンドエフェクタの描画 ---
    ee_frame_id = model.getFrameId("ee_tip")
    if ee_frame_id < model.nframes:
        frame = model.frames[ee_frame_id]
        pos = data.oMf[ee_frame_id].translation
        parent_pos = data.oMi[frame.parentJoint].translation
        
        xs, ys, zs = zip(parent_pos, pos)
        ax.plot(xs, ys, zs, c=p['ee_color'], linewidth=p['ee_lw'])
        ax.scatter(pos[0], pos[1], pos[2], c=p['ee_color'], s=p['ee_size'], marker='o')
        if detail:
            ax.text(pos[0], pos[1], pos[2], frame.name, color=p['ee_color'], fontsize=7)
    
    # 描画範囲設定は変更なし
    if max_range is not None:
        max_range_val = max_range * 1.1
    else:
        max_range_val = np.sum([np.linalg.norm(model.jointPlacements[i].translation) for i in range(1, model.njoints)]) * 1.3

    ax.set_xlim([-max_range_val, max_range_val])
    ax.set_ylim([-max_range_val, max_range_val])
    ax.set_zlim([-max_range_val, max_range_val])

    # グラフ調整は変更なし
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if not detail:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_title('')
        
    if show:
        plt.show()
    return ax

def plot_robot_2D(model: pin.Model, q: Array, ax: Optional[plt.Axes] = None, 
                  show: bool = True, detail: bool = False, grid: bool = False, 
                  scope: Optional[ScopeType] = None, title: str = 'Robot Arm 2D', 
                  plane: str = 'xz', max_range: Optional[float] = None,
                  draw_params: Optional[DrawParamsType] = None) -> plt.Axes:
    """
    Pinocchioモデルの現在姿勢をmatplotで2Dプロットする関数
    """
    
    # 描画パラメータの統合
    p = DEFAULT_DRAW_PARAMS.copy()
    if draw_params:
        p.update(draw_params)
        
    # draw_baseとdraw_gripperのON/OFF
    draw_base = p.get('draw_base', True)
    draw_gripper = p.get('draw_gripper', True)

    # ... (順運動学の計算, 平面選択, ジョイント位置取得は省略) ...
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)

    if ax is None:
        # axがない場合のFig/Ax作成ロジックは簡略化
        fig, ax = plt.subplots() 
        ax.set_aspect('equal', adjustable='box')
    
    # 平面選択ロジック（前回の実装と同じ）
    plane_map = {'xy': (0, 1), 'yz': (1, 2), 'zy': (1, 2), 'zx': (0, 2), 'xz': (0, 2)}
    axis_idx = np.array(plane_map.get(plane.lower(), (0, 2))) # 'xz'をデフォルトとする

    # ジョイント位置の取得
    positions = [np.array([0.0, 0.0])]  # ワールド原点
    for i in range(1, model.njoints):
        pos = data.oMi[i].translation[axis_idx]
        positions.append(pos)
        if detail:
            ax.text(pos[0], pos[1], model.names[i], fontsize=8)
    positions = np.array(positions)
    
    # --- 土台を描画 ---
    if draw_base:
        base_w = p['base_width']
        base_h = p['base_height']
        base = plt.Rectangle((-base_w/2, -base_h), base_w, base_h, color=p['base_color'], alpha=p['base_alpha'])
        ax.add_patch(base)

    # --- グリッパを描画 ---
    ee_frame_id = model.getFrameId("ee_tip")
    if ee_frame_id < model.nframes:
        frame = model.frames[ee_frame_id]
        ee_pos = data.oMf[ee_frame_id].translation[axis_idx]
        parent_pos = data.oMi[frame.parentJoint].translation[axis_idx]
        
        # エンドエフェクタのリンク
        ax.plot([parent_pos[0], ee_pos[0]], [parent_pos[1], ee_pos[1]], c=p['link_color'], linewidth=p['ee_lw'])
        ax.scatter(ee_pos[0], ee_pos[1], c=p['ee_color'], s=p['ee_size'], marker='o')

        if draw_gripper:
            # グリッパ描画ロジック
            v = ee_pos - parent_pos
            v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([1, 0])
            
            base_len = p['gripper_size'] * 0.5
            tip_len = p['gripper_size'] * 0.5
            bend_angle = np.deg2rad(p['gripper_angle_bend'])
            open_angle = np.deg2rad(p['gripper_angle_open'])
            
            def R(angle): # 回転行列
                return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            for side in [+1, -1]:
                open_dir = R(open_angle * side) @ v
                bend_dir = R(bend_angle * (-side)) @ open_dir
                
                base_end = ee_pos + open_dir * base_len
                tip_end = base_end + bend_dir * tip_len

                # 描画
                ax.plot([ee_pos[0], base_end[0]], [ee_pos[1], base_end[1]], c=p['gripper_color'], lw=p['gripper_lw'])
                ax.plot([base_end[0], tip_end[0]], [base_end[1], tip_end[1]], c=p['gripper_color'], lw=p['gripper_lw'])
                
    # --- ロボットのリンクとジョイントを描画 ---
    ax.plot(positions[:, 0], positions[:, 1], 'o-', linewidth=p['link_lw'], c=p['link_color'],
            markersize=p['joint_size'], markerfacecolor=p['joint_face_color'], markeredgecolor=p['joint_edge_color'])
    
    # 軸設定 (max_rangeが指定されていればscopeを上書き)
    # （軸設定ロジックは変更なし）
    if scope is None:
        if max_range is None:
            # max_range = np.sum([np.linalg.norm(model.jointPlacements[i].translation) for i in range(1, model.njoints)]) + p['gripper_size']
            # 1. 全フレーム（リンク先端含む）の初期位置から最大リーチを計算
            # translationのノルムの合計だけでなく、各フレームの配置を確認
            reach_coords = []
            for f in model.frames:
                reach_coords.append(np.linalg.norm(f.placement.translation))
            
            # ジョイント間の累積リーチを計算
            cumulative_reach = np.sum([np.linalg.norm(model.jointPlacements[i].translation) for i in range(1, model.njoints)])
            
            # グリッパの展開分も含めた最大値
            max_range = cumulative_reach + p.get('gripper_size', 0.3) * 1.5 # 少し余裕を持たせる

        # 2. 描画領域の決定
        margin = 1.2  # 1.1から1.2へ拡大
        base_h = p.get('base_height', 0.1)
        
        # 基本の描画幅
        lim_val = max_range * margin
        
        xlim = (-lim_val, lim_val)
        # 土台が下に見切れないよう、下方向(ylim[0])にbase_hを考慮
        ylim = (-lim_val - base_h, lim_val)
    else:
        xlim, ylim = scope['x'], scope['y']

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(grid)
    
    if detail:
        ax.set_title(title)
        ax.set_xlabel(plane[0].upper())
        ax.set_ylabel(plane[1].upper())
    else:
        ax.axis('off')
        
    if show:
        plt.show()

    return ax



def create_ax_with_scope(scope: ScopeType, base_height: float = 5.0, 
                         equal_aspect: bool = True, is3d: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """
    scope に基づいて適切なサイズの Figure/Axes を生成する。

    Parameters
    ----------
    scope : Dict[str, Tuple[float, float]]
        {'x': (xmin, xmax), 'y': (ymin, ymax)} の形式で指定。
    base_height : float
        図全体の高さ（inch単位）。幅はscopeの比に合わせて自動計算。
    equal_aspect : bool
        True の場合、スケールを等しくする。
    is3d : bool
        True の場合、3D Axes を作成する。

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    """

    # 範囲と比率を計算
    x_range = scope['x'][1] - scope['x'][0]
    y_range = scope['y'][1] - scope['y'][0]
    aspect_ratio = x_range / y_range

    # 図の生成（サイズをscope比に合わせる）
    fig_width = base_height * aspect_ratio
    
    if is3d:
        # 3Dの場合、アスペクト比を1:1:1に固定するため、幅=高さを維持し、projectionを指定
        fig, ax = plt.subplots(figsize=(base_height, base_height), subplot_kw={'projection': '3d'})
        # 3Dではequal_aspectの設定はset_box_aspectを使うためここでは設定しない
    else:
        fig, ax = plt.subplots(figsize=(fig_width, base_height))

    # 軸設定
    ax.set_xlim(scope['x'])
    ax.set_ylim(scope['y'])
    
    if not is3d:
        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')
            
    return fig, ax

def get_keyframes(filepath: str, num_samples: int, output_dir: Optional[str] = None) -> List[Path]:
    """
    GIFファイルから指定された数のキーフレームを等間隔に抽出し、PNGファイルとして保存する。
    
    Args:
        filepath (str): GIFファイルのパス。
        num_samples (int): 抽出したいキーフレームの総数 (最初と最後を含む)。
        output_dir (Optional[str]): 出力先フォルダのパス。
                                    Noneの場合、GIFファイルと同じディレクトリに 
                                    "{GIF名}_keyframes" フォルダが作成される。
                                    
    Returns:
        List[Path]: 保存されたすべてのキーフレームファイルのパスリスト。
    """
    
    # Pathオブジェクトに変換
    gif_path = Path(filepath)
    gif_name = gif_path.stem  # 拡張子を除いたファイル名

    # 1. 出力先フォルダの決定と作成
    if output_dir is None:
        output_path = gif_path.parent / f"{gif_name}_keyframes"
    else:
        output_path = Path(output_dir)
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. GIFを開き、フレーム数を取得
    try:
        img = Image.open(gif_path)
    except FileNotFoundError:
        print(f"❌ エラー: ファイルが見つかりません -> {filepath}")
        return []
    except Exception as e:
        print(f"❌ エラー: GIFファイルの読み込み中に問題が発生しました -> {e}")
        return []

    total_frames = img.n_frames
    
    if total_frames < num_samples:
        print(f"⚠️ 注意: 要求されたフレーム数 ({num_samples}) が総フレーム数 ({total_frames}) より多いため、すべて保存します。")
        num_samples = total_frames
    
    # 3. 等間隔のフレーム番号を計算 (最初と最後を含む)
    # (total_frames - 1) / (num_samples - 1) は間隔のステップサイズ
    step_size = (total_frames - 1) / (num_samples - 1)
    frame_indices = [round(i * step_size) for i in range(num_samples)]
    
    print(f'全体: {total_frames} フレーム。抽出インデックス: {frame_indices}')

    # 4. 指定したフレームを保存
    saved_paths = []
    for i, frame_index in enumerate(frame_indices):
        
        # img.seek は例外を出す可能性があるため try-except で囲む
        try:
            img.seek(frame_index)  # 指定フレームへ移動
        except EOFError:
            print(f"⚠️ 警告: フレーム {frame_index} にシークできませんでした。スキップします。")
            continue
            
        # ファイル名を生成: frame_00.png, frame_01.png...
        # 桁数を揃えることでソートしやすくする
        frame_name = f'frame_{i:0{len(str(num_samples-1))}}.png'
        save_file = output_path / frame_name
        
        # RGB変換して保存
        img.convert("RGB").save(save_file)
        saved_paths.append(save_file)
        
    print(f'✅ 完了しました。{len(saved_paths)} 個のキーフレームを {output_path} に保存しました。')
    
    return saved_paths

# --- 使用例 (実行には PIL がインストールされている必要があります) ---
# import os
# # 存在しないダミーファイル
# # dummy_gif_path = "path/to/your/animation.gif" 
# # if os.path.exists(dummy_gif_path):
# #     get_keyframes(dummy_gif_path, num_samples=10)

def save_numpy_array(folder_path: str, file_name: str, data: np.ndarray, compressed: bool = False) -> Path:
    """
    指定されたフォルダパスにNumPy配列を保存する。

    保存先フォルダが存在しない場合は自動で作成される。

    Args:
        folder_path (str): 保存先のフォルダパス。
        file_name (str): 保存するファイル名 (拡張子 .npy または .npz を推奨)。
        data (np.ndarray): 保存したいNumPy配列。
        compressed (bool): True の場合、np.savez_compressed (.npz) を使用して圧縮保存する。
                           False の場合、np.save (.npy) を使用する。

    Returns:
        pathlib.Path: 実際に保存されたファイルのフルパス。
    """
    
    # Pathオブジェクトに変換
    folder = Path(folder_path)
    
    # フォルダが存在しない場合は作成
    folder.mkdir(parents=True, exist_ok=True)
    
    # フォルダパスとファイル名を結合
    save_path = folder / file_name
    
    if compressed:
        # 圧縮保存 (複数の配列を保存する用途にも使えるが、ここでは単一配列を辞書形式で保存)
        # np.savez_compressed はキーワード引数を受け取るため、辞書形式で渡す
        np.savez_compressed(save_path, data=data)
        
        # 注意: np.savez_compressed で保存した場合、読み込みは np.load(path)['data'] となる
        
    else:
        # 通常のバイナリ保存 (.npy)
        np.save(save_path, data)
        
    return save_path

import json
import pandas as pd
from typing import Union, Any

def visualize_json(data: Union[str, dict, list], max_rows: int = 20) -> None:
    """
    JSONデータ（文字列またはPythonオブジェクト）を「いい感じに」可視化する。
    
    1. データがリストで、内部が辞書オブジェクトで構成されている場合: Pandas DataFrameに変換して表示する。
    2. それ以外の場合: 構造を整形（Pretty Print）して表示する。

    Parameters
    ----------
    data : str, dict, or list
        JSONデータ、またはそれを表現するPythonオブジェクト。
    max_rows : int, default=20
        DataFrame表示時に最大で表示する行数（データが多い場合に省略表示するため）。
    """
    
    # 1. データが文字列の場合はパースを試みる
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            print("エラー: 入力された文字列は有効なJSON形式ではありません。")
            return

    # 2. リスト of Dict の場合は DataFrame に変換して表示
    # 全ての要素が辞書であり、かつデータが空でないことを確認
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        try:
            # json_normalizeでネストされたJSONもフラット化する
            df = pd.json_normalize(data)
            print("=" * 60)
            print("<<< Tabular View (Pandas DataFrame) >>>")
            
            # DataFrameをきれいに表示（列数が多い場合は省略せず表示、行数が多い場合は省略）
            with pd.option_context('display.max_rows', max_rows, 'display.max_columns', None, 'display.width', 1000):
                if len(df) > max_rows:
                    head_count = max_rows // 2
                    tail_count = max_rows - head_count
                    print(df.head(head_count).to_markdown(index=False))
                    print(f"\n... (全 {len(df)} 行中、最初の {head_count} 行と最後の {tail_count} 行を表示) ...\n")
                    print(df.tail(tail_count).to_markdown(index=False))
                else:
                    print(df.to_markdown(index=False))
            print("=" * 60)
            return
        except Exception as e:
            # 構造が複雑すぎてフラット化に失敗した場合、Pretty Printにフォールバック
            print(f"警告: Pandas DataFrameへの変換に失敗しました（{e}）。代わりに階層構造を整形して表示します。")
            pass

    # 3. それ以外（Dictや複雑な構造）の場合は Pretty Print
    if isinstance(data, (dict, list)):
        print("=" * 60)
        print("<<< Hierarchical View (Pretty Print) >>>")
        # indent=4 で整形、ensure_ascii=False で日本語文字化けを防ぐ
        print(json.dumps(data, indent=4, ensure_ascii=False))
        print("=" * 60)
        return

    # 4. JSON以外のオブジェクトの場合
    print(f"--- Simple Representation of Object ({type(data).__name__}) ---")
    print(data)

# 以下は使用例です（実行環境で試す場合は、DataFrameの表示にPandasのインストールが必要です）
# import numpy as np
# from collections import OrderedDict

# # 例1: リスト of Dict (表形式に向いている)
# sample_tabular_data = [
#     {"iter": 1, "cost": 1.25, "metrics": {"time": 0.5, "updates": 10}},
#     {"iter": 2, "cost": 1.01, "metrics": {"time": 0.8, "updates": 15}},
#     {"iter": 3, "cost": 0.99, "metrics": {"time": 1.1, "updates": 18}}
# ]
# visualize_json(sample_tabular_data)

# # 例2: 複雑な設定ファイル (階層形式に向いている)
# sample_config_data = {
#     "robot": {"dof": 7, "model": "panda"},
#     "optimizer": {"type": "PSO", "params": {"n_particles": 50, "max_iter": 100}},
#     "costs": [
#         {"type": "energy", "weight": 1.0},
#         {"type": "jerk", "weight": 0.1}
#     ]
# }
# visualize_json(sample_config_data)

import numpy as np
from typing import List, Dict, Any, Tuple

# === 以前のステップで生成された targets_list が存在すると仮定 ===
# targets_list は前のターンで生成されています。
# 例: visualize_json(targets_list) の結果に基づき、targets_listは既に定義済みとします。
# -------------------------------------------------------------------

def get_level_label(value: float, low: float, middle: float, high: float, tolerance: float = 1e-6) -> str:
    """
    ターゲット値が、範囲のどのレベル（L/M/H）に最も近いかを判定する。
    
    Parameters
    ----------
    value : float
        判定したいターゲット値。
    low : float
        範囲の最小値 (Low)
    middle : float
        範囲の中間値 (Middle)
    high : float
        範囲の最大値 (High)
    tolerance : float
        比較許容誤差。
        
    Returns
    -------
    str
        'L', 'M', 'H' のいずれか。
    """
    
    # ターゲット値がlow, middle, highのいずれかと完全に一致すると仮定し、比較。
    # 浮動小数点誤差を考慮し、最も近いものを選ぶか、一致するものを選ぶ。
    if np.isclose(value, low, atol=tolerance):
        return 'L'
    elif np.isclose(value, middle, atol=tolerance):
        return 'M'
    elif np.isclose(value, high, atol=tolerance):
        return 'H'
    else:
        # 万が一、設定した3値のいずれとも一致しない場合は、最も近いものを選んでフォールバックする
        targets = {'L': low, 'M': middle, 'H': high}
        closest_label = min(targets, key=lambda label: abs(targets[label] - value))
        print(f"警告: ターゲット値 {value:.4f} はL/M/Hのいずれとも一致しませんでした。最も近い '{closest_label}' を使用します。")
        return closest_label

def generate_motion_names(targets_list: List[Dict[str, Any]], dof: int) -> List[Tuple[Dict[str, Any], str]]:
    """
    ターゲットリストの各組み合わせに対し、一意で意味のある名前を生成する。
    
    Parameters
    ----------
    targets_list : List[Dict[str, Any]]
        igとenergyのターゲット設定のリスト。
    dof : int
        ロボットの自由度（DOF）。
        
    Returns
    -------
    List[Tuple[Dict[str, Any], str]]
        元のターゲット設定と、生成された名前のタプルのリスト。
    """
    
    named_targets = []
    dof_label = f"dof{dof}"
    
    # ターゲット値の共通の計算パラメータ（どの組み合わせでもmin/max/middleは同じ）
    # リストの最初の要素から代表値を取得する
    first_target = targets_list[0]
    ig_low = first_target['ig']['min']
    ig_middle = (first_target['ig']['min'] + first_target['ig']['max']) / 2
    ig_high = first_target['ig']['max']
    
    energy_low = first_target['energy']['min']
    energy_middle = (first_target['energy']['min'] + first_target['energy']['max']) / 2
    energy_high = first_target['energy']['max']
    
    for targets in targets_list:
        # IGのレベル判定
        ig_target_value = targets['ig']['target']
        ig_level = get_level_label(ig_target_value, ig_low, ig_middle, ig_high)
        
        # Energyのレベル判定
        energy_target_value = targets['energy']['target']
        energy_level = get_level_label(energy_target_value, energy_low, energy_middle, energy_high)
        
        # 名前を結合: [IG_Level]_[Energy_Level]_[DOF]
        motion_name = f"IG{ig_level}_E{energy_level}_{dof_label}"
        
        named_targets.append((targets, motion_name))
        
    return named_targets

# -------------------------------------------------------------------
# 使用例
# -------------------------------------------------------------------

# # 1. ロボットの自由度を設定 (例として DOF=3 を使用)
# ROBOT_DOF = 3 

# # 2. 関数を実行 (targets_listは前のステップで計算されたものを使用)
# named_motions = generate_motion_names(targets_list, ROBOT_DOF)

# # 3. 結果の表示
# print("=" * 60)
# print(f"🤖 DOF={ROBOT_DOF} に基づく生成された動作名 ({len(named_motions)} 種類)")
# print("=" * 60)

# for i, (targets, name) in enumerate(named_motions):
#     ig_val = targets['ig']['target']
#     energy_val = targets['energy']['target']
#     print(f"[{i+1:02d}] {name.ljust(15)}: IG={ig_val:.4f}, Energy={energy_val:.4f}")

# print("=" * 60)

import json
import pandas as pd
from typing import Union, Any

def visualize_json(data: Union[str, dict, list], max_rows: int = 20) -> None:
    """
    JSONデータ（文字列またはPythonオブジェクト）を「いい感じに」可視化する。
    
    1. データがリストで、内部が辞書オブジェクトで構成されている場合: Pandas DataFrameに変換して表示する。
    2. それ以外の場合: 構造を整形（Pretty Print）して表示する。

    Parameters
    ----------
    data : str, dict, or list
        JSONデータ、またはそれを表現するPythonオブジェクト。
    max_rows : int, default=20
        DataFrame表示時に最大で表示する行数（データが多い場合に省略表示するため）。
    """
    
    # 1. データが文字列の場合はパースを試みる
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            print("エラー: 入力された文字列は有効なJSON形式ではありません。")
            return

    # 2. リスト of Dict の場合は DataFrame に変換して表示
    # 全ての要素が辞書であり、かつデータが空でないことを確認
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        try:
            # json_normalizeでネストされたJSONもフラット化する
            df = pd.json_normalize(data)
            print("=" * 60)
            print("<<< Tabular View (Pandas DataFrame) >>>")
            
            # DataFrameをきれいに表示（列数が多い場合は省略せず表示、行数が多い場合は省略）
            with pd.option_context('display.max_rows', max_rows, 'display.max_columns', None, 'display.width', 1000):
                if len(df) > max_rows:
                    head_count = max_rows // 2
                    tail_count = max_rows - head_count
                    print(df.head(head_count).to_markdown(index=False))
                    print(f"\n... (全 {len(df)} 行中、最初の {head_count} 行と最後の {tail_count} 行を表示) ...\n")
                    print(df.tail(tail_count).to_markdown(index=False))
                else:
                    print(df.to_markdown(index=False))
            print("=" * 60)
            return
        except Exception as e:
            # 構造が複雑すぎてフラット化に失敗した場合、Pretty Printにフォールバック
            print(f"警告: Pandas DataFrameへの変換に失敗しました（{e}）。代わりに階層構造を整形して表示します。")
            pass

    # 3. それ以外（Dictや複雑な構造）の場合は Pretty Print
    if isinstance(data, (dict, list)):
        print("=" * 60)
        print("<<< Hierarchical View (Pretty Print) >>>")
        # indent=4 で整形、ensure_ascii=False で日本語文字化けを防ぐ
        print(json.dumps(data, indent=4, ensure_ascii=False))
        print("=" * 60)
        return

    # 4. JSON以外のオブジェクトの場合
    print(f"--- Simple Representation of Object ({type(data).__name__}) ---")
    print(data)

# 以下は使用例です（実行環境で試す場合は、DataFrameの表示にPandasのインストールが必要です）
# import numpy as np
# from collections import OrderedDict

# # 例1: リスト of Dict (表形式に向いている)
# sample_tabular_data = [
#     {"iter": 1, "cost": 1.25, "metrics": {"time": 0.5, "updates": 10}},
#     {"iter": 2, "cost": 1.01, "metrics": {"time": 0.8, "updates": 15}},
#     {"iter": 3, "cost": 0.99, "metrics": {"time": 1.1, "updates": 18}}
# ]
# visualize_json(sample_tabular_data)

# # 例2: 複雑な設定ファイル (階層形式に向いている)
# sample_config_data = {
#     "robot": {"dof": 7, "model": "panda"},
#     "optimizer": {"type": "PSO", "params": {"n_particles": 50, "max_iter": 100}},
#     "costs": [
#         {"type": "energy", "weight": 1.0},
#         {"type": "jerk", "weight": 0.1}
#     ]
# }
# visualize_json(sample_config_data)


def _get_keyframes(filepath, num_samples, output_dir=None):
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


def save_optimizer(folder_path: str, optimizer: Any):
    sim_dir = folder_path
    opt = optimizer
    os.makedirs(sim_dir, exist_ok=True)
    # simulationの設定を保存するフォルダ
    config_dir = os.path.join(sim_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # 設定を保存
    # config_jsonを保存
    file_name = "config.json"
    config_json_path = os.path.join(config_dir, file_name)
    with open(config_json_path, 'w') as f:
        json.dump(config_json_path, f, indent=4)
    print(f"Config JSON saved to: {config_json_path}")

    # robot modelを保存
    robot = opt.model
    file_name = "robot.pkl"
    robot_path = os.path.join(config_dir, file_name)
    with open(robot_path, 'wb') as f:
        pickle.dump(robot, f)
    print(f"Robot model saved to: {robot_path}")

    # agentを保存
    agent = opt.agent
    file_name = "agent.pkl"
    agent_path = os.path.join(config_dir, file_name)
    with open(agent_path, 'wb') as f:
        pickle.dump(agent, f)
    print(f"Agent saved to: {agent_path}")

    # const_beliefsを保存
    file_name = "const_beliefs_result.pkl"
    const_beliefs_result = opt.const_beliefs_result
    const_beliefs_result.save(config_dir, file_name)
    # print(f"Const beliefs saved to: {const_beliefs_path}")


    # 1. オブジェクトの状態（__dict__）を取得
    optimizer_state = opt.__dict__
    # 2. 状態辞書を別のファイル名で保存
    # (例: "optimizer_state.pkl"として保存)
    state_file_name = "optimizer_state.pkl"
    state_optimizer_path = os.path.join(config_dir, state_file_name)
    with open(state_optimizer_path, 'wb') as f:
        pickle.dump(optimizer_state, f)
    print(f"Optimizer state (data) saved to: {state_optimizer_path}")

    try:
        # optimizerそのものを保存
        used_optimizer = opt
        file_name = "used_optimizer.pkl"
        optimizer_path = os.path.join(config_dir, file_name)
        with open(optimizer_path, 'wb') as f:
            pickle.dump(used_optimizer, f)
        print(f"Optimizer saved to: {optimizer_path}")
    except:
        print(f"Standard pickle failed for {file_name}, using state data as backup.")
        

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Optional, List, Dict, Any, Union

# Matplotlibのスタイル設定
plt.style.use('ggplot')


def plot_landscape_with_pso_results(
    df_landscape: pd.DataFrame,
    df_pso_results: Union[pd.DataFrame, Dict[str, Dict[str, float]]],
    save_dir: Optional[str] = None,
    plot_title: str = "IG vs Energy Landscape with PSO Results",
    x_metric: str = 'ig',
    y_metric: str = 'energy',
    color_size_metric: str = 'vfe',
    landscape_alpha: float = 0.3,
    pso_marker_size: float = 150,
    cmap_name: str = 'viridis'
):
    """
    指標空間（IG vs Energy）に、ランドスケープ分析データとPSOの最適化結果を重ねてプロットする。
    VFEの値は、点のサイズと色で表現し、3次元的な関係を示す。

    Parameters
    ----------
    df_landscape : pd.DataFrame
        ランドスケープ分析から得られたデータフレーム (ig, energy, vfeを含む)。
    df_pso_results : Union[pd.DataFrame, Dict[str, Dict[str, float]]]
        PSO最適化の結果。
        - DataFrameの場合: 1行が1つの最適化結果のメトリクス。
        - Dictの場合: target_idをキーとし、値が最終メトリクスを持つ辞書 (例: {"ig": 120.8, "energy": 228.0})。
    save_dir : Optional[str], default=None
        プロットを保存するディレクトリ。Noneの場合、プロットを表示する。
    plot_title : str
        プロットのタイトル。
    x_metric, y_metric, color_size_metric : str
        プロットに使用する指標名 (デフォルト: 'ig', 'energy', 'vfe')
    landscape_alpha : float
        ランドスケープデータの透明度。
    pso_marker_size : float
        PSO結果点のマーカーサイズ。
    cmap_name : str
        使用するカラーマップ名。
    """
    
    # ----------------------------------------------------
    # 1. PSO結果データの準備と整形
    # ----------------------------------------------------
    if isinstance(df_pso_results, dict):
        # 辞書形式の場合、DataFrameに変換
        pso_data = []
        for target_id, metrics in df_pso_results.items():
            row = {x_metric: metrics.get(x_metric), 
                   y_metric: metrics.get(y_metric), 
                   color_size_metric: metrics.get(color_size_metric),
                   'target_id': target_id}
            # ターゲット値もプロットするために追加
            target_ig = metrics.get('ig', {}).get('target') if isinstance(metrics.get('ig'), dict) else None
            target_energy = metrics.get('energy', {}).get('target') if isinstance(metrics.get('energy'), dict) else None
            row['target_ig'] = target_ig
            row['target_energy'] = target_energy
            pso_data.append(row)
        df_pso = pd.DataFrame(pso_data)
        
    elif isinstance(df_pso_results, pd.DataFrame):
        df_pso = df_pso_results.copy()
    else:
        raise ValueError("df_pso_results must be a DataFrame or a Dict of metrics.")

    # ----------------------------------------------------
    # 2. ランドスケープデータの VFEによるサイズ・描画順序の計算
    # ----------------------------------------------------
    color_size_data = df_landscape[color_size_metric] # VFE
    
    cmap = plt.cm.get_cmap(cmap_name) 
    normalize = plt.Normalize(color_size_data.min(), color_size_data.max())
    
    # サイズの計算 (VFEが大きいほど大きく)
    min_size = 10
    max_size = 150
    size_range = max_size - min_size
    normalized_vfe = (color_size_data - color_size_data.min()) / (color_size_data.max() - color_size_data.min())
    marker_sizes_vfe = normalized_vfe * size_range + min_size

    # 描画順序の決定 (VFEが小さいものほど先に描画し、大きいものが上書きされるように)
    sort_indices = np.argsort(color_size_data.values)

    # ----------------------------------------------------
    # 3. プロットの実行 (IG vs Energy)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    # **A. ランドスケープデータのプロット (背景)**
    scatter_landscape = ax.scatter(df_landscape[x_metric].iloc[sort_indices],
                                   df_landscape[y_metric].iloc[sort_indices],
                                   c=df_landscape[color_size_metric].iloc[sort_indices], # 色: VFE
                                   cmap=cmap, 
                                   norm=normalize, 
                                   s=marker_sizes_vfe[sort_indices], # サイズ: VFE
                                   alpha=landscape_alpha, 
                                   edgecolors='gray', 
                                   linewidths=0.2)

    # --- A. PSO/Hybrid 結果のプロット準備 ---
    # dict形式で渡された場合にDataFrameに変換する処理（念のため）
    if isinstance(df_pso_results, dict):
        df_pso = pd.DataFrame.from_dict(df_pso_results, orient='index')
    else:
        df_pso = df_pso_results.copy()

    # --- B. 動的なターゲットカラム名の特定 ---
    target_x_col = f"target_{x_metric}"
    target_y_col = f"target_{y_metric}"

    # **B. PSO最適化結果のプロット (重ね合わせ)**
    # ランドスケープのカラーマップと正規化を流用
    scatter_pso = ax.scatter(df_pso[x_metric],
                             df_pso[y_metric],
                             c='red', # 目立つ色 (赤)
                             marker='X', # 目立つマーカー
                             s=pso_marker_size, 
                             alpha=1.0, 
                             edgecolors='black', 
                             linewidths=1.5,
                             label='PSO Final Result')
    
    # # **C. ターゲット点のプロット**
    # # ターゲット値が存在する場合のみプロット (ターゲットが辞書形式で設定されていた場合)
    # if 'target_ig' in df_pso.columns and 'target_energy' in df_pso.columns:
    #     ax.scatter(df_pso['target_ig'],
    #                df_pso['target_energy'],
    #                c='blue', # ターゲットは青
    #                marker='o', # 丸マーカー
    #                s=pso_marker_size * 0.5, 
    #                alpha=0.7, 
    #                edgecolors='black', 
    #                linewidths=1.0,
    #                label='Target Point')
        
    #     # ターゲットと結果点を結ぶ線 (ずれの視覚化)
    #     for _, row in df_pso.iterrows():
    #         if row['target_ig'] is not None and row['target_energy'] is not None:
    #             ax.plot([row['target_ig'], row[x_metric]], 
    #                     [row['target_energy'], row[y_metric]], 
    #                     linestyle='--', color='gray', alpha=0.5, linewidth=1)
                
    # --- C. ターゲット点のプロット (存在する場合のみ) ---
    # ターゲットカラムが両方存在し、かつ値が全てNaNでないかチェック
    if target_x_col in df_pso.columns and target_y_col in df_pso.columns:
        
        # NaNを除去した有効なターゲットペアのみを抽出
        mask = df_pso[target_x_col].notna() & df_pso[target_y_col].notna()
        valid_targets = df_pso[mask]

        if not valid_targets.empty:
            # ターゲット地点の散布図
            ax.scatter(valid_targets[target_x_col],
                       valid_targets[target_y_col],
                       c='blue', 
                       marker='o', 
                       s=pso_marker_size * 0.6, 
                       alpha=0.8, 
                       edgecolors='white', 
                       linewidths=1.5,
                       label=f'Target ({x_metric}, {y_metric})',
                       zorder=5) # ターゲットを最前面に

            # ターゲットと結果点を結ぶ線 (ずれの視覚化)
            for _, row in valid_targets.iterrows():
                ax.plot([row[target_x_col], row[x_metric]], 
                        [row[target_y_col], row[y_metric]], 
                        linestyle=':', # 点線
                        color='blue', 
                        alpha=0.4, 
                        linewidth=1.2,
                        zorder=4)

    # ----------------------------------------------------
    # 4. ラベルとカラーバーの設定
    # ----------------------------------------------------
    ax.set_xlabel(f"{x_metric.upper()} (Information Gain)", fontsize=12)
    ax.set_ylabel(f"{y_metric.upper()} (Energy)", fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    ax.grid(True, alpha=0.5)
    ax.legend(loc='best')

    # カラーバーの追加 (VFEの値を示す)
    cbar = fig.colorbar(scatter_landscape, ax=ax, label=f'{color_size_metric.upper()} (Surprise) Value', pad=0.02)

    # ----------------------------------------------------
    # 5. 保存または表示
    # ----------------------------------------------------
    plt.tight_layout()
    if save_dir:
        # フォルダが存在しない場合は作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plot_filename = f"landscape_pso_overlay_{x_metric}_vs_{y_metric}.png"
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"✅ プロットを保存しました: {plot_path}")
    else:
        plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Optional, List, Dict, Any, Union

# Matplotlibのスタイル設定
plt.style.use('ggplot')


def plot_metrics_landscape(
    df_landscape: pd.DataFrame,
    df_pso_results: Union[pd.DataFrame, Dict[str, Dict[str, Any]]], # Anyに変更して柔軟性を向上
    save_dir: Optional[str] = None,
    landscape_alpha: float = 0.3,
    pso_marker_size: float = 150,
    cmap_name: str = 'viridis'
):
    """
    ランドスケープ分析データとPSOの最適化結果を重ねてプロットする。
    IG-EnergyとIG-VFEの両方を、データに存在するメトリクスに応じて柔軟に生成する。
    """
    
    # ----------------------------------------------------
    # 1. PSO結果データの準備と整形
    # ----------------------------------------------------
    df_pso = pd.DataFrame()
    is_target_only = False
    
    if isinstance(df_pso_results, dict):
        pso_data = []
        for target_id, metrics in df_pso_results.items():
            row = {'target_id': target_id}
            
            # PSO達成値 (ig, energy, vfe) を取得。存在しない場合はNone
            row['ig'] = metrics.get('ig', None)
            row['energy'] = metrics.get('energy', None)
            row['vfe'] = metrics.get('vfe', None)
            
            # ターゲット値を取得 (辞書形式を想定し、'target'キーから取得)
            target_ig = metrics.get('ig', {}).get('target') if isinstance(metrics.get('ig'), dict) else metrics.get('target_ig', None)
            target_energy = metrics.get('energy', {}).get('target') if isinstance(metrics.get('energy'), dict) else metrics.get('target_energy', None)
            target_vfe = metrics.get('vfe', {}).get('target') if isinstance(metrics.get('vfe'), dict) else metrics.get('target_vfe', None)

            row['target_ig'] = target_ig
            row['target_energy'] = target_energy
            row['target_vfe'] = target_vfe
            
            pso_data.append(row)
            
        df_pso = pd.DataFrame(pso_data)
    
    elif isinstance(df_pso_results, pd.DataFrame):
        df_pso = df_pso_results.copy()
    
    # ターゲット値のみが含まれているかチェック
    pso_metrics = ['ig', 'energy', 'vfe']
    has_pso_result = df_pso[pso_metrics].notna().any().any()
    
    if not has_pso_result and df_pso[['target_ig', 'target_energy', 'target_vfe']].notna().any().any():
        is_target_only = True
        print("⚠️ PSO結果の達成値がないため、ターゲット点のみをプロットします。")
        # 達成値がない場合、ターゲット点をダミーとして達成値にコピー（プロットの失敗を回避）
        for metric in pso_metrics:
            target_col = f'target_{metric}'
            if target_col in df_pso.columns:
                # ターゲット値を持つ行の達成値列をターゲット値で埋める
                df_pso.loc[df_pso[target_col].notna(), metric] = df_pso[df_pso[target_col].notna()][target_col]


    # ----------------------------------------------------
    # 2. プロットするメトリクスの決定 (IG vs ...)
    # ----------------------------------------------------
    x_metric = 'ig'
    y_candidates = ['energy', 'vfe']
    
    # ランドスケープデータに存在するY軸メトリクスをリストアップ
    y_metrics_to_plot = [y for y in y_candidates if y in df_landscape.columns]
    
    if not y_metrics_to_plot:
        print(f"❌ プロットに必要なメトリクス ({x_metric}, {y_candidates}) がランドスケープデータに見つかりませんでした。")
        return

    # ----------------------------------------------------
    # 3. 各メトリクス組み合わせに対するプロットの実行
    # ----------------------------------------------------
    
    for y_metric in y_metrics_to_plot:
        
        # VFEを色とサイズに使うが、IG vs VFEの場合はEnergyを色とサイズに使う
        if y_metric == 'vfe' and 'energy' in df_landscape.columns:
            color_size_metric = 'energy'
        elif y_metric == 'energy' and 'vfe' in df_landscape.columns:
            color_size_metric = 'vfe'
        else:
            # 3軸目のデータがない場合は、VFEを使用（その場合サイズや色付けはスキップされる）
            color_size_metric = 'vfe' # ダミーとして設定
        
        # ----------------------------------------------------
        # 3.1 ランドスケープデータの VFE/Energyによるサイズ・描画順序の計算
        # ----------------------------------------------------
        if color_size_metric in df_landscape.columns:
            color_size_data = df_landscape[color_size_metric] 
            cmap = plt.cm.get_cmap(cmap_name) 
            normalize = plt.Normalize(color_size_data.min(), color_size_data.max())
            
            # サイズの計算
            min_size = 10
            max_size = 150
            size_range = max_size - min_size
            normalized_csm = (color_size_data - color_size_data.min()) / (color_size_data.max() - color_size_data.min())
            marker_sizes_csm = normalized_csm * size_range + min_size

            # 描画順序の決定 (色が小さいものほど先に描画し、大きいものが上書きされるように)
            sort_indices = np.argsort(color_size_data.values)
        else:
            # 3軸目がない場合、全て同じサイズでプロット
            marker_sizes_csm = np.full(len(df_landscape), 50)
            sort_indices = np.arange(len(df_landscape))
            cmap, normalize = None, None

        
        plot_title = f"{x_metric.upper()} vs {y_metric.upper()} Landscape"
        fig, ax = plt.subplots(figsize=(10, 8))

        # **A. ランドスケープデータのプロット (背景)**
        scatter_landscape = ax.scatter(df_landscape[x_metric].iloc[sort_indices],
                                       df_landscape[y_metric].iloc[sort_indices],
                                       c=df_landscape[color_size_metric].iloc[sort_indices] if color_size_metric in df_landscape.columns else 'gray',
                                       cmap=cmap, 
                                       norm=normalize, 
                                       s=marker_sizes_csm[sort_indices],
                                       alpha=landscape_alpha, 
                                       edgecolors='gray', 
                                       linewidths=0.2)

        # ----------------------------------------------------
        # B. PSO最適化結果のプロット (重ね合わせ - 達成値)
        # ----------------------------------------------------
        
        # 達成値を持つ行のみをフィルタリング
        df_pso_results_achieved = df_pso[df_pso[x_metric].notna() & df_pso[y_metric].notna()]
        
        if not df_pso_results_achieved.empty:
            ax.scatter(df_pso_results_achieved[x_metric],
                       df_pso_results_achieved[y_metric],
                       c='red', # 目立つ色 (赤)
                       marker='X', # 目立つマーカー
                       s=pso_marker_size, 
                       alpha=1.0, 
                       edgecolors='black', 
                       linewidths=1.5,
                       label='PSO Final Result')
        
        # ----------------------------------------------------
        # C. ターゲット点のプロット
        # ----------------------------------------------------
        target_x_col = f'target_{x_metric}'
        target_y_col = f'target_{y_metric}'
        
        df_pso_targets = df_pso[df_pso[target_x_col].notna() & df_pso[target_y_col].notna()]
        
        if not df_pso_targets.empty:
            ax.scatter(df_pso_targets[target_x_col],
                       df_pso_targets[target_y_col],
                       c='blue', # ターゲットは青
                       marker='o', # 丸マーカー
                       s=pso_marker_size * 0.5, 
                       alpha=0.7, 
                       edgecolors='black', 
                       linewidths=1.0,
                       label='Target Point')
            
            # ターゲットと結果点を結ぶ線 (ずれの視覚化)
            for _, row in df_pso_targets.iterrows():
                # ターゲットと達成値のどちらも存在する場合のみ線を描画
                if row[x_metric] is not None and row[y_metric] is not None:
                    ax.plot([row[target_x_col], row[x_metric]], 
                            [row[target_y_col], row[y_metric]], 
                            linestyle='--', color='gray', alpha=0.5, linewidth=1)

        # ----------------------------------------------------
        # 4. ラベルとカラーバーの設定
        # ----------------------------------------------------
        ax.set_xlabel(f"{x_metric.upper()}", fontsize=12)
        ax.set_ylabel(f"{y_metric.upper()}", fontsize=12)
        ax.set_title(plot_title, fontsize=14)
        ax.grid(True, alpha=0.5)
        
        # ラベルが重複しないようにlegendを調整
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        # カラーバーの追加 (3軸目が存在する場合のみ)
        if color_size_metric in df_landscape.columns:
            cbar = fig.colorbar(scatter_landscape, ax=ax, label=f'{color_size_metric.upper()} Value', pad=0.02)
        
        # ----------------------------------------------------
        # 5. 保存または表示
        # ----------------------------------------------------
        plt.tight_layout()
        plot_filename = f"landscape_pso_overlay_{x_metric}_vs_{y_metric}.png"

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            plot_path = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_path, dpi=300)
            plt.close(fig)
            print(f"✅ プロットを保存しました: {plot_path}")
        else:
            plt.show()

# ------------------------------------------------------------
# 🎯 呼び出し方の例 (ターゲットのみをプロットするケース)
# ------------------------------------------------------------
# # 以下のデータは、このコードブロックの外部で定義されている必要があります
# # df = ... (ランドスケープデータフレーム)
# # targets_list_LE = ... (VFEとEnergyのL, M, Hターゲットを含む辞書)

# # ターゲットデータ形式の準備
# target_for_plot = {}
# for target_id, result in targets_list_LE.items():
#     # PSOの達成値（'ig', 'energy', 'vfe'）がないため、ターゲット値のみを辞書に追加
#     target_for_plot[target_id] = {
#         'target_ig': result['ig']['target'],
#         'target_energy': result['energy']['target'],
#         'target_vfe': result['vfe']['target'],
#         # 達成値のキーは省略 (Noneとなる)
#     }

# # plot_metrics_landscape(
# #     df_landscape=df,
# #     df_pso_results=target_for_plot,
# #     # save_dir=landscape_dir,
# #     # cmap_name='plasma'
# # )


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Optional, List, Dict, Any, Union

class ResearchVisualizer:
    def __init__(self, df: pd.DataFrame):
        """
        初期化: メインのデータフレーム（Landscapeデータ）を保持
        """
        self.df = df
        # デフォルトのカラースタイル設定
        plt.style.use('ggplot')
        self.custom_palette = {'kld': 'coral', 'bs': 'seagreen', 'ig': 'royalblue'}

    def _get_ax(self, ax: Optional[plt.Axes], figsize=(10, 8), projection=None) -> plt.Axes:
        """Axesを生成または取得するユーティリティ"""
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=projection)
        return ax

    # --- 1. ベースレイヤー: 散布図 ---
    def plot_landscape(self, ax: Optional[plt.Axes] = None, x='ig', y='energy', z='vfe', 
                       cmap='viridis', alpha=0.6, use_size=True) -> plt.Axes:
        """
        2次元散布図。z引数で色とサイズ（奥行き）を表現。
        """
        ax = self._get_ax(ax)
        
        # z値に基づいてソート（大きい値を手前に描画）
        sort_idx = np.argsort(self.df[z].values)
        df_s = self.df.iloc[sort_idx]
        
        # サイズの計算
        sizes = 20 + 180 * (df_s[z] - df_s[z].min()) / (df_s[z].max() - df_s[z].min()) if use_size else 50
        
        scatter = ax.scatter(df_s[x], df_s[y], c=df_s[z], s=sizes, 
                             cmap=cmap, alpha=alpha, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(x.upper())
        ax.set_ylabel(y.upper())
        plt.colorbar(scatter, ax=ax, label=f'{z.upper()} (Depth/Color)')
        return ax

    # --- 2. 統計レイヤー: バイオリン & ターゲットライン ---
    def plot_distribution(self, column: str, ax: Optional[plt.Axes] = None, color='#FF9999') -> plt.Axes:
        """
        特定の変数の分布と、P5, P50, P95, Mean(P5,P95) を表示
        """
        ax = self._get_ax(ax, figsize=(6, 4))
        sns.violinplot(y=self.df[column], ax=ax, color=color, inner='quartile', width=0.3)
        
        # 統計値の計算
        p5, p50, p95 = self.df[column].quantile([0.05, 0.5, 0.95])
        m_val = (p5 + p95) / 2
        
        # 水平線の追加
        ax.axhline(p5, color='black', ls='--', lw=1, alpha=0.7)
        ax.axhline(p50, color='black', ls='-', lw=1.5, alpha=0.8)
        ax.axhline(p95, color='black', ls='--', lw=1, alpha=0.7)
        ax.axhline(m_val, color='red', ls='-.', lw=1.5, label='Mean(P5,P95)')
        
        # テキストラベル
        ax.text(ax.get_xlim()[1], p50, f' P50:{p50:.2f}', va='center')
        ax.text(ax.get_xlim()[0], m_val, ' Mean ', color='red', ha='left', va='bottom', fontweight='bold')
        
        ax.set_title(f'Distribution: {column}')
        return ax

    # --- 3. 分析レイヤー: 回帰曲線 ---
    def add_regression_curves(self, ax: plt.Axes, pred_df: pd.DataFrame, 
                             x_col='vfe', target_vars=['ig', 'kld', 'bs']):
        """
        statsmodels等で計算済みの予測データ(pred_df)を線として追加
        """
        for var in target_vars:
            if var in pred_df.columns:
                ax.plot(pred_df[x_col], pred_df[var], 
                        color=self.custom_palette.get(var, 'black'), 
                        lw=3, label=f'{var} Fit', zorder=10)
        return ax



    # --- 4. 最適化レイヤー: PSO結果 ---
    # def add_pso_results(self, ax: plt.Axes, pso_df: pd.DataFrame, 
    #                     x_col='ig', y_col='energy', show_displacement=True):
    #     """
    #     PSOの最終到達点とターゲット地点をプロットし、その間を線で結ぶ
    #     """
    #     # ターゲット地点 (Blue)
    #     ax.scatter(pso_df[f'target_{x_col}'], pso_df[f'target_{y_col}'], 
    #                c='blue', marker='o', s=80, edgecolors='black', label='Target', zorder=11)
        
    #     # 最終結果 (Red)
    #     ax.scatter(pso_df[x_col], pso_df[y_col], 
    #                c='red', marker='X', s=150, edgecolors='black', label='PSO Result', zorder=12)
        
    #     # ズレを可視化する破線
    #     if show_displacement:
    #         for _, row in pso_df.iterrows():
    #             ax.plot([row[f'target_{x_col}'], row[x_col]], 
    #                     [row[f'target_{y_col}'], row[y_col]], 
    #                     'k--', alpha=0.4, lw=1, zorder=5)
    #     return ax
    def add_pso_results(self, ax: plt.Axes, pso_df: pd.DataFrame, 
                            x='ig', y='energy', show_displacement=True):
            """
            pso_df 内に x, y があれば結果をプロット。
            target_x, target_y があればターゲットをプロット。
            両方あればその間を破線で結ぶ。
            """
            # カラム名の定義
            res_x, res_y = x, y
            tar_x, tar_y = f'target_{x}', f'target_{y}'

            # 1. 結果 (Optimization Result) のプロット
            # x と y のカラムが DataFrame に存在するかチェック
            if res_x in pso_df.columns and res_y in pso_df.columns:
                # プロット用にNaNを除去
                valid_res = pso_df.dropna(subset=[res_x, res_y])
                if not valid_res.empty:
                    ax.scatter(valid_res[res_x], valid_res[res_y], 
                            c='red', marker='X', s=150, edgecolors='black', 
                            label=f'Result ({x}, {y})', zorder=12)
            
            # 2. ターゲット (Target) のプロット
            # target_x と target_y のカラムが存在するかチェック
            has_targets = tar_x in pso_df.columns and tar_y in pso_df.columns
            if has_targets:
                valid_tar = pso_df.dropna(subset=[tar_x, tar_y])
                if not valid_tar.empty:
                    ax.scatter(valid_tar[tar_x], valid_tar[tar_y], 
                            c='blue', marker='o', s=80, edgecolors='black', 
                            label=f'Target ({x}, {y})', zorder=11)

            # 3. ズレの可視化 (Displacement Line)
            # 「結果」「ターゲット」の両方のカラムが存在し、かつ show_displacement が True の場合
            if show_displacement and (res_x in pso_df.columns and res_y in pso_df.columns and has_targets):
                # 全ての要素が揃っている行だけを対象に線を引く
                valid_pairs = pso_df.dropna(subset=[res_x, res_y, tar_x, tar_y])
                for _, row in valid_pairs.iterrows():
                    ax.plot([row[tar_x], row[res_x]], 
                            [row[tar_y], row[res_y]], 
                            'k--', alpha=0.4, lw=1, zorder=10)

            return ax

    # --- 5. 3Dレイヤー ---
    def plot_3d(self, x='vfe', y='energy', z='ig', ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        3次元空間でのプロット
        """
        ax = self._get_ax(ax, figsize=(10, 8), projection='3d')
        sc = ax.scatter(self.df[x], self.df[y], self.df[z], 
                        c=self.df[z], cmap='viridis', s=40, alpha=0.8)
        
        ax.set_xlabel(x.upper())
        ax.set_ylabel(y.upper())
        ax.set_zlabel(z.upper())
        plt.colorbar(sc, ax=ax, label=z.upper(), pad=0.1)
        return ax

    # --- 6. 仕上げ ---
    def finalize(self, ax: plt.Axes, title: str, save_path: Optional[str] = None):
        ax.set_title(title, fontsize=14, pad=15)
        ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ Saved: {save_path}")
        return ax
    



def stimuli_to_target_df(targets_list, folder_path, ext=".pkl"):
    rows = []
    
    for target_id, targets in targets_list.items():
        try:
            # 1. ファイルの読み込み
            # target_id が 'stimuli_001' なら 'stimuli_001.pkl' を探す
            file_name = target_id if target_id.endswith(ext) else f"{target_id+ext}"
            full_path = os.path.join(folder_path, file_name)
            
            if not os.path.exists(full_path):
                print(f"Skip: {full_path} does not exist.")
                continue

            with open(full_path, 'rb') as f:
                # クラス定義が読み込み環境にある前提
                result = pickle.load(f) 
            
            # 2. データの抽出
            metrics = result.final_metrics
            summary = result.optimization_summary
            
            # 1行分のデータを作成
            row = {
                'target_id': target_id,
                # --- PSOが到達した結果 (Achieved) ---
                'ig': metrics.get('ig'),
                'energy': metrics.get('energy'),
                'vfe': metrics.get('vfe'),
                
                # --- 目標としていた値 (Target) ---
                'target_ig': targets.get('ig', {}).get('target'),
                'target_energy': targets.get('energy', {}).get('target'),
                'target_vfe': targets.get('vfe', {}).get('target'),
                
                # --- 分析に役立つ付加情報 ---
                'success': summary.get('success'),
                'nit': summary.get('nit', summary.get('n_actual_iterations')), # PSO/Scipy両対応
                'best_cost': result.best_cost,
                'robot_model': result.robot_model_key,
                'optimizer': result.optimizer_type
            }
            rows.append(row)
            
        except Exception as e:
            print(f"Error loading {target_id}: {e}")
            continue
            
    # 全ての行をまとめて DataFrame に変換
    return pd.DataFrame(rows)



def load_simulation_data(sim_dir_path):
    root = Path(sim_dir_path)
    data = {
        "config": {},
        "landscape": {},
        "stimuli": {}
    }

    # --- 1. Config フォルダのロード ---
    config_dir = root / "config"
    if config_dir.exists():
        # JSON
        json_path = config_dir / "config.json"
        if json_path.exists():
            data["config"]["config_json"] = json.loads(json_path.read_text())

        # Pickle系 (クラス定義変更に強い順にトライ)
        pkl_files = {
            "robot": "robot.pkl",
            "agent": "agent.pkl",
            "const_beliefs": "const_beliefs_result.pkl",
            "optimizer_state": "optimizer_state.pkl",
            "used_optimizer": "used_optimizer.pkl"
        }

        for key, file_name in pkl_files.items():
            file_path = config_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        data["config"][key] = pickle.load(f)
                except (AttributeError, ImportError, ModuleNotFoundError) as e:
                    print(f"⚠️ Warning: {file_name} のロードに失敗しました (クラス定義の変更が原因かもしれません): {e}")
                    data["config"][key] = None

    # --- 2. Landscape フォルダのロード ---
    landscape_dir = root / "landscape"
    if landscape_dir.exists():
        # CSV
        csv_path = landscape_dir / "simulation_results.csv"
        if csv_path.exists():
            data["landscape"]["df"] = pd.read_csv(csv_path)

        # Raw Data (Pickle)
        raw_path = landscape_dir / "raw_data.pkl"
        if raw_path.exists():
            with open(raw_path, 'rb') as f:
                data["landscape"]["raw_data"] = pickle.load(f)

    # --- 3. Stimuli フォルダのロード (サブフォルダ含む) ---
    stimuli_dir = root / "stimuli"
    if stimuli_dir.exists():
        for sub_dir in stimuli_dir.iterdir():
            if sub_dir.is_dir():
                target_json = sub_dir / "target.json"
                if target_json.exists():
                    data["stimuli"][sub_dir.name] = json.loads(target_json.read_text())

    return data
