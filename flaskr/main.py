from flask import Flask, render_template, request, session, redirect, url_for
import yaml
import os
import random
import time
import pandas as pd
from flaskr import app

import threading
import webbrowser

# サーバーが起動してからブラウザを開くまでの遅延時間（秒）を設定
# サーバーの準備が整うのを待つため、1秒程度の遅延を入れるのが一般的です。
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/') # サーバーのURLを指定

# 別のスレッドでタイマーを設定し、指定時間後にブラウザを開く関数を実行
# debug=True（リローダー機能）を使用している場合、この処理が二重に実行されないよう注意が必要です。
# 以下の例では、Timerを使用することで、メインのアプリケーション実行とは別のスレッドで実行しています。
# サーバーが起動してからブラウザを開くため、通常はthreading.Timerを使用するのが最も確実です。
threading.Timer(1.0, open_browser).start()

# 練習試行回数を定数として定義
PRACTICE_TRIALS_PER_RUN = 2 

# 注意: 本番環境ではより複雑な秘密鍵を使用してください
app.secret_key = "very_secure_secret_key_for_psychology_experiment" 

# 設問データ格納用
QUESTIONS_ALL = {}
MOTION_EVAL_QUESTIONS = []
CURIOSITY_SCALE_QUESTIONS = []
# 回答インデックスとデータキーを対応させるためのマップ
# QUESTION_ID_MAP = {} 
# 刺激データ格納用
ALL_STIMULI = {}

# フォルダパス
BASE_FOLDER = 'static/video/stimuli'


# --- データの読み込み ---
def load_questions():
    """質問ファイルを読み込み、質問リストとデータキーマップを生成し、英語レポートを標準出力に表示する。"""
    global MOTION_EVAL_QUESTIONS, CURIOSITY_SCALE_QUESTIONS#, QUESTION_ID_MAP
    
    config_path = os.path.join(app.root_path, "config", "questions.yml")
    
    print("\n=== [Question Loading Report] ===")
    print(f"📄 Target configuration file: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            
            # --- 動きの評価質問 ---
            MOTION_EVAL_QUESTIONS = [
                q for q in config_data.get("questions", []) 
                if q.get('category') == 'Motion_Eval'
            ]
            
            # --- 好奇心スケール質問 ---
            CURIOSITY_SCALE_QUESTIONS = [
                q for q in config_data.get("questions_curiosity", []) 
                if q.get('category') == 'Curiosity_Scale'
            ]
            
            # # --- 質問IDマップ生成 ---
            # QUESTION_ID_MAP = {
            #     i: q['data_key'] for i, q in enumerate(MOTION_EVAL_QUESTIONS)
            # }

            # --- レポート出力 ---
            print("\n🧩 Motion Evaluation Questions:")
            if MOTION_EVAL_QUESTIONS:
                print(f" ├─ Loaded {len(MOTION_EVAL_QUESTIONS)} questions.")
                for q in MOTION_EVAL_QUESTIONS:
                    print(f" │   - ID: {q.get('id')} | Key: {q.get('data_key')} | Text: {q.get('text')}")
            else:
                print(" ├─ ⚠️ No Motion_Eval questions found.")
            
            print("\n🔍 Curiosity Scale Questions:")
            if CURIOSITY_SCALE_QUESTIONS:
                print(f" ├─ Loaded {len(CURIOSITY_SCALE_QUESTIONS)} questions.")
                for q in CURIOSITY_SCALE_QUESTIONS[:5]:  # 先頭5件のみ表示（多い場合）
                    print(f" │   - ID: {q.get('id')} | Text: {q.get('text')}")
                if len(CURIOSITY_SCALE_QUESTIONS) > 5:
                    print(f" │   ... ({len(CURIOSITY_SCALE_QUESTIONS)-5} more)")
            else:
                print(" ├─ ⚠️ No Curiosity_Scale questions found.")

            # print("\n🗺️  Generated Question ID Map:")
            # for i, key in QUESTION_ID_MAP.items():
            #     print(f" │   {i}: {key}")
            
            print("\n✅ Question loading completed successfully.")
            print("==========================================\n")
            
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {config_path}")
    except yaml.YAMLError as ye:
        print(f"❌ YAML parsing error in {config_path}: {ye}")
    except Exception as e:
        print(f"❌ Unexpected error while loading questions: {e}")

def load_all_stimuli():
    """刺激フォルダ構造に基づいて、メイン刺激と基準刺激の情報を読み込み、構成を標準出力に表示する。"""
    global ALL_STIMULI
    ALL_STIMULI = {
        'main': {},      # {'type1': [{'id':..., 'path':...}, ...], 'practice': [...]}
        'standards': {}  # {'type1': {'standard': '/path/a.gif', 'chaotic': '/path/b.gif'}, ...}
    }

    print("\n=== [Stimuli Loading Report] ===")

    # practice, type1, type2 ... の type_id を統一的に扱う
    for type_id in ['practice', 'type1', 'type2']:
        base_dir = os.path.join(app.root_path, BASE_FOLDER, type_id)
        if not os.path.exists(base_dir):
            print(f"⚠️  Skipped '{type_id}' — folder not found: {base_dir}")
            continue

        print(f"\n📂 Loading stimuli for block: {type_id}")
        print(f" └ Base directory: {base_dir}")

        # 1. メイン/練習刺激 (mainフォルダ)
        main_path = os.path.join(base_dir, 'main')
        if os.path.exists(main_path):
            main_files = [f for f in os.listdir(main_path) if f.endswith('.mp4')]
            ALL_STIMULI['main'][type_id] = [
                {'id': f.split('.')[0], 'path': f'/{BASE_FOLDER}/{type_id}/main/{f}', 'type': type_id}
                for f in main_files
            ]
            print(f"   ├─ Main stimuli ({len(main_files)} files):")
            for f in main_files:
                print(f"   │   - {f}")
        else:
            ALL_STIMULI['main'][type_id] = []
            print(f"   ├─ Main folder not found.")

        # 2. 基準刺激 (standard / chaotic)
        standards_data = {}
        for category in ['standard']:#, 'chaotic']:
            cat_dir = os.path.join(base_dir, category)
            if os.path.exists(cat_dir):
                gif_files = [f for f in os.listdir(cat_dir) if f.endswith('.mp4')]
                if gif_files:
                    file = gif_files[0]  # 最初のGIFを採用
                    standards_data[category] = f'/{BASE_FOLDER}/{type_id}/{category}/{file}'
                    print(f"   ├─ {category.capitalize()} stimulus: {file}")
                else:
                    print(f"   ├─ {category.capitalize()} folder found but no GIFs inside.")
            else:
                print(f"   ├─ {category.capitalize()} folder not found.")

        ALL_STIMULI['standards'][type_id] = standards_data

        # summary
        if not standards_data:
            print(f"   ⚠️  No standard/chaotic stimuli loaded for {type_id}")

    print("\n✅ Stimuli loading completed.")
    print("===============================\n")


# アプリケーション起動時にデータを読み込む
with app.app_context():
    load_questions()
    load_all_stimuli()

# --- ヘルパー関数 ---

def create_block_stimuli_list(block_id, seed, is_practice=False):
    """特定のブロックIDの試行リストを生成・シャッフルする。"""
    stimuli = ALL_STIMULI['main'].get(block_id, [])[:]
    
    # if is_practice:
    #     # 練習は最初のN個だけを使用
    #     stimuli = stimuli[:PRACTICE_TRIALS_PER_RUN]
        
    random.seed(seed)
    random.shuffle(stimuli)
    
    return stimuli

def get_current_trial_data(stimuli_map, current_block, current_trial_index):
    """現在のブロックとインデックスに基づいて、試行に必要なデータを取得する。"""
    # ブロック内の刺激リスト
    block_list = stimuli_map.get(current_block, [])
    # ブロック中の処理
    if current_trial_index < len(block_list):
        stim = block_list[current_trial_index]
        #　小文字に
        standards_key = current_block.lower() if current_block != 'PRACTICE' else 'practice'
        standards = ALL_STIMULI['standards'].get(standards_key, {})
        
        # 質問のシャッフル順序をセッションから取得
        questions = session.get('shuffled_questions', [])

        return {
            'stim': stim,
            'standards': standards,
            'questions': questions,
            'current_trial_num': current_trial_index + 1,
            'total_trials_in_block': len(block_list)
        }
    # ブロックの終端ではNoneを返す
    return None

def initialize_session(seed):

    # --- セッション初期化 ---
    session.clear()
    session["subject_id"] = seed
    session["seed"] = seed
    session["start_time"] = time.time()
    
    # データ保存用リストを分離
    session['trial_responses'] = [] 
    session['survey_responses'] = []
    
    # フェーズ管理変数の初期設定
    session['current_block'] = 'INSTRUCTION' # ブロック
    session['current_trial_index'] = 0 # ブロック内のインデックス
    session['trial_num'] = 1 # 全ての試行数
    session['all_trial_num'] = 30

    # --- 質問のシャッフルと保存 ---
    shuffled_questions = MOTION_EVAL_QUESTIONS[:]
    random.seed(seed)
    random.shuffle(shuffled_questions)
    session['shuffled_questions'] = shuffled_questions

    # --- デバッグ出力: シャッフル後の質問リスト ---
    print("=== デバッグ: シャッフル後の質問リスト ===")
    for i, q in enumerate(shuffled_questions, 1):
        print(f"{i}. {q}")

    # --- 刺激順序の条件分岐とリスト生成 ---
    # 4で割った余りを使うことで、提示順(2種) × 尺度方向(2種) を網羅
    condition_code = seed % 4

    # 尺度方向の決定 (0, 1なら正方向 / 2, 3なら逆方向)
    is_reverse_scale = True if condition_code >= 2 else False
    session['is_reverse_scale'] = is_reverse_scale

    # 提示順の決定
    main_order = ['TYPE1', 'BREAK', 'TYPE2'] if condition_code % 2 == 0 else ['TYPE2', 'BREAK', 'TYPE1']
    session['block_order'] = ['INSTRUCTION', 'PRACTICE'] + main_order + ['SURVEY', 'COMPLETE']

    # --- デバッグ出力: ブロック順序 ---
    print("\n=== デバッグ: ブロック順序 ===")
    print(session['block_order'])

    # --- 全刺激リストの辞書 (stimuli_map) を生成 ---
    stimuli_map = {}
    stimuli_map['PRACTICE'] = create_block_stimuli_list('practice', seed, is_practice=True)
    stimuli_map['TYPE1'] = create_block_stimuli_list('type1', seed)
    stimuli_map['TYPE2'] = create_block_stimuli_list('type2', seed)
    session['stimuli_map'] = stimuli_map

    # --- デバッグ出力: 刺激リスト ---
    print("\n=== デバッグ: 各ブロックの刺激リスト ===")
    for block, stimuli_list in stimuli_map.items():
        print(f"\n[{block}] ({len(stimuli_list)} stimuli)")
        for s in stimuli_list:
            print(f"  - {s}")
    return None

# --- ルーティング ---

@app.route("/", methods=["GET", "POST"])
def root():
    """アプリ起動時は常にログイン画面へリダイレクトする"""
    session.clear()
    load_questions()
    load_all_stimuli()
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    """ログイン（被験者ID入力）"""
    if 'subject_id' in session:
        # 中断・再開ロジック (簡素化)
        if session.get('current_block') == 'SURVEY':
             return redirect(url_for('survey'))
        if session.get('current_block'):
            # 現在のブロックの状態に応じて適切な遷移説明ページへリダイレクト
            return redirect(url_for('progress'))
    
    # IDの入力があった場合
    if request.method == "POST":
        subject_id = request.form.get("subject_id")
        if not subject_id or not subject_id.isdigit():
            return render_template("login.html", error="参加者IDを数値で入力してください。")
            
        seed = int(subject_id)
        # セッションの初期化
        initialize_session(seed)

        return redirect(url_for("instruction"))
    
    return render_template("login.html")

@app.route("/instruction", methods=["GET", "POST"])
def instruction():
    """実験全体の説明と同意確認"""
    # 念の為
    if 'subject_id' not in session or session.get('current_block') != 'INSTRUCTION':
        return redirect(url_for('login'))
        
    # 開始ボタン
    if request.method == "POST":
        # INSTRUCTIONからPRACTICEへ遷移
        return redirect(url_for("progress"))
        
    return render_template("instruction.html")


@app.route("/experiment_prepare", methods=["GET", "POST"])
def experiment_prepare():
    """試行開始前の準備完了待機画面"""
    # 念の為
    if 'subject_id' not in session:
        return redirect(url_for('login'))
        
    current_block = session.get('current_block')
    current_index = session.get('current_trial_index')
    print(f"{current_block}, {current_index}")
    
    # trial_data = get_current_trial_data(session.get('stimuli_map', {}), current_block, current_index)
    
    # if not trial_data:
    #     # 試行リストの終端に達した場合、次のブロックへ遷移
    #     return handle_block_completion()
    
    if request.method == "POST":
        return redirect(url_for("experiment_process"))

    return render_template(
        'experiment_prepare.html',
        current_block=current_block,
        current_index=current_index,
        trial_num=session['trial_num'],
        all_trial_num=session['all_trial_num']
    )


@app.route("/experiment_process", methods=["GET", "POST"])
def experiment_process():
    """基準確認(5秒) -> メイン評価 のロジックを処理する単一ルート"""
    # 念の為
    if 'subject_id' not in session:
        return redirect(url_for('login'))

    current_block = session.get('current_block')
    current_index = session.get('current_trial_index')
    
    trial_data = get_current_trial_data(session.get('stimuli_map', {}), current_block, current_index)

    if not trial_data:
        # 試行リストの終端に達した場合、次のブロックへ遷移
        return redirect(url_for('progress'))
    
    # 刺激ごとの質問シャッフルをする
    shuffled_questions = trial_data['questions']
    # 被験者IDと試行インデックスを組み合わせて固有のSeedを作る
    random.seed(session.get('seed') + session.get('current_trial_index'))
    random.shuffle(shuffled_questions)
    trial_data['questions'] = shuffled_questions
    # 質問の順序を記録
    question_order = []
    for q in shuffled_questions:
        question_order.append(q['id'])

    # --- POST (回答記録) ---
    if request.method == 'POST':
        # 1. タイムスタンプ
        response_time_unix = time.time()
        
        # 2. 基本情報
        response_data = {
            'subject_id': session['subject_id'],
            'trial_num': current_index + 1,
            'block_id': current_block,
            'stim_id': trial_data['stim']['id'], 
            'stim_type': trial_data['stim']['type'],
            'response_time_unix': response_time_unix,
            'question_order': question_order,
            'is_reverse_scale': session['is_reverse_scale'],
        }

        # 3. 回答スコアを記録
        questions = trial_data['questions']
        for q in questions:
            id = q['id'] # q1~q5
            data_key = q['data_key'] # like, beauty, strength, interest, understand
            response_data[data_key] = request.form.get(id, 'NaN') 
            
        # 4. 評価にかかった時間 (クライアント側で計算された時間を受け取る想定)
        response_data['rt_evaluation'] = request.form.get('rt_evaluation', 'NaN')

        session['trial_responses'].append(response_data)
        
        # 次の試行へ
        session['current_trial_index'] += 1
        
        # 次の試行があるかチェック
        if session['current_trial_index'] < trial_data['total_trials_in_block']:
            # 次の試行の準備画面へ
            return redirect(url_for('experiment_prepare'))
        else:
            # ブロック終了処理へ
            return redirect(url_for('progress'))

    # --- GET (刺激提示) ---
    # GETリクエストはクライアント側のJavaScriptによって STANDARD_CHECK または MAIN_EVALUATION を表示
    return render_template(
        'experiment_process.html',
        standard_path=trial_data['standards']['standard'],
        # chaotic_path=trial_data['standards']['chaotic'],
        main_path=trial_data['stim']['path'],
        questions=trial_data['questions'],
        current_trial_index=trial_data['current_trial_num'],
        total_trials=trial_data['total_trials_in_block'],
        block_id=current_block,
        qnum=current_index,
        is_reverse_scale=session['is_reverse_scale']
    )

@app.route("/progress", methods=["GET", "POST"])
def progress():
    """実験進行の中心ハブ。現在の状態に応じて次ページを自動制御。"""
    # if 'subject_id' not in session:
    #     return redirect(url_for('login'))
    
    if request.method == 'POST':
        return redirect(url_for('experiment_prepare'))

    current_block = session.get('current_block')
    block_order = session.get('block_order', [])
    
    MAIN_BLOCKS = ['TYPE1', 'TYPE2']

    try:
        current_index = block_order.index(current_block)
        next_block = block_order[current_index + 1]
    except (ValueError, IndexError):
        next_block = 'COMPLETE'

    session['current_block'] = next_block
    session['current_trial_index'] = 0

    if next_block == 'COMPLETE':
        session['end_time'] = time.time()
    
    # --- GET（次に表示すべき画面を判断）時 ---
    current_block = session.get('current_block')

    if current_block == 'PRACTICE':
        # セッション開始画面を表示。post待ち。
        return render_template('progress.html', block_id='PRACTICE')
    elif current_block in MAIN_BLOCKS:
        # セッション開始画面を表示。post待ち。
        return render_template('progress.html', block_id=current_block)
    elif current_block == 'BREAK':
        return redirect(url_for('break_time'))
    elif current_block == 'SURVEY':
        return redirect(url_for('survey'))
    elif current_block == 'COMPLETE':
        return redirect(url_for('complete'))
    else:
        # 想定外：安全策としてcompleteへ
        print(f"[WARN] Unknown block: {current_block}")
        return redirect(url_for('complete'))



# def handle_block_completion():
#     """ブロックが終了した際の次の遷移先を決定する。"""
#     current_block = session.get('current_block')
#     block_order = session.get('block_order', [])
#     print(f"current_block: {current_block}, block_order: {block_order}")
    
#     # 1. block_order上の次のブロックを特定
#     try:
#         current_index_in_order = block_order.index(current_block)
#         next_block_in_order = block_order[current_index_in_order + 1]
#     except (ValueError, IndexError):
#         # 通常は起きない。終端に到達(block_orderの最後、'COMPLETE'の次)
#         print("Block Transition Error")
#         return redirect(url_for('complete'))

#     # 2. セッションを次のブロックに更新（暫定）
#     session['current_block'] = next_block_in_order
#     session['current_trial_index'] = 0

#     MAIN_BLOCKS = ['TYPE1', 'TYPE2']
#     # 3. 休憩の挿入判定
#     # # 最初のメインブロックが終了し、次に別のメインブロックがある場合に限り、休憩を挿入
#     # if current_block in MAIN_BLOCKS and next_block_in_order in MAIN_BLOCKS:
#     #     # TYPE1 -> TYPE2 または TYPE2 -> TYPE1 の間
#     #     session['current_block'] = 'BREAK' # セッションを'BREAK'に上書き
#     #     return redirect(url_for('break_time'))
    
#     # 4. 通常の遷移先へのリダイレクト
    
#     # PRACTICE終了後の本番開始 (告知画面へ)
#     # if current_block == 'PRACTICE' and next_block_in_order in MAIN_BLOCKS:
#     #     return redirect(url_for('transition'))
#     if next_block_in_order in MAIN_BLOCKS:
#         return redirect(url_for('transition'))
        
#     # メインブロック後半終了後のアンケート
#     elif next_block_in_order == 'SURVEY':
#         return redirect(url_for('survey'))
        
#     # 実験全体完了 (COMPLETE)
#     elif next_block_in_order == 'COMPLETE':
#         session['end_time'] = time.time()
#         return redirect(url_for('complete'))
        
#     # 予期しない遷移 (安全のためcompleteへ)
#     print(f"[DEBUG] handle_block_completion: current_block={current_block!r}")
#     return redirect(url_for('complete'))


# @app.route("/transition", methods=["GET", "POST"])
# def transition():
#     """ブロック開始前の説明/確認ページ (汎用)"""
#     # 念の為
#     if 'subject_id' not in session:
#         return redirect(url_for('login'))
    
#     # 現在の進行状況（ブロック）によってリダイレクト先を変化させる
#     current_block = session.get('current_block')

#     # ブロックが休憩の場合は /break_time にリダイレクト
#     if current_block == 'BREAK':
#         return redirect(url_for('break_time'))
#     # ブロックが SURVEY の場合は /survey にリダイレクト
#     elif current_block == 'SURVEY':
#         return redirect(url_for('survey'))
    
#     # 開始ボタンが押された後の処理
#     if request.method == "POST":
#         # ブロックが PRACTICE の場合は experiment_prepare へ
#         if current_block == 'PRACTICE':
#             return redirect(url_for('experiment_prepare'))
#         # 既に試行が始まっている場合は experiment_prepare へ
#         else:
#             return redirect(url_for('experiment_prepare'))
        
#     # GET: ブロック開始前の説明を表示
#     return render_template("transition.html", block_id=current_block)






@app.route("/break_time", methods=["GET", "POST"])
def break_time():
    """休憩時間。POSTで次のブロックへ遷移。"""
    # if session.get('current_block') != 'BREAK':
    #     # 不正な遷移を防ぐ
    #     return redirect(url_for('login'))

    if request.method == 'POST':
        # 休憩終了。
        return redirect(url_for('progress'))
        
        # try:
        #     # 'BREAK'の次にくるブロック（TYPE2 or TYPE1）を探す
        #     current_index_in_order = block_order.index('BREAK') # 'BREAK'が現在のブロックとして検索
        #     next_block = block_order[current_index_in_order + 1]
        # except (ValueError, IndexError):
        #     # 予期しないエラー
        #     return redirect(url_for('complete'))
        
        # # セッションを次の本番ブロックに更新
        # session['current_block'] = next_block
        # session['current_trial_index'] = 0
        
        # 本番ブロック開始前の告知画面へ
        # return redirect(url_for('transition'))

    # GET: 休憩画面の表示
    return render_template('break.html')


@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """好奇心スケールの回答ページ"""
    # if 'subject_id' not in session or session.get('current_block') != 'SURVEY':
    #     return redirect(url_for('login'))
    
    if request.method == 'POST':
        # アンケート回答を記録
        survey_response = {
            'subject_id': session['subject_id'],
            'survey_time_unix': time.time(),
            'survey_name': 'Curiosity_Scale',
            **request.form # 全質問の回答 (data_keyがそのままキーとして使われる想定)
        }
        
        session['survey_responses'].append(survey_response)
        
        session['current_block'] = 'COMPLETE'
        return redirect(url_for('complete'))
        
    return render_template('survey.html', questions=CURIOSITY_SCALE_QUESTIONS)


@app.route('/complete')
def complete():
    """データ保存と実験終了"""
    # if 'subject_id' not in session:
    #     return redirect(url_for('login'))
        
    session['end_time'] = time.time()
    subject_id = session['subject_id']
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    
    # --- 1. 試行応答データ (Trial Data) の保存 ---
    if session['trial_responses']:
        df_trials = pd.DataFrame(session['trial_responses'])
        trial_filename = f"data/{subject_id}_trial_data_{time_stamp}.csv"
        
        # データフレームに必要なメタデータ列を追加（最終的なデータ解析用）
        df_trials['Seed'] = session['seed']

        #　main orderはメタデータに保存することに変更
        # df_trials['Main_Order'] = str(session['block_order'][1:3]) # Type1, Type2の順序

        try:
            os.makedirs('data', exist_ok=True)
            df_trials.to_csv(trial_filename, index=False, encoding='utf-8')
        except Exception as e:
            print(f"試行データ保存エラー: {e}")

    # --- 2. メタ・アンケートデータ (Meta/Survey Data) の保存 ---
    
    # メタデータ
    meta_data = {
        'Subject_ID': subject_id,
        'Seed': session['seed'],
        'Experiment_StartTime': pd.to_datetime(session['start_time'], unit='s'),
        'Experiment_EndTime': pd.to_datetime(session['end_time'], unit='s'),
        'Block_Order': str(session['block_order']),
        'Question_Shuffle_Order': str([q['data_key'] for q in session.get('shuffled_questions', [])]),
    }
    
    # アンケートデータをメタデータに追加
    if session['survey_responses']:
        # アンケート結果は1行になるはず
        for key, value in session['survey_responses'][0].items():
            if key not in ['subject_id', 'survey_time_unix', 'survey_name']:
                 meta_data[key] = value

    df_meta = pd.DataFrame([meta_data])
    meta_filename = f"data/{subject_id}_meta_survey_{time_stamp}.csv"

    try:
        os.makedirs('data', exist_ok=True)
        df_meta.to_csv(meta_filename, index=False, encoding='utf-8')
    except Exception as e:
        print(f"メタデータ保存エラー: {e}")
    
    session.clear() 
    return render_template('complete.html')



@app.route("/command/<cmd>")
def command(cmd):
    """
    Debug command:
    /command/<cmd> で current_block を強制変更する。
    例: /command/type1 → current_block='TYPE1'
    """
    # 本番環境では無効化したい場合
    if not app.debug:
        return "Command disabled in production", 403

    # 文字を大文字に統一
    target_block = cmd.upper()

    # ここで許可されるブロック一覧
    VALID_BLOCKS = ['PRACTICE', 'TYPE1', 'TYPE2', 'BREAK', 'SURVEY', 'COMPLETE']

    if target_block not in VALID_BLOCKS:
        return f"Unknown block: {cmd}", 400

    # コマンドを使用したらデータは無効となる(subject_idは999)
    DEBUG_USER = int(-1)
    initialize_session(DEBUG_USER)
    # 指定されたブロックの一つ前のブロックをprogressに渡したいので加工
    block_order = session.get('block_order')
    current_index = block_order.index(target_block)
    prev_block = block_order[current_index - 1]


    # セッションを書き換え
    session['current_block'] = prev_block
    session['current_trial_index'] = 0  # 念のためリセット

    print(f"[DEBUG COMMAND] current_block → {target_block}")

    # 変更後は progress ハブに任せる
    return redirect(url_for("progress"))
