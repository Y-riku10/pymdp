# flaskr/main.py
from flaskr import app
from flask import Flask, render_template, request, session, redirect, url_for
import yaml, os, random

app.secret_key = "secret_key_for_session"

# 設問読み込み（5問固定）
config_path = os.path.join(os.path.dirname(__file__), "config", "questions.yml")
with open(config_path, "r", encoding="utf-8") as f:
    questions = yaml.safe_load(f)["questions"]

# 動画一覧
videos = [
    "static/video/sample.gif", # テスト用
    "static/video/move1.gif", # ... 追加
    "static/video/move2.gif", # ... 追加
    "static/video/move3.gif", # ... 追加
    "static/video/move4.gif", # ... 追加
    # ... 追加
]

@app.route("/", methods=["GET"])
def start():
    # ランダム順序などはここで設定可能
    session.clear()
    session["video_order"] = random.sample(videos, len(videos))
    session["current"] = 0
    session["answers"] = []
    return redirect(url_for("question_page"))

@app.route("/question", methods=["GET", "POST"])
def question_page():
    current = session.get("current", 0)
    video_order = session.get("video_order", videos)
    answers_all = session.get("answers", [])

    if request.method == "POST":
        answers = []
        for q in questions:
            ans = request.form.get(q["id"])
            if not ans:
                message = "すべての質問に回答してください"
                return render_template("question.html",
                                       video=video_order[current],
                                       qnum=current+1,
                                       total=len(video_order),
                                       questions=questions,
                                       message=message)
            answers.append(ans)
        answers_all.append({"video": video_order[current], "answers": answers})
        session["answers"] = answers_all
        session["current"] = current + 1
        if current + 1 >= len(video_order):
            return redirect(url_for("finish"))
        return redirect(url_for("question_page"))

    # GET 時も question.html を使用
    return render_template("question.html",
                           video=video_order[current],
                           qnum=current+1,
                           total=len(video_order),
                           questions=questions,
                           message=None)

@app.route("/finish")
def finish():
    answers = session.get("answers", [])
    print("全回答結果:", answers)
    return "回答ありがとうございました！"


# import yaml, os
# from flaskr import app
# from flask import render_template, request

# # YAMLファイルの読み込み
# config_path = os.path.join(os.path.dirname(__file__), "config", "questions.yml")
# with open(config_path, "r", encoding="utf-8") as f:
#     questions = yaml.safe_load(f)["questions"]

# @app.route("/", methods=["GET", "POST"])
# def index():
#     message = None
#     if request.method == "POST":
#         # フォームの内容を受け取る
#         results = {q["id"]: request.form.get(q["id"]) for q in questions}
#         print(results)  # ターミナルに出力（必要に応じて保存）

#         message = "回答完了"

#     return render_template("index.html", message=message, questions=questions)

# if __name__ == "__main__":
#     app.run(debug=True)
