# base_llm_answer_generator.py (Adapted for Google Colab Environment using Ollama)

import os
import json
# --- 使用するLLMライブラリをインポート ---
# Ollama を使用する場合
import ollama

# --- 設定 ---
# Ollamaは通常、ローカル実行ではAPIキーは不要です。
# Colab環境でOllamaサーバーを起動し、モデルをプルする必要があります（事前準備ステップ参照）。

# 使用するOllamaモデル名を選択
# Colab上で Ollama pull 済み のモデルを指定してください (例: 'llama3', 'mistral', 'gemma', 'codeqwen' など)
LLM_MODEL_NAME = "deepseek-r1:1.5b" # <-- ここに使用するOllamaモデル名を設定

# Ollamaサーバーのアドレス（Colab上でデフォルトはlocalhost）
# 異なるアドレスでOllamaを実行している場合はここを変更
OLLAMA_HOST = "http://localhost:11434"

# 入力/出力ファイルのパス (assuming script is run from the 'code' directory after cloning)
# リポジトリをクローン後、codeディレクトリに移動して実行することを想定しています。
QUESTIONS_FILE = os.path.join("..", "questions", "questions.txt")
OUTPUT_DIR = os.path.join("..", "results") # Directory to save results
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_answers_base_llm.md") # Base LLM results file


# --- LLMクライアントの初期化 ---
# Ollama クライアントの初期化
try:
    # Colab上で Ollama serve が起動している必要があります
    llm_client = ollama.Client(host=OLLAMA_HOST)
    # モデルが存在するか簡単な確認 (pingの代わり、存在しないとここでエラーになる)
    llm_client.show(LLM_MODEL_NAME) # モデル情報表示は不要だが、存在確認になる
    print(f"Colab + Ollama 環境用 LLMクライアントの初期化に成功しました。使用モデル: {LLM_MODEL_NAME}")
except ollama.ResponseError as e:
     print(f"エラー: Ollamaクライアントまたはモデル ({LLM_MODEL_NAME}) の初期化に失敗しました。")
     print(f"Ollamaサーバーが '{OLLAMA_HOST}' で起動しているか、指定モデルがインストール済みか確認してください。")
     print(f"詳細: {e}")
     exit()
except Exception as e:
    print(f"エラー: Ollamaクライアントの初期化中に予期せぬエラーが発生しました。")
    print(f"詳細: {e}")
    exit()


# --- ベースLLMから回答を取得する関数 ---
def get_base_llm_answer(question: str) -> str:
    """
    設定されたOllamaモデルを使用して、指定された質問に対する回答を生成します。
    RAGは使用しません。

    Args:
        question: 質問文字列。

    Returns:
        生成された回答文字列、または呼び出しに失敗した場合はエラーメッセージ。
    """
    try:
        # Ollama chat completions API を使用
        response = llm_client.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {'role': 'user', 'content': question},
            ],
            # stream=False # デフォルトはFalseなので省略可
            # オプション例:
            # options={
            #     'temperature': 0.8,
            #     'num_ctx': 4096,
            # }
        )
        answer = response['message']['content']

        print(f"--- 質問 ---")
        print(question)
        print(f"--- ベースLLM回答 (先頭のみ表示) ---")
        # 回答が長い場合は省略して表示し、改行をスペースに置換
        display_answer = answer.replace('\n', ' ')
        print(display_answer[:150] + "..." if len(display_answer) > 150 else display_answer)
        print("-" * 30)

        return answer

    except ollama.ResponseError as e:
        print(f"エラー: Ollamaモデル '{LLM_MODEL_NAME}' 呼び出し中にエラーが発生しました。")
        print(f"モデル名が正しいか、Ollamaにインストールされているか確認してください。")
        print(f"詳細: {e}")
        return f"【エラー】モデル呼び出しエラー: {e}"
    except Exception as e:
        print(f"エラー: 質問 '{question[:50]}...' に対する回答生成中に予期せぬエラーが発生しました。")
        print(f"詳細: {e}")
        return f"【エラー】回答生成中にエラーが発生しました: {e}"


# --- メイン処理 ---
if __name__ == "__main__":
    print(f"Colab + Ollama 環境で実行中...")
    print(f"カレントディレクトリ: {os.getcwd()}") # Colabでのカレントディレクトリ確認用

    print(f"'{QUESTIONS_FILE}' から質問を読み込んでいます...")
    questions = []
    try:
        # utf-8 エンコーディングでファイルを開く
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            # 空行を除外して各行を質問として読み込む
            questions = [line.strip() for line in f if line.strip()]
        if not questions:
            print("エラー: 質問ファイルに有効な質問が見つかりませんでした。")
            print(f"パスを確認してください: {QUESTIONS_FILE}")
            # Colabの場合、ls ../questions/ や !cat ../questions/questions.txt で存在確認可能
            exit()
        print(f"{len(questions)} 件の質問を読み込みました。")

    except FileNotFoundError:
        print(f"エラー: 質問ファイルが見つかりません。パスを確認してください: {QUESTIONS_FILE}")
        print(f"Colabの場合、カレントディレクトリを確認し、%cd コマンドで移動してください。")
        exit()
    except Exception as e:
        print(f"エラー: 質問ファイルの読み込み中にエラーが発生しました: {e}")
        exit()


    print(f"Ollama ({LLM_MODEL_NAME}) を使用して回答を生成します...")
    all_answers = {}

    # 各質問に対して回答を生成
    for i, question in enumerate(questions):
        print(f"({i+1}/{len(questions)}) 質問処理中...")
        answer = get_base_llm_answer(question)
        all_answers[question] = answer # 質問をキーとして回答を保存

    print(f"\n回答生成が完了しました。")

    # --- 結果の保存 ---
    print(f"生成された回答を '{OUTPUT_FILE}' に保存します...")
    try:
        # results ディレクトリが存在しない場合は作成
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# ベースLLM (Colab + Ollama) による回答 (RAGなし)\n\n")
            f.write(f"使用Ollamaモデル: {LLM_MODEL_NAME}\n")
            f.write(f"Ollama Host: {OLLAMA_HOST}\n\n")
            f.write("このファイルには、RAG（検索拡張生成）を使用せず、指定されたOllamaモデルの知識のみに基づいて生成された回答が格納されています。\n\n")
            f.write("---\n\n") # セパレーター

            for i, question in enumerate(questions):
                f.write(f"## 質問 {i+1}\n\n")
                f.write(f"**質問:** {question}\n\n")
                f.write(f"**ベースLLM回答:**\n{all_answers.get(question, '【エラー】回答が取得できませんでした')}\n\n")
                f.write("---\n\n") # セパレーター

        print(f"回答が正常に保存されました: {OUTPUT_FILE}")

    except Exception as e:
        print(f"エラー: 回答の保存中にエラーが発生しました: {e}")