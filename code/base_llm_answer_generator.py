# base_llm_answer_generator.py (Adapted for Colab Environment using Ollama - Argument Version)

import os
import json
import argparse # コマンドライン引数処理のためにインポート

# --- 使用するLLMライブラリをインポート ---
# Ollama を使用する場合
import ollama

# --- 設定 ---
# Ollama Host (assuming it's running in the Colab VM)
OLLAMA_HOST = "http://localhost:11434"

# LLM Model Name は引数で指定されるようになりました

# 入力/出力ファイルのパス (assuming script is run from the 'code' directory after cloning)
# リポジトリをクローン後、codeディレクトリに移動して実行することを想定しています。
REPO_ROOT = os.path.join("..") # Go up from 'code' to repository root
QUESTIONS_FILE = os.path.join(REPO_ROOT, "questions", "questions.txt")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results") # Directory to save results
# 出力ファイル名にモデル名を含めると、どのモデルで生成した結果か分かりやすくなります（任意）
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"raw_answers_base_llm_{LLM_MODEL_NAME.replace(':', '_')}.md")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_answers_base_llm.md") # シンプルなファイル名

# --- LLM Client Initialization ---
def initialize_ollama_client(host: str, model_name: str):
    """Initializes and tests the Ollama client."""
    try:
        client = ollama.Client(host=host)
        # Test connection and model existence
        client.generate(model=model_name, prompt='test', stream=False) # generateを試す
        print(f"Ollama client initialized successfully for model '{model_name}' at {host}.")
        return client
    except ollama.ResponseError as e:
         print(f"エラー: Ollamaクライアントまたはモデル ('{model_name}') の初期化に失敗しました。")
         print(f"Ollamaサーバーが '{host}' で起動しているか、指定モデルがインストール済みか確認してください。")
         print(f"詳細: {e}")
         return None # Return None if initialization fails
    except Exception as e:
        print(f"エラー: Ollamaクライアントの初期化中に予期せぬエラーが発生しました。")
        print(f"詳細: {e}")
        return None

# --- ベースLLMから回答を取得する関数 ---
# この関数自体はモデル名を直接受け取らず、初期化済みのクライアントを使います
def get_base_llm_answer(question: str, llm_client, model_name: str) -> str:
    """
    Generates an answer for a given question using the configured base LLM.

    Args:
        question: The question string.
        llm_client: The initialized Ollama client.
        model_name: The name of the LLM model being used (for logging/error messages).

    Returns:
        The generated answer string, or an error message if the call fails.
    """
    try:
        response = llm_client.chat(
            model=model_name, # 初期化時のモデル名を使用
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
        print(f"--- ベースLLM回答 ({model_name} / 先頭のみ表示) ---")
        # 回答が長い場合は省略して表示し、改行をスペースに置換
        display_answer = answer.replace('\n', ' ')
        print(display_answer[:150] + "..." if len(display_answer) > 150 else display_answer)
        print("-" * 30)

        return answer

    except ollama.ResponseError as e:
        print(f"エラー: Ollamaモデル '{model_name}' 呼び出し中にエラーが発生しました。")
        print(f"モデル名が正しいか、Ollamaにインストールされているか確認してください。")
        print(f"詳細: {e}")
        return f"【エラー】モデル呼び出しエラー: {e}"
    except Exception as e:
        print(f"エラー: 質問 '{question[:50]}...' に対する回答生成中に予期せぬエラーが発生しました。")
        print(f"詳細: {e}")
        return f"【エラー】回答生成中にエラーが発生しました: {e}"


# --- メイン処理 ---
if __name__ == "__main__":
    # コマンドライン引数の定義とパース
    parser = argparse.ArgumentParser(description='Generate answers using a base Ollama LLM.')
    parser.add_argument('--model_name', type=str, required=True, help='Ollama model name to use for generation.')
    args = parser.parse_args()

    llm_model_name = args.model_name # 引数からモデル名を取得

    print(f"Colab + Ollama 環境でBase LLM回答生成スクリプトを実行中...")
    print(f"カレントディレクトリ: {os.getcwd()}") # Colabでのカレントディレクトリ確認用
    print(f"使用モデル（引数より）: {llm_model_name}")


    # --- LLMクライアントの初期化 ---
    print("LLMクライアントを初期化します...")
    # 引数で受け取ったモデル名でクライアントを初期化
    ollama_llm_client = initialize_ollama_client(OLLAMA_HOST, llm_model_name)
    if ollama_llm_client is None:
        print("エラー: LLMクライアントの初期化に失敗したため、処理を中断します。")
        exit()


    # --- 質問の読み込み ---
    print(f"'{QUESTIONS_FILE}' から質問を読み込んでいます...")
    questions = []
    try:
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        if not questions:
            print("エラー: 質問ファイルに有効な質問が見つかりませんでした。")
            print(f"パスを確認してください: {QUESTIONS_FILE}")
            exit()
        print(f"{len(questions)} 件の質問を読み込みました。")

    except FileNotFoundError:
        print(f"エラー: 質問ファイルが見つかりません。パスを確認してください: {QUESTIONS_FILE}")
        print(f"Colabの場合、カレントディレクトリを確認し、%cd コマンドで移動してください。")
        exit()
    except Exception as e:
        print(f"エラー: 質問ファイルの読み込み中にエラーが発生しました: {e}")
        exit()


    print(f"Ollama ({llm_model_name}) を使用して回答を生成します...")
    all_answers = {}

    # 各質問に対して回答を生成
    for i, question in enumerate(questions):
        print(f"({i+1}/{len(questions)}) 質問処理中...")
        # モデル名を get_base_llm_answer 関数に渡す
        answer = get_base_llm_answer(question, ollama_llm_client, llm_model_name)
        all_answers[question] = answer # 質問をキーとして回答を保存

    print(f"\n回答生成が完了しました。")

    # --- 結果の保存 ---
    # 出力ファイル名にモデル名を含める場合はここでもモデル名を使う
    # OUTPUT_FILE_WITH_MODEL = os.path.join(OUTPUT_DIR, f"raw_answers_base_llm_{llm_model_name.replace(':', '_')}.md")
    print(f"生成された回答を '{OUTPUT_FILE}' に保存します...")
    try:
        # results ディレクトリが存在しない場合は作成
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# ベースLLM (Colab + Ollama - モデル: {llm_model_name}) による回答 (RAGなし)\n\n")
            f.write(f"使用Ollamaモデル: {llm_model_name}\n")
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