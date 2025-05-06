# LLM RAG 実装課題

## 概要

本リポジトリは、LLM（大規模言語モデル）におけるRAG（検索拡張生成）の仕組みを理解し、実装・評価を行うための課題の成果物です。具体的には、提供された架空の参照資料に基づき、RAGなしのBase LLMとRAGありのシステムを実装し、その回答性能を比較評価します。評価には、別のLLM（Google Gemini Flash）を用いた自動評価を行います。

## リポジトリ構成

```
.
├── code/
│   ├── base_llm_answer_generator.py   # RAGなしBase LLMによる回答生成スクリプト
│   ├── rag_answer_generator.py        # RAGによる回答生成スクリプト
│   ├── evaluate_answers.py            # 回答評価スクリプト
│   └── (その他のHelperファイルなど)
├── questions/
│   ├── questions.txt                  # 評価用質問リスト
│   └── correct_answers.json           # 評価用質問に対する正解データ (JSON形式)
├── references/
│   └── (参照資料ファイル群 - .txt, .mdなど)
├── results/
│   ├── raw_answers_base_llm.md      # Base LLMによる生成回答 (実行後に生成)
│   ├── raw_answers_rag.md           # RAGによる生成回答 (実行後に生成)
│   └── evaluation_results.csv       # 回答評価結果 (実行後に生成)
├── HOMEWORK.md                      # 課題レポート本体（分析、考察、改善案など）
├── README.md                        # このファイル
└── Run_Experiments_Colab.ipynb      # 実験実行用 Colab ノートブックのエントリポイント
```

## 必要要件

* Git
* Python 3.7+
* Python ライブラリ (詳細は Colab ノートブックのセットアップ手順参照)
    * `ollama`
    * `sentence-transformers`
    * `numpy`
    * `scikit-learn`
    * `pandas`
    * `google-generativeai`
* Ollama (ローカルまたはColab上で動作させるための環境)
* Google Gemini API Key (評価者LLM用)
* Colab 環境 (推奨)

## セットアップと実行手順 (Colab での実行を推奨)

本リポジトリの実験は、リポジトリのルートにある **`Run_Experiments_Colab.ipynb`** ノートブックを Colab で開いて順番にセルを実行することで進められます。以下の手順は、ノートブック内で実行される主な内容と、事前に行うべき設定について説明しています。

1.  **Colab でノートブックを開く:**
    GitHub 上の本リポジトリページにアクセスし、`Run_Experiments_Colab.ipynb` ファイルをクリックします。ファイルの内容表示画面で、「Open in Colab」ボタンをクリックすると Colab でノートブックが開きます。

2.  **リポジトリのクローンとディレクトリ移動:**
    ノートブックの最初のコードセルを実行し、本リポジトリを Colab 環境にクローンし、`code` ディレクトリへ移動します。
    ```python
    # @title 1. Git リポジトリのクローンと移動
    # ... (リポジトリのクローンとディレクトリ移動のコード) ...
    ```
    * もしクローン時に `destination path 'LLM_RAG_Homework' already exists` エラーが出た場合は、`!rm -rf /content/LLM_RAG_Homework` コマンドで既存のディレクトリを削除してから再度クローンしてください。

3.  **Ollama 環境のセットアップ:**
    ノートブックの「2. Ollama 環境のセットアップ」セクションのセルを順番に実行します。これにより、Colab VM 上に Ollama サーバーが起動し、実験で使用するOllamaモデル（例: `deepseek-r1:1.5b` や `llama3` など）がプル（ダウンロード）されます。
    * 使用したいOllamaモデル名は、ノートブックの該当セル (`!ollama pull ...`) および回答生成スクリプト実行セル (`!python ... --model_name ...`) で指定してください。

4.  **必要な Python ライブラリのインストール:**
    ノートブックの「3. スクリプトが必要とする Python ライブラリのインストール」セルの `!pip install ...` コマンドを実行し、必要なライブラリをインストールします。

5.  **Google Gemini API キーの設定 (評価に必須):**
    * Google Gemini API キーを事前に取得しておきます。
    * Colab ノートブックを開いている状態で、左サイドバーにある**鍵アイコン（シークレット）**をクリックします。
    * 「シークレットを追加」をクリックし、**名前**に `EVALUATOR_GOOGLE_API_KEY` 、**値**に取得した Gemini API キーを正確に貼り付けます。
    * 「ノートブックでのアクセスを許可」が**オン**になっていることを確認し、保存します。
    * **注意:** APIキーをコードファイル自体に書き込まないでください。

6.  **Base LLM 回答生成スクリプトの実行:**
    ノートブックの「4. Base LLM (RAGなし) スクリプトの実行」セクションのセルを実行します。`--model_name` 引数で、使用するOllamaモデル名を指定してください。
    ```python
    # @title 4.1. base_llm_answer_generator.py スクリプトの実行
    # ... (スクリプト実行コード) ...
    ```
    * 生成された回答は `../results/raw_answers_base_llm.md` に保存されます。

7.  **RAG 回答生成スクリプトの実行:**
    ノートブックの「5. RAG あり スクリプトの実行」セクションのセルを実行します。`--model_name` 引数で、使用するOllamaモデル名を指定してください。
    ```python
    # @title 5.1. rag_answer_generator.py スクリプトの実行
    # ... (スクリプト実行コード) ...
    ```
    * 生成された回答は `../results/raw_answers_rag.md` に保存されます。

8.  **評価スクリプトの実行:**
    ノートブックの「6. 回答の評価 (Gemini Flash 評価者)」セクションのセルを実行します。ステップ5でAPIキーが正しく設定されている必要があります。
    ```python
    # @title 6.1. evaluate_answers.py スクリプトの実行
    !python evaluate_answers.py
    ```
    * 評価結果は `../results/evaluation_results.csv` に保存されます。
    * APIのレート制限に起因するエラー（`429 Resource Exhausted` など）が頻繁に発生する場合は、`code/evaluate_answers.py` スクリプト内の待機時間 (`API_CALL_DELAY`, `DELAY_BETWEEN_QUESTIONS`) やリトライ設定 (`MAX_RETRIES`, `RETRY_DELAY_BASE`) を調整してみてください。

9.  **結果の確認とダウンロード:**
    ノートブックの「7. 生成された回答と評価結果の確認」セクションで、生成された回答ファイルや評価結果CSVの内容を Colab 上で確認できます。Colab のファイルブラウザ (`../results/` ディレクトリ) から、これらのファイルをダウンロードすることも可能です。

## 成果物

実験の実行によって生成される主な成果物は以下のファイルです。

* `results/raw_answers_base_llm.md`: Base LLM による質問への回答
* `results/raw_answers_rag.md`: RAG システムによる質問への回答
* `results/evaluation_results.csv`: Base LLM と RAG の回答に対する自動評価結果

これらの結果の分析、考察、および発展的な改善案については、リポジトリルートにある **`HOMEWORK.md`** ファイルを参照してください。


## 著者

ikedatakayasu
