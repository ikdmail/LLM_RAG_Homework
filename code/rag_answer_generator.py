# rag_answer_generator.py (Adapted for Colab Environment using Ollama and sentence-transformers - Argument Version)

import os
import json
import re
import numpy as np
import argparse # コマンドライン引数処理のためにインポート

# --- Import necessary libraries ---
# Ollama for LLM
import ollama
# Sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity # For similarity calculation

# --- Configuration ---
# Ollama Host (assuming it's running in the Colab VM)
OLLAMA_HOST = "http://localhost:11434"

# LLM Model Name は引数で指定されるようになりました

# Embedding Model Name (from sentence-transformers library)
# Lightweight models are good for Colab
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Or similar lightweight model

# Paths (assuming script is run from the 'code' directory)
REPO_ROOT = os.path.join("..") # Go up from 'code' to repository root
QUESTIONS_FILE = os.path.join(REPO_ROOT, "questions", "questions.txt")
REFERENCES_DIR = os.path.join(REPO_ROOT, "references")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results")
# 出力ファイル名にモデル名を含めると、どのモデルで生成した結果か分かりやすくなります（任意）
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"raw_answers_rag_{LLM_MODEL_NAME.replace(':', '_')}.md")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_answers_rag.md") # シンプルなファイル名

# RAG Parameters
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50 # Overlap between chunks
NUM_RETRIEVED_CHUNKS = 3 # Number of top relevant chunks to retrieve

# Prompt Template for RAG
# The template instructs the LLM to use the provided context.
RAG_PROMPT_TEMPLATE = """以下の参考情報を元に、質問に回答してください。
参考情報:
{context}

質問: {question}

回答:"""

# --- RAG Components ---

def load_documents(directory: str) -> list[str]:
    """Loads all text-based files from a directory."""
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # .gitkeep などの隠しファイルやディレクトリはスキップ
        if os.path.isfile(filepath) and filename.endswith(('.txt', '.md')) and not filename.startswith('.'):
            try:
                # TODO: ファイルの実際のエンコーディングに合わせて 'utf-8' を修正
                # Windowsなどで作成したファイルの場合、'shift_jis' や 'cp932' の可能性あり
                # 最も推奨される解決策は、事前にファイルをUTF-8で保存し直すことです。
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                print(f"Loaded document: {filename}")
            except Exception as e:
                print(f"Error loading document {filename} with utf-8: {e}. Trying other encodings if necessary...")
                try:
                    # 例: UTF-8で失敗した場合にShift JISを試すフォールバック
                    with open(filepath, 'r', encoding='shift_jis') as f:
                        documents.append(f.read())
                    print(f"Loaded document: {filename} with shift_jis.")
                except Exception as e2:
                     print(f"Error loading document {filename} with shift_jis: {e2}. Skipping file.")


    return documents

def split_documents(documents: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits documents into overlapping chunks."""
    chunks = []
    # Refined simple splitter (split by paragraphs/sentences then combine with overlap)
    for doc in documents:
        # Split by double newline (paragraphs) or single newline
        paragraphs = doc.split('\n\n')
        if len(paragraphs) == 1:
            paragraphs = doc.split('\n') # If double newline doesn't work, try single

        current_chunk_lines = []
        current_chunk_len = 0

        for para in paragraphs:
            # Clean up extra whitespace
            cleaned_para = para.strip()
            if not cleaned_para: continue

            # Estimate length (using characters)
            para_len = len(cleaned_para)

            # If adding this paragraph exceeds chunk size, finalize current chunk and start new one
            # Add some buffer (e.g., 20) for safety or transition words
            if current_chunk_len + para_len > chunk_size + 20 and current_chunk_len > 0:
                 # Finalize the current chunk
                 chunk_text = " ".join(current_chunk_lines).strip()
                 if chunk_text:
                     chunks.append(chunk_text)

                 # Start a new chunk with overlap
                 # Take the last 'chunk_overlap' characters from the last chunk added
                 overlap_text = chunk_text[-chunk_overlap:].strip() if len(chunk_text) > chunk_overlap else ""
                 current_chunk_lines = [overlap_text, cleaned_para] if overlap_text else [cleaned_para]
                 current_chunk_len = len(overlap_text) + len(cleaned_para) + (1 if overlap_text else 0) # +1 for potential space
            else:
                 # Add paragraph to current chunk
                 current_chunk_lines.append(cleaned_para)
                 current_chunk_len += para_len + (1 if current_chunk_len > 0 else 0) # Add space for separation

        # Add the last chunk
        chunk_text = " ".join(current_chunk_lines).strip()
        if chunk_text:
             chunks.append(chunk_text)


    print(f"Split documents into {len(chunks)} chunks.")
    return chunks


class VectorStore:
    """Simple in-memory vector store using Sentence Transformers."""
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None # Will store embeddings as a numpy array
        self.is_indexed = False

    def add_chunks(self, chunks: list[str]):
        """Adds chunks and computes their embeddings."""
        if not chunks:
            print("No chunks to add.")
            return

        print(f"Creating embeddings for {len(chunks)} chunks using '{EMBEDDING_MODEL_NAME}'...")
        self.chunks = chunks
        # Compute embeddings
        try:
            self.embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
            self.is_indexed = True
            print("Embedding creation complete.")
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            self.embeddings = None
            self.is_indexed = False


    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieves top-k most similar chunks to the query."""
        if not self.is_indexed or self.embeddings is None or not self.chunks:
            print("Vector store is not indexed or empty.")
            return []

        # Get query embedding
        try:
            query_embedding = self.embedding_model.encode(query)
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return []


        # Calculate cosine similarity between query embedding and all chunk embeddings
        # query_embedding needs to be 2D for cosine_similarity ([1, embedding_dim])
        try:
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        except Exception as e:
             print(f"Error calculating similarity: {e}")
             return []


        # Get the indices of the top-k most similar chunks
        # argsort sorts in ascending order, so use [::-1] to get descending, then take top k
        top_k_indices = similarities.argsort()[::-1][:k]

        # Ensure indices are valid
        top_k_indices = [i for i in top_k_indices if i < len(self.chunks)]

        # Return the corresponding chunks
        retrieved_chunks = [self.chunks[i] for i in top_k_indices]

        # Optional: Print retrieved chunks for debugging
        # print("\n--- Retrieved Chunks ---")
        # for i, chunk in enumerate(retrieved_chunks):
        #     print(f"Chunk {i+1} (Score: {similarities[top_k_indices[i]]:.4f}): {chunk[:100]}...") # Print snippet
        # print("------------------------")

        return retrieved_chunks

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


# --- Function to get answer with RAG ---
# この関数自体はモデル名を直接受け取らず、初期化済みのクライアントを使います
def get_rag_answer(question: str, vector_store: VectorStore, llm_client, model_name: str) -> str:
    """
    Generates an answer using RAG: retrieves context from vector store and queries LLM.

    Args:
        question: The question string.
        vector_store: The VectorStore instance containing indexed documents.
        llm_client: The initialized Ollama client.
        model_name: The name of the LLM model being used (for logging/error messages).

    Returns:
        The generated answer string, or an error message if retrieval or LLM call fails.
    """
    if llm_client is None:
        return "【エラー】LLMクライアントが初期化されていません。"
    if not vector_store or not vector_store.is_indexed:
         return "【エラー】参照資料がインデックス化されていません。"

    try:
        # 1. Retrieve relevant chunks
        retrieved_chunks = vector_store.retrieve(question, k=NUM_RETRIEVED_CHUNKS)

        if not retrieved_chunks:
             return f"【注意】質問 '{question[:50]}...' に関連する情報が参照資料から見つかりませんでした。"

        # 2. Format context for the prompt
        # Add markers around context for clarity to LLM (optional but helps some models)
        context = "---- 参考情報の開始 ----\n" + "\n\n".join(retrieved_chunks) + "\n---- 参考情報の終了 ----"

        # 3. Create RAG-augmented prompt
        augmented_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # 4. Call LLM with augmented prompt (using Ollama chat API)
        response = llm_client.chat(
            model=model_name, # 初期化時のモデル名を使用
            messages=[
                {'role': 'user', 'content': augmented_prompt},
            ],
            # stream=False # デフォルトはFalse
            # オプション例:
            # options={
            #     'temperature': 0.1, # temperatureを調整して回答の多様性を制御（RAGでは低め推奨）
            # }
        )
        answer = response['message']['content']

        print(f"--- 質問 ---")
        print(question)
        print(f"--- RAG回答 ({model_name} / 先頭のみ表示) ---")
        # 回答が長い場合は省略して表示し、改行をスペースに置換
        display_answer = answer.replace('\n', ' ')
        print(display_answer[:150] + "..." if len(display_answer) > 150 else display_answer)
        print("-" * 30)

        return answer

    except Exception as e:
        print(f"エラー: 質問 '{question[:50]}...' に対するRAG回答生成中にエラーが発生しました。")
        print(f"詳細: {e}")
        return f"【エラー】RAG回答生成中にエラーが発生しました: {e}"


# --- Main Execution ---
if __name__ == "__main__":
    # コマンドライン引数の定義とパース
    parser = argparse.ArgumentParser(description='Generate answers using Ollama LLM with RAG.')
    parser.add_argument('--model_name', type=str, required=True, help='Ollama model name to use for RAG generation.')
    args = parser.parse_args()

    llm_model_name = args.model_name # 引数からモデル名を取得


    print(f"Colab + Ollama 環境でRAG回答生成スクリプトを実行中...")
    print(f"カレントディレクトリ: {os.getcwd()}") # Colabでのカレントディレクトリ確認用
    print(f"使用LLMモデル（引数より）: {llm_model_name}")
    print(f"使用Embeddingモデル: {EMBEDDING_MODEL_NAME}")


    # --- 1. LLMクライアントとEmbeddingモデルの初期化 ---
    print("LLMクライアントとEmbeddingモデルを初期化します...")
    # 引数で受け取ったLLMモデル名でクライアントを初期化
    ollama_llm_client = initialize_ollama_client(OLLAMA_HOST, llm_model_name)
    if ollama_llm_client is None:
        print("エラー: LLMクライアントの初期化に失敗したため、処理を中断します。")
        exit()

    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Embeddingモデル '{EMBEDDING_MODEL_NAME}' の初期化に成功しました。")
    except Exception as e:
        print(f"エラー: Embeddingモデル '{EMBEDDING_MODEL_NAME}' の初期化に失敗しました。モデル名の確認、またはインターネット接続を確認してください。")
        print(f"詳細: {e}")
        exit()


    # --- 2. 参照資料の読み込み、分割、インデックス作成 ---
    print(f"\n参照資料を '{REFERENCES_DIR}' から読み込み、インデックスを作成します...")
    documents = load_documents(REFERENCES_DIR)
    if not documents:
        print(f"エラー: '{REFERENCES_DIR}' ディレクトリに読み込める参照資料が見つかりませんでした。パスとファイルを確認してください。")
        exit()

    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
         print("エラー: ドキュメントのチャンク分割に失敗しました。")
         exit()

    vector_store = VectorStore(embedding_model)
    vector_store.add_chunks(chunks)
    if not vector_store.is_indexed:
         print("エラー: ベクトルインデックスの作成に失敗したため、処理を中断します。")
         exit()
    print("参照資料のインデックス作成が完了しました。")


    # --- 3. 質問の読み込み ---
    print(f"\n'{QUESTIONS_FILE}' から質問を読み込んでいます...")
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


    # --- 4. RAGを使用して回答を生成 ---
    print(f"\nRAG ({llm_model_name} + {EMBEDDING_MODEL_NAME}) を使用して回答を生成します...")
    all_answers = {}

    for i, question in enumerate(questions):
        print(f"({i+1}/{len(questions)}) 質問処理中...")
        # モデル名を get_rag_answer 関数に渡す
        answer = get_rag_answer(question, vector_store, ollama_llm_client, llm_model_name)
        all_answers[question] = answer # 質問をキーとして回答を保存

    print(f"\nRAG回答生成が完了しました。")


    # --- 5. 結果の保存 ---
    # 出力ファイル名にモデル名を含める場合はここでもモデル名を使う
    # OUTPUT_FILE_WITH_MODEL = os.path.join(OUTPUT_DIR, f"raw_answers_rag_{llm_model_name.replace(':', '_')}.md")
    print(f"生成されたRAG回答を '{OUTPUT_FILE}' に保存します...")
    try:
        # results ディレクトリが存在しない場合は作成
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# RAGによる回答 (Colab + Ollama - モデル: {llm_model_name})\n\n")
            f.write(f"使用LLM: Ollama ({llm_model_name})\n")
            f.write(f"使用Embeddingモデル: {EMBEDDING_MODEL_NAME}\n")
            f.write(f"RAGパラメータ: Chunk Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}, Retrieved Chunks={NUM_RETRIEVED_CHUNKS}\n\n")
            f.write("このファイルには、RAG（検索拡張生成）を使用して生成された回答が格納されています。\n\n")
            f.write("---\n\n") # セパレーター

            for i, question in enumerate(questions):
                f.write(f"## 質問 {i+1}\n\n")
                f.write(f"**質問:** {question}\n\n")
                f.write(f"**RAG回答:**\n{all_answers.get(question, '【エラー】回答が取得できませんでした')}\n\n")
                f.write("---\n\n") # セパレーター

        print(f"RAG回答が正常に保存されました: {OUTPUT_FILE}")

    except Exception as e:
        print(f"エラー: 回答の保存中にエラーが発生しました: {e}")