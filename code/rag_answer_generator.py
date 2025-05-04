# rag_answer_generator.py (Adapted for Colab Environment using Ollama and sentence-transformers)

import os
import json
import re
import numpy as np
# --- Import necessary libraries ---
# Ollama for LLM
import ollama
# Sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity # For similarity calculation

# --- Configuration ---
# Ollama Host (assuming it's running in the Colab VM)
OLLAMA_HOST = "http://localhost:11434"

# LLM Model Name (must be pulled in Colab Ollama)
# TODO: Base LLMと同じモデル名を設定
LLM_MODEL_NAME = "deepseek-r1:1.5b"

# Embedding Model Name (from sentence-transformers library)
# Lightweight models are good for Colab
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Or similar lightweight model

# Paths (assuming script is run from the 'code' directory)
REPO_ROOT = os.path.join("..") # Go up from 'code' to repository root
QUESTIONS_FILE = os.path.join(REPO_ROOT, "questions", "questions.txt")
REFERENCES_DIR = os.path.join(REPO_ROOT, "references")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_answers_rag.md") # RAG answers file

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
        if os.path.isfile(filepath) and filename.endswith(('.txt', '.md')):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                print(f"Loaded document: {filename}")
            except Exception as e:
                print(f"Error loading document {filename}: {e}")
    return documents

def split_documents(documents: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits documents into overlapping chunks."""
    chunks = []
    # Simple split logic (more advanced might use NLTK, spaCy, recursive splitters)
    for doc in documents:
        # Remove excessive whitespace/newlines for cleaner chunks
        cleaned_doc = re.sub(r'\s+', ' ', doc).strip()
        # Split by a large enough chunk size, then handle overlap
        words = cleaned_doc.split()
        if not words: continue

        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunks.append(' '.join(chunk_words))
            # Move index for next chunk
            i += chunk_size - chunk_overlap
            if i < 0: i = 0 # Ensure index doesn't go below zero for overlap

        # Ensure the last part is included even if smaller than chunk_size
        if (i - (chunk_size - chunk_overlap)) < len(words) and len(chunks) > 0:
             if ' '.join(words[i:]) != chunks[-1]: # Avoid adding duplicate if overlap makes it identical
                 chunks.append(' '.join(words[i:]))
             # Basic fallback for very short docs not chunked correctly
             if not chunks and words:
                 chunks.append(' '.join(words))


    # Refined simple splitter (split by paragraphs/sentences then combine)
    # This is often better than just splitting words/chars
    chunks = []
    for doc in documents:
        # Split by lines or simple patterns
        paragraphs = doc.split('\n\n') # Split by double newline
        current_chunk = ""
        for para in paragraphs:
            # Clean up extra whitespace
            cleaned_para = para.strip()
            if not cleaned_para: continue

            # If adding paragraph exceeds chunk size, start new chunk
            # Add some buffer like +20 for safety or transition words
            if len(current_chunk) + len(cleaned_para) > chunk_size + 20 and current_chunk:
                 chunks.append(current_chunk)
                 # Add overlap - take last part of previous chunk
                 overlap_text = current_chunk[-chunk_overlap:].strip()
                 current_chunk = overlap_text + " " + cleaned_para if overlap_text else cleaned_para
            else:
                 current_chunk += (" " if current_chunk else "") + cleaned_para

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)

    print(f"Split documents into {len(chunks)} chunks.")
    return chunks


class VectorStore:
    """Simple in-memory vector store."""
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None # Will store embeddings as a numpy array

    def add_chunks(self, chunks: list[str]):
        """Adds chunks and computes their embeddings."""
        print(f"Creating embeddings for {len(chunks)} chunks...")
        self.chunks = chunks
        # Compute embeddings
        # The encode method automatically handles batching
        self.embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        print("Embedding creation complete.")


    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieves top-k most similar chunks to the query."""
        if not self.chunks or self.embeddings is None:
            print("Vector store is empty. Add chunks first.")
            return []

        # Get query embedding
        query_embedding = self.embedding_model.encode(query)

        # Calculate cosine similarity between query embedding and all chunk embeddings
        # query_embedding needs to be 2D for cosine_similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get the indices of the top-k most similar chunks
        # argsort sorts in ascending order, so use [::-1] to get descending, then take top k
        top_k_indices = similarities.argsort()[::-1][:k]

        # Return the corresponding chunks
        retrieved_chunks = [self.chunks[i] for i in top_k_indices]

        # Optional: Print retrieved chunks for debugging
        # print("\n--- Retrieved Chunks ---")
        # for i, chunk in enumerate(retrieved_chunks):
        #     print(f"Chunk {i+1}: {chunk[:100]}...") # Print snippet
        # print("------------------------")

        return retrieved_chunks

# --- LLM Client Initialization ---
def initialize_ollama_client(host: str, model_name: str):
    """Initializes and tests the Ollama client."""
    try:
        client = ollama.Client(host=host)
        # Test connection and model existence
        client.show(model_name) # This checks if the model exists
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
def get_rag_answer(question: str, vector_store: VectorStore, llm_client) -> str:
    """
    Generates an answer using RAG: retrieves context from vector store and queries LLM.

    Args:
        question: The question string.
        vector_store: The VectorStore instance containing indexed documents.
        llm_client: The initialized Ollama client.

    Returns:
        The generated answer string, or an error message if retrieval or LLM call fails.
    """
    if llm_client is None:
        return "【エラー】LLMクライアントが初期化されていません。"
    if not vector_store or not vector_store.chunks:
         return "【エラー】参照資料が読み込まれていません。"

    try:
        # 1. Retrieve relevant chunks
        retrieved_chunks = vector_store.retrieve(question, k=NUM_RETRIEVED_CHUNKS)

        if not retrieved_chunks:
             return f"【注意】質問 '{question[:50]}...' に関連する情報が見つかりませんでした。"

        # 2. Format context for the prompt
        context = "\n\n".join(retrieved_chunks)

        # 3. Create RAG-augmented prompt
        augmented_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # 4. Call LLM with augmented prompt (using Ollama chat API)
        response = llm_client.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {'role': 'user', 'content': augmented_prompt},
            ],
            # stream=False # デフォルトはFalse
            # オプション例:
            # options={
            #     'temperature': 0.8, # temperatureを調整して回答の多様性を制御
            #     'num_ctx': 4096, # context window size
            # }
        )
        answer = response['message']['content']

        print(f"--- 質問 ---")
        print(question)
        print(f"--- RAG回答 (先頭のみ表示) ---")
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
    print(f"Colab + Ollama 環境でRAG回答生成スクリプトを実行中...")
    print(f"カレントディレクトリ: {os.getcwd()}") # Colabでのカレントディレクトリ確認用

    # --- 1. LLMクライアントとEmbeddingモデルの初期化 ---
    print("LLMクライアントとEmbeddingモデルを初期化します...")
    ollama_llm_client = initialize_ollama_client(OLLAMA_HOST, LLM_MODEL_NAME)
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
    print(f"\nRAG ({LLM_MODEL_NAME} + {EMBEDDING_MODEL_NAME}) を使用して回答を生成します...")
    all_answers = {}

    for i, question in enumerate(questions):
        print(f"({i+1}/{len(questions)}) 質問処理中...")
        answer = get_rag_answer(question, vector_store, ollama_llm_client)
        all_answers[question] = answer # 質問をキーとして回答を保存

    print(f"\nRAG回答生成が完了しました。")


    # --- 5. 結果の保存 ---
    print(f"生成されたRAG回答を '{OUTPUT_FILE}' に保存します...")
    try:
        # results ディレクトリが存在しない場合は作成
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# RAGによる回答\n\n")
            f.write(f"使用LLM: Ollama ({LLM_MODEL_NAME})\n")
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