# base_llm_answer_generator.py (Adapted for Google Colab Environment using Ollama)

import os
import json
# --- �g�p����LLM���C�u�������C���|�[�g ---
# Ollama ���g�p����ꍇ
import ollama

# --- �ݒ� ---
# Ollama�͒ʏ�A���[�J�����s�ł�API�L�[�͕s�v�ł��B
# Colab����Ollama�T�[�o�[���N�����A���f�����v������K�v������܂��i���O�����X�e�b�v�Q�Ɓj�B

# �g�p����Ollama���f������I��
# Colab��� Ollama pull �ς� �̃��f�����w�肵�Ă������� (��: 'llama3', 'mistral', 'gemma', 'codeqwen' �Ȃ�)
LLM_MODEL_NAME = "deepseek-r1:1.5b" # <-- �����Ɏg�p����Ollama���f������ݒ�

# Ollama�T�[�o�[�̃A�h���X�iColab��Ńf�t�H���g��localhost�j
# �قȂ�A�h���X��Ollama�����s���Ă���ꍇ�͂�����ύX
OLLAMA_HOST = "http://localhost:11434"

# ����/�o�̓t�@�C���̃p�X (assuming script is run from the 'code' directory after cloning)
# ���|�W�g�����N���[����Acode�f�B���N�g���Ɉړ����Ď��s���邱�Ƃ�z�肵�Ă��܂��B
QUESTIONS_FILE = os.path.join("..", "questions", "questions.txt")
OUTPUT_DIR = os.path.join("..", "results") # Directory to save results
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_answers_base_llm.md") # Base LLM results file


# --- LLM�N���C�A���g�̏����� ---
# Ollama �N���C�A���g�̏�����
try:
    # Colab��� Ollama serve ���N�����Ă���K�v������܂�
    llm_client = ollama.Client(host=OLLAMA_HOST)
    # ���f�������݂��邩�ȒP�Ȋm�F (ping�̑���A���݂��Ȃ��Ƃ����ŃG���[�ɂȂ�)
    llm_client.show(LLM_MODEL_NAME) # ���f�����\���͕s�v�����A���݊m�F�ɂȂ�
    print(f"Colab + Ollama ���p LLM�N���C�A���g�̏������ɐ������܂����B�g�p���f��: {LLM_MODEL_NAME}")
except ollama.ResponseError as e:
     print(f"�G���[: Ollama�N���C�A���g�܂��̓��f�� ({LLM_MODEL_NAME}) �̏������Ɏ��s���܂����B")
     print(f"Ollama�T�[�o�[�� '{OLLAMA_HOST}' �ŋN�����Ă��邩�A�w�胂�f�����C���X�g�[���ς݂��m�F���Ă��������B")
     print(f"�ڍ�: {e}")
     exit()
except Exception as e:
    print(f"�G���[: Ollama�N���C�A���g�̏��������ɗ\�����ʃG���[���������܂����B")
    print(f"�ڍ�: {e}")
    exit()


# --- �x�[�XLLM����񓚂��擾����֐� ---
def get_base_llm_answer(question: str) -> str:
    """
    �ݒ肳�ꂽOllama���f�����g�p���āA�w�肳�ꂽ����ɑ΂���񓚂𐶐����܂��B
    RAG�͎g�p���܂���B

    Args:
        question: ���╶����B

    Returns:
        �������ꂽ�񓚕�����A�܂��͌Ăяo���Ɏ��s�����ꍇ�̓G���[���b�Z�[�W�B
    """
    try:
        # Ollama chat completions API ���g�p
        response = llm_client.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {'role': 'user', 'content': question},
            ],
            # stream=False # �f�t�H���g��False�Ȃ̂ŏȗ���
            # �I�v�V������:
            # options={
            #     'temperature': 0.8,
            #     'num_ctx': 4096,
            # }
        )
        answer = response['message']['content']

        print(f"--- ���� ---")
        print(question)
        print(f"--- �x�[�XLLM�� (�擪�̂ݕ\��) ---")
        # �񓚂������ꍇ�͏ȗ����ĕ\�����A���s���X�y�[�X�ɒu��
        display_answer = answer.replace('\n', ' ')
        print(display_answer[:150] + "..." if len(display_answer) > 150 else display_answer)
        print("-" * 30)

        return answer

    except ollama.ResponseError as e:
        print(f"�G���[: Ollama���f�� '{LLM_MODEL_NAME}' �Ăяo�����ɃG���[���������܂����B")
        print(f"���f���������������AOllama�ɃC���X�g�[������Ă��邩�m�F���Ă��������B")
        print(f"�ڍ�: {e}")
        return f"�y�G���[�z���f���Ăяo���G���[: {e}"
    except Exception as e:
        print(f"�G���[: ���� '{question[:50]}...' �ɑ΂���񓚐������ɗ\�����ʃG���[���������܂����B")
        print(f"�ڍ�: {e}")
        return f"�y�G���[�z�񓚐������ɃG���[���������܂���: {e}"


# --- ���C������ ---
if __name__ == "__main__":
    print(f"Colab + Ollama ���Ŏ��s��...")
    print(f"�J�����g�f�B���N�g��: {os.getcwd()}") # Colab�ł̃J�����g�f�B���N�g���m�F�p

    print(f"'{QUESTIONS_FILE}' ���玿���ǂݍ���ł��܂�...")
    questions = []
    try:
        # utf-8 �G���R�[�f�B���O�Ńt�@�C�����J��
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            # ��s�����O���Ċe�s������Ƃ��ēǂݍ���
            questions = [line.strip() for line in f if line.strip()]
        if not questions:
            print("�G���[: ����t�@�C���ɗL���Ȏ��₪������܂���ł����B")
            print(f"�p�X���m�F���Ă�������: {QUESTIONS_FILE}")
            # Colab�̏ꍇ�Als ../questions/ �� !cat ../questions/questions.txt �ő��݊m�F�\
            exit()
        print(f"{len(questions)} ���̎����ǂݍ��݂܂����B")

    except FileNotFoundError:
        print(f"�G���[: ����t�@�C����������܂���B�p�X���m�F���Ă�������: {QUESTIONS_FILE}")
        print(f"Colab�̏ꍇ�A�J�����g�f�B���N�g�����m�F���A%cd �R�}���h�ňړ����Ă��������B")
        exit()
    except Exception as e:
        print(f"�G���[: ����t�@�C���̓ǂݍ��ݒ��ɃG���[���������܂���: {e}")
        exit()


    print(f"Ollama ({LLM_MODEL_NAME}) ���g�p���ĉ񓚂𐶐����܂�...")
    all_answers = {}

    # �e����ɑ΂��ĉ񓚂𐶐�
    for i, question in enumerate(questions):
        print(f"({i+1}/{len(questions)}) ���⏈����...")
        answer = get_base_llm_answer(question)
        all_answers[question] = answer # ������L�[�Ƃ��ĉ񓚂�ۑ�

    print(f"\n�񓚐������������܂����B")

    # --- ���ʂ̕ۑ� ---
    print(f"�������ꂽ�񓚂� '{OUTPUT_FILE}' �ɕۑ����܂�...")
    try:
        # results �f�B���N�g�������݂��Ȃ��ꍇ�͍쐬
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# �x�[�XLLM (Colab + Ollama) �ɂ��� (RAG�Ȃ�)\n\n")
            f.write(f"�g�pOllama���f��: {LLM_MODEL_NAME}\n")
            f.write(f"Ollama Host: {OLLAMA_HOST}\n\n")
            f.write("���̃t�@�C���ɂ́ARAG�i�����g�������j���g�p�����A�w�肳�ꂽOllama���f���̒m���݂̂Ɋ�Â��Đ������ꂽ�񓚂��i�[����Ă��܂��B\n\n")
            f.write("---\n\n") # �Z�p���[�^�[

            for i, question in enumerate(questions):
                f.write(f"## ���� {i+1}\n\n")
                f.write(f"**����:** {question}\n\n")
                f.write(f"**�x�[�XLLM��:**\n{all_answers.get(question, '�y�G���[�z�񓚂��擾�ł��܂���ł���')}\n\n")
                f.write("---\n\n") # �Z�p���[�^�[

        print(f"�񓚂�����ɕۑ�����܂���: {OUTPUT_FILE}")

    except Exception as e:
        print(f"�G���[: �񓚂̕ۑ����ɃG���[���������܂���: {e}")