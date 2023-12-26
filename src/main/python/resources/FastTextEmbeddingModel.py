from pathlib import Path


def get_fast_text_embedding_model_path():
    return str(Path(__file__).parent) + '/cc.en.50.bin'
