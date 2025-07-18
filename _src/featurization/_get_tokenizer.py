from . import __dict__ as tokenizers

def get_tokenizer(name):
    return tokenizers[name]
