import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents

def get_custom_tokenizer(text_data, vocab_size=4096):
    """
    Highly efficient BPE tokenizer.
    Vocab 4096 is the 'Goldilocks' zone for 20% Shakespeare.
    """
    # 1. Initialize BPE
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # 2. Normalization: Essential for Path B to prevent 'King' and 'king' 
    # from being treated as different concepts.
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    
    # 3. Pre-tokenization: Splits by whitespace so BPE doesn't merge across words.
    tokenizer.pre_tokenizer = Whitespace()
    
    # 4. Trainer settings
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        initial_alphabet=[], # Start from scratch
        show_progress=True
    )
    
    # 5. Train (Takes ~2-5 seconds on Shakespeare)
    tokenizer.train_from_iterator([text_data], trainer=trainer)
    
    # Wrapper to match your script's API
    class BDHTokenizer:
        def __init__(self, t):
            self.t = t
            self.n_vocab = t.get_vocab_size()
            
        def encode(self, text):
            # Returns a list of integers
            return self.t.encode(text).ids
            
        def decode(self, tokens):
            # Returns a clean string
            return self.t.decode(tokens)

    return BDHTokenizer(tokenizer)