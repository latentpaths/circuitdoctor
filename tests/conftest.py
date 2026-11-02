import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from circuitdoctor.runtime import Runtime
from circuitdoctor.schema import RunConfig, Sampling


@pytest.fixture
def runtime():
    torch.manual_seed(0)
    vocab = {
        "[PAD]": 0,
        "[EOS]": 1,
        "[UNK]": 2,
        "What": 3,
        "is": 4,
        "2": 5,
        "+": 6,
        "3": 7,
        "?": 8,
        "5": 9,
        "The": 10,
        "answer": 11,
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    cfg = Qwen3Config(
        vocab_size=12,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = Qwen3ForCausalLM(cfg)
    config = RunConfig(
        model="tiny-fixture",
        revision="fixture",
        dtype="float32",
        device="cpu",
        batch_size=2,
        capture_token_limit=64,
        validation_max_tokens=16,
        sampling=Sampling(max_new_tokens=16),
    )
    return Runtime(config, model=model, tokenizer=tokenizer)
