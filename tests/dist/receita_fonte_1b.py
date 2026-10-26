"""Monta a pasta de origem que o convert_hf_to_gguf.py do llama.cpp b3821 transforma no
Llama-3.2-1B-Instruct-f16.gguf do bartowski com o mesmo SHA-256 (1f33ad43…).

O repositório da Meta é fechado; esta pasta parte da cópia aberta do unsloth (mesmo
model.safetensors) e desfaz o que o unsloth mudou: config da Meta, sem pad_token, merges no
formato antigo de texto, e o cabeçalho do README da Meta (o conversor tira dali general.license,
tags e languages). O nome da pasta vira general.name/basename/size_label.

    python3 tests/dist/receita_fonte_1b.py --saida fonte [--safetensors model.safetensors]
    python3 llama.cpp-b3821/convert_hf_to_gguf.py fonte/Llama-3.2-1B-Instruct --outtype f16 \\
        --outfile Llama-3.2-1B-Instruct-f16.gguf
"""
import argparse, json, os, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--saida", required=True)
ap.add_argument("--safetensors", help="model.safetensors já baixado (senão baixa do unsloth)")
a = ap.parse_args()
D = os.path.join(a.saida, "Llama-3.2-1B-Instruct")
os.makedirs(D, exist_ok=True)
U = "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/"


def baixa(nome):
    return urllib.request.urlopen(U + nome).read()


# config.json, generation_config.json e special_tokens_map.json como estão na Meta
eos = [128001, 128008, 128009]
config = {
    "architectures": ["LlamaForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": 128000, "eos_token_id": eos, "head_dim": 64, "hidden_act": "silu",
    "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 8192,
    "max_position_embeddings": 131072, "mlp_bias": False, "model_type": "llama",
    "num_attention_heads": 32, "num_hidden_layers": 16, "num_key_value_heads": 8,
    "pretraining_tp": 1, "rms_norm_eps": 1e-05,
    "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
    "rope_theta": 500000.0, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0.dev0", "use_cache": True, "vocab_size": 128256,
}
geracao = {"bos_token_id": 128000, "do_sample": True, "eos_token_id": eos, "temperature": 0.6,
           "top_p": 0.9, "transformers_version": "4.45.0.dev0"}
tok = {"content": "", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
especiais = {"bos_token": {**tok, "content": "<|begin_of_text|>"}, "eos_token": {**tok, "content": "<|eot_id|>"}}
for nome, v in (("config.json", config), ("generation_config.json", geracao), ("special_tokens_map.json", especiais)):
    json.dump(v, open(os.path.join(D, nome), "w"), indent=2)

# tokenizer_config.json: o do unsloth sem o que ele acrescentou
t = json.loads(baixa("tokenizer_config.json"))
for k in ("pad_token", "padding_side", "extra_special_tokens", "add_bos_token"):
    t.pop(k, None)
t["tokenizer_class"] = "PreTrainedTokenizerFast"
json.dump(t, open(os.path.join(D, "tokenizer_config.json"), "w"), indent=2, ensure_ascii=False)

# tokenizer.json: merges em pares (formato novo) voltam a "a b"; o gguf-py da b3821 só lê texto
k = json.loads(baixa("tokenizer.json"))
m = k["model"]["merges"]
if m and isinstance(m[0], list):
    k["model"]["merges"] = [" ".join(p) for p in m]
json.dump(k, open(os.path.join(D, "tokenizer.json"), "w"), ensure_ascii=False)

open(os.path.join(D, "README.md"), "w").write(
    "---\nlanguage:\n- en\n- de\n- fr\n- it\n- pt\n- hi\n- es\n- th\nlibrary_name: transformers\n"
    "pipeline_tag: text-generation\ntags:\n- facebook\n- meta\n- pytorch\n- llama\n- llama-3\n"
    "license: llama3.2\n---\n")

st = os.path.join(D, "model.safetensors")
if not os.path.exists(st):
    if a.safetensors:
        os.symlink(os.path.abspath(a.safetensors), st)
    else:
        urllib.request.urlretrieve(U + "model.safetensors", st)
print(D)
