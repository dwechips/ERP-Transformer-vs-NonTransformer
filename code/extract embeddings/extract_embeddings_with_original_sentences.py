import os
import time
import torch
import numpy as np
import pandas as pd
from scipy.io import savemat
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

CONCEPTS_CSV = 'all_concepts_sentences.csv'
OUTPUT_DIR = 'embeddings_original_sentences'

MODEL_LIST = [
    {'name': 'llama2-7b', 'model_id': 'meta-llama/Llama-2-7b-hf'},
    {'name': 'mistral-7b', 'model_id': 'mistralai/Mistral-7B-v0.1'},
    {'name': 'rwkv-7b', 'model_id': 'RWKV/rwkv-6-world-7b'},
    {'name': 'stripedhyena-7b', 'model_id': 'togethercomputer/StripedHyena-Hessian-7B'}
]

def load_concepts_sentences(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} concepts")
    return df

def parse_sentences(s, sep='|'):
    if pd.isna(s):
        return []
    return [x.strip() for x in s.split(sep) if x.strip()]

def pick_device_dtype():
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dev = torch.device("cpu")
        dt = torch.float32
    return dev, dt

def sanitize_config(cfg):
    try:
        cfg.output_hidden_states = True
    except Exception:
        pass
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    return cfg

def _get_last_hidden_with_hook(model, inputs):
    final_hidden = {}
    norm_mod = getattr(model, "final_norm", None)
    if norm_mod is None:
        raise RuntimeError("No final norm found")
    def _hook(_m, _inp, out):
        final_hidden["x"] = out
    h = norm_mod.register_forward_hook(_hook)
    _ = model(**inputs, output_hidden_states=False, return_dict=True, use_cache=False)
    h.remove()
    return final_hidden["x"]

@torch.no_grad()
def get_single_embedding(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hs = getattr(outputs, "hidden_states", None)
    if hs is not None and len(hs) > 0:
        last = hs[-1]
    else:
        last = _get_last_hidden_with_hook(model, inputs)

    attn = inputs.get("attention_mask", None)
    token_ids = inputs["input_ids"][0]
    mask = tokenizer.get_special_tokens_mask(token_ids.tolist(), already_has_special_tokens=True)
    mask = torch.tensor([0 if m else 1 for m in mask], device=last.device)
    keep = (attn[0] > 0).to(last.dtype) * mask.to(last.dtype) if attn is not None else mask.to(last.dtype)
    keep = keep.unsqueeze(-1)

    pooled = (last[0] * keep).sum(dim=0) / keep.sum().clamp_min(1.0)
    emb = pooled.to(torch.float32).cpu().numpy()
    nrm = np.linalg.norm(emb)
    if nrm > 1e-12:
        emb = emb / nrm
    return emb

@torch.no_grad()
def extract_embedding(model, tokenizer, word, sentences, device):
    if not sentences:
        sentences = [f"The concept {word} means"]
    vecs = [torch.from_numpy(get_single_embedding(model, tokenizer, s, device)) for s in sentences]
    arr = torch.stack(vecs, dim=0).mean(dim=0).numpy()
    nrm = np.linalg.norm(arr)
    return arr / nrm if nrm > 1e-12 else arr

def main():
    if not os.path.exists(CONCEPTS_CSV):
        raise FileNotFoundError(f"Missing {CONCEPTS_CSV}")

    df = load_concepts_sentences(CONCEPTS_CSV)
    device, dtype = pick_device_dtype()
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_info in MODEL_LIST:
        model_name, model_id = model_info['name'], model_info['model_id']
        out_file = os.path.join(OUTPUT_DIR, f"embeddings_{model_name}.mat")
        if os.path.exists(out_file):
            print(f"Skip {out_file}")
            continue

        print(f"\n=== {model_name} ===")
        start = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if 'stripedhyena' in model_name.lower():
                cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                cfg = sanitize_config(cfg)
                model = AutoModelForCausalLM.from_pretrained(model_id, config=cfg, trust_remote_code=True,
                                                             torch_dtype=dtype, device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                             device_map="auto", trust_remote_code=True)
            model.eval()

            embeddings, words = [], []
            for idx, row in df.iterrows():
                word, s = row['word'], row['sentences']
                sentences = parse_sentences(s)
                if idx % 30 == 0:
                    print(f"{idx}/{len(df)} - {word}")
                try:
                    emb = extract_embedding(model, tokenizer, word, sentences, device)
                except Exception as e:
                    print(f"fail {word}: {e}")
                    emb = np.full((4096,), np.nan, dtype=np.float32)
                embeddings.append(emb)
                words.append(word)

            arr = np.vstack(embeddings).astype(np.float32)
            savemat(out_file, {'embeddings': arr, 'words': np.array(words, dtype=object)})
            print(f"Saved {out_file}, shape={arr.shape}, time={time.time()-start:.1f}s")

        except Exception as e:
            print(f"Model {model_name} failed: {e}")

        finally:
            del model, tokenizer
            torch.cuda.empty_cache()

    print("All done!")

if __name__ == "__main__":
    main()
