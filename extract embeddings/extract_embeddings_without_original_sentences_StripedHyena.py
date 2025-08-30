import os
import time
import traceback
import numpy as np
import torch
from scipy.io import savemat
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

STIMULI_FILE = "stimuli_180concepts.txt"
OUTPUT_DIR   = "embeddings_output"
MODEL_NAME   = "stripedhyena-7b"
MODEL_ID     = "togethercomputer/StripedHyena-Hessian-7B"

def load_stimuli(path: str):
    with open(path, "r", encoding="utf-8") as f:
        words = [ln.strip() for ln in f if ln.strip()]
    if not words:
        raise RuntimeError(f"{path} is empty or has no valid lines.")
    return words

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

    # Turn off potentially problematic flags if present
    for k in [
        "use_flash_attn", "use_flash_attention", "use_flash_attn_2",
        "use_mem_eff_attn", "use_memory_efficient_attention",
        "use_flash_msnorm", "use_flash_rmsnorm", "use_flash_ms_norm",
        "fused_mlp", "fused_rmsnorm", "fused_msnorm"
    ]:
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, False)
            except Exception:
                pass

    if hasattr(cfg, "attn_implementation"):
        try:
            cfg.attn_implementation = "eager"
        except Exception:
            pass
    return cfg

@torch.inference_mode()
def get_embedding(model, tokenizer, text, device=None):
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False
    ).to(device)

    out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hs = getattr(out, "hidden_states", None)
    if isinstance(hs, (tuple, list)) and len(hs) > 0:
        last = hs[-1]                    # (B, T, D)
        vec = last[:, -1, :].squeeze(0)  # (D,)
        return vec.detach().float().cpu().numpy()
    final_hidden = {}
    backbone = getattr(model, "backbone", None)
    norm_mod = getattr(backbone, "norm", None) if backbone is not None else None
    if norm_mod is None:
        raise RuntimeError("No backbone.norm found; SH hook cannot be used.")

    def _hook(_m, _inp, out):
        final_hidden["x"] = out  # (B, T, D)

    h = norm_mod.register_forward_hook(_hook)
    try:
        _ = model(**inputs, output_hidden_states=False, return_dict=True, use_cache=False)
    finally:
        h.remove()

    if "x" not in final_hidden:
        raise RuntimeError("Hook didn't capture final hidden states.")

    last = final_hidden["x"]              # (B, T, D)
    vec  = last[:, -1, :].squeeze(0)      # (D,)
    return vec.detach().float().cpu().numpy()

def text_embedding(model, tokenizer, text, device=None):
    return get_embedding(model, tokenizer, text, device=device)

def main():
    t0 = time.time()
    print(f"Model: {MODEL_NAME}  ({MODEL_ID})")
    device, dtype = pick_device_dtype()
    print(f"Device: {device} | dtype: {dtype}")

    words = load_stimuli(STIMULI_FILE)
    print(f"Loaded {len(words)} words (e.g., {words[:5]})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"embeddings_{MODEL_NAME}.mat")
    if os.path.exists(out_path):
        print(f"Note: will overwrite {out_path}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg = sanitize_config(cfg)

    # model
    print("Loading model (try device_map='auto')...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"device_map='auto' failed, fallback to single device: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=dtype
        )
        model.to(device)
    model.eval()

    hidden_dim = None
    for key in ("hidden_size", "d_model", "model_dim", "dim"):
        if hasattr(model.config, key):
            try:
                v = int(getattr(model.config, key))
                if v > 0:
                    hidden_dim = v
                    break
            except Exception:
                pass
    if hidden_dim is None:
        test_vec = get_embedding(model, tokenizer, "hello", device=next(model.parameters()).device)
        hidden_dim = int(test_vec.shape[-1])
    print(f"Embedding dim: {hidden_dim}")

    # loop
    N = len(words)
    embeddings = np.zeros((N, hidden_dim), dtype=np.float32)
    fail_cnt = 0
    dev = next(model.parameters()).device

    for i, w in enumerate(words, 1):
        if i % 30 == 0:
            print(f"{i}/{N} - {w}")
        try:
            emb = get_embedding(model, tokenizer, w, device=dev)
            if emb.shape[-1] != hidden_dim:  # ultra-rare safety
                if emb.shape[-1] > hidden_dim:
                    emb = emb[:hidden_dim]
                else:
                    tmp = np.zeros((hidden_dim,), dtype=np.float32)
                    tmp[:emb.shape[-1]] = emb
                    emb = tmp
            embeddings[i-1] = emb.astype(np.float32, copy=False)
        except Exception as ex:
            fail_cnt += 1
            print(f"fail {i}/{N} '{w}': {ex}")

    savemat(out_path, {"embeddings": embeddings, "words": words})
    dt = time.time() - t0
    print("-" * 50)
    print(f"Saved: {out_path} | items={N}, fails={fail_cnt}, time={dt:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("uncaught error:", e)
        traceback.print_exc()
