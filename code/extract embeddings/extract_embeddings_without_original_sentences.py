import os
import time
import torch
import numpy as np
from scipy.io import savemat
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Config ---
STIMULI_FILE = 'stimuli_180concepts.txt'
OUTPUT_DIR = 'embeddings_output'

MODEL_LIST = [
    {'name': 'llama2-7b', 'model_id': 'meta-llama/Llama-2-7b-hf'},
    {'name': 'mistral-7b', 'model_id': 'mistralai/Mistral-7B-v0.1'},
    {'name': 'rwkv-7b', 'model_id': 'RWKV/rwkv-6-world-7b'},
    #{'name': 'stripedhyena-7b', 'model_id': 'togethercomputer/StripedHyena-Hessian-7B'}
]

def load_stimuli(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stimuli = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(stimuli)} stimuli")
    return stimuli

def get_embedding(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    hs = getattr(outputs, 'hidden_states', None)
    last_hidden = hs[-1] if hs else outputs.last_hidden_state
    emb = last_hidden[:, -1, :].squeeze().detach().cpu().numpy()
    return emb

def main():
    if not torch.cuda.is_available():
        print("No GPU found, running on CPU (will be slow).")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = torch.float16
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    stimuli = load_stimuli(STIMULI_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_info in MODEL_LIST:
        model_name, model_id = model_info['name'], model_info['model_id']
        out_file = os.path.join(OUTPUT_DIR, f"embeddings_{model_name}.mat")

        if os.path.exists(out_file):
            print(f"Skip {out_file}, already exists")
            continue

        print(f"\n=== {model_name} ===")
        start = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )

            embeddings = []
            for i, word in enumerate(stimuli):
                if i % 30 == 0:
                    print(f"{i}/{len(stimuli)} - {word}")
                try:
                    emb = get_embedding(model, tokenizer, word, device)
                except Exception as e:
                    print(f"fail {word}: {e}")
                    emb = np.full((4096,), np.nan, dtype=np.float32)
                embeddings.append(emb)

            arr = np.vstack(embeddings).astype(np.float32)
            savemat(out_file, {'embeddings': arr})
            print(f"Saved {out_file}, shape={arr.shape}, time={time.time()-start:.1f}s")

        except Exception as e:
            print(f"Model {model_name} failed: {e}")

        finally:
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            torch.cuda.empty_cache()

    print("\nAll models done.")

if __name__ == "__main__":
    main()
