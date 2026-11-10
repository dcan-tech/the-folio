
"""TF-18: Run a local language model and record generation metrics."""

import json
import resource
import sys
import time
from pathlib import Path

MODEL_DIR = Path("./model")
PROMPT = "Write a four-line poem about an old computer."
MAX_NEW_TOKENS = 80
RESULTS_PATH = Path("results/tf18_qwen_baseline.json")


def build_messages(prompt):
    """Format a prompt as a user chat message."""
    return [{"role": "user", "content": prompt}]


def tokens_per_second(token_count, elapsed_seconds):
    """Calculate the number of generated tokens per second."""
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return token_count / elapsed_seconds


def peak_memory_mib():
    """Return peak process memory in MiB (Linux reports ru_maxrss in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def save_result(path, record):
    """Save a JSON record, creating the directory if necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def generate(model_dir, prompt, max_new_tokens):
    """Generate text using a locally stored model.

    Return the generated text, token count, and elapsed time.
    """
    # Keep model dependencies separate from helper functions used in CI.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
    )
    model.eval()

    inputs = tokenizer.apply_chat_template(
        build_messages(prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    prompt_length = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    elapsed = time.perf_counter() - start

    new_tokens = output[0, prompt_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, len(new_tokens), elapsed


def main():
    if not MODEL_DIR.exists():
        sys.exit(f"Model folder not found: {MODEL_DIR}")

    print("Loading Qwen...")
    poem, count, elapsed = generate(MODEL_DIR, PROMPT, MAX_NEW_TOKENS)
    speed = tokens_per_second(count, elapsed)
    memory = peak_memory_mib()

    print(poem)
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Speed: {speed:.2f} tokens/second")
    print(f"Peak memory: {memory:.0f} MiB")

    save_result(
        RESULTS_PATH,
        {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "prompt": PROMPT,
            "settings": {
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "dtype": "float32",
                "prompt_format": "default chat template",
            },
            "output": poem,
            "metrics": {
                "generation_seconds": round(elapsed, 2),
                "new_tokens": count,
                "tokens_per_second": round(speed, 2),
                "peak_memory_mib": round(memory),
            },
        },
    )
    print(f"Saved result to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
