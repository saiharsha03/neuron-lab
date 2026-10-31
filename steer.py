"""Push one MLP neuron and ask the model who it is (neuronpersonas.com method).

h_L <- h_L + c * W_down[:, n],  c = +/- alpha * r_L / ||W_down[:, n]||
r_L = median residual-stream norm after layer L over the prompt tokens (first token excluded).

Usage:
  python steer.py --layer 18 --neuron 4021
  python steer.py --model Qwen/Qwen3.5-4B --layer 10 --neuron 77 --sign up
"""

import argparse
import json
from pathlib import Path
from threading import Thread

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TextIteratorStreamer

ALPHAS = [1.25, 1.0, 0.75, 0.5]


def looks_incoherent(text: str) -> bool:
    # TODO: decide when a steered reply counts as broken, so we retry at a lower alpha.
    # The site used an LLM judge; a cheap local heuristic is fine here.
    return False


def find_layers(model):
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) and hasattr(getattr(module[0], "mlp", None), "down_proj"):
            return module
    raise RuntimeError("no decoder layers with mlp.down_proj found")


def load(model_id, gpu_mem="6.5GiB"):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: gpu_mem, "cpu": "48GiB"},
    )
    return tok, model, find_layers(model)


def chat_inputs(tok, model, prompt):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tok(text, return_tensors="pt").to(model.device)


def residual_norm(model, inputs, layer_idx):
    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True).hidden_states
    return hs[layer_idx + 1][0, 1:].float().norm(dim=-1).median().item()


def down_proj_column(model_id, layers, layer_idx, neuron, model):
    # Offloaded layers are empty (meta) outside accelerate's forward, so read the column from the checkpoint.
    prefix = next(name for name, m in model.named_modules() if m is layers)
    key = f"{prefix}.{layer_idx}.mlp.down_proj.weight"
    folder = Path(snapshot_download(model_id, local_files_only=True))
    index = folder / "model.safetensors.index.json"
    shard = json.loads(index.read_text())["weight_map"][key] if index.exists() else "model.safetensors"
    with safe_open(folder / shard, framework="pt") as f:
        return f.get_slice(key)[:, neuron].float()


def neuron_direction(model_id, model, layers, layer_idx, neuron):
    col = down_proj_column(model_id, layers, layer_idx, neuron, model)
    return col / col.norm()


class _StopWhen(StoppingCriteria):
    def __init__(self, should_stop):
        self.should_stop = should_stop

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.should_stop(), dtype=torch.bool, device=input_ids.device)


def stream(model, tok, inputs, max_new_tokens, layer=None, push=None, should_stop=lambda: False):
    """Yield the reply as it grows. If push is given, it is added to layer's output at every position."""
    handle = None
    if push is not None:
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + push.to(h.device, h.dtype)
            return (h, *out[1:]) if isinstance(out, tuple) else h

        handle = layer.register_forward_hook(hook)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer,
                  stopping_criteria=[_StopWhen(should_stop)])
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    text = ""
    try:
        for piece in streamer:
            text += piece
            yield text.strip()
    finally:
        thread.join()
        if handle:
            handle.remove()


def generate(model, tok, inputs, max_new_tokens, layer=None, push=None):
    text = ""
    for text in stream(model, tok, inputs, max_new_tokens, layer, push):
        pass
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--neuron", type=int, required=True)
    p.add_argument("--sign", choices=["up", "down", "both"], default="both")
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--gpu-mem", default="6.5GiB")
    args = p.parse_args()

    tok, model, layers = load(args.model, args.gpu_mem)
    layer = layers[args.layer]
    inputs = chat_inputs(tok, model, args.prompt)
    r_L = residual_norm(model, inputs, args.layer)
    direction = neuron_direction(args.model, model, layers, args.layer, args.neuron)

    print(f"model={args.model} layers={len(layers)} neurons/layer={layer.mlp.down_proj.weight.shape[1]} r_L={r_L:.1f}")
    print(f"\n[baseline]\n{generate(model, tok, inputs, args.max_new_tokens)}")

    signs = {"up": [1], "down": [-1], "both": [1, -1]}[args.sign]
    for sign in signs:
        for alpha in ALPHAS:
            reply = generate(model, tok, inputs, args.max_new_tokens, layer, sign * alpha * r_L * direction)
            if not looks_incoherent(reply) or alpha == ALPHAS[-1]:
                break
        label = "up" if sign > 0 else "down"
        print(f"\n[L{args.layer} n{args.neuron} {label} alpha={alpha}]\n{reply}")


if __name__ == "__main__":
    main()
