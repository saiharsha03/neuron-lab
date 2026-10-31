r"""Local UI for single-neuron steering.

Run:  .venv\Scripts\python.exe app.py [model_id]   then open http://127.0.0.1:7860
"""

import csv
import gc
import random
import sys
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch

from steer import chat_inputs, load, neuron_direction, residual_norm, stream

# Only one model fits on the 8 GB GPU at a time, so switching unloads the current one.
MODELS = ["Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen3-8B"]
START_MODEL = sys.argv[1] if len(sys.argv) > 1 else MODELS[0]
if START_MODEL not in MODELS:
    MODELS.append(START_MODEL)

LOG = Path(__file__).with_name("runs.csv")
LOG_FIELDS = ["time", "model", "prompt", "layer", "neuron", "push", "alpha", "reply"]

current = {}
gpu_lock = threading.Lock()  # a run and a model switch must never overlap
baselines = {}
stop_flags = {}  # one per browser tab / API client, so Stop only cancels your own run


def load_model(model_id):
    current.clear()
    gc.collect()
    torch.cuda.empty_cache()
    tok, model, layers = load(model_id)
    current.update(id=model_id, tok=tok, model=model, layers=layers,
                   n_layers=len(layers), n_neurons=layers[0].mlp.down_proj.weight.shape[1])


def model_info():
    return f"**{current['id']}** · {current['n_layers']} layers × {current['n_neurons']:,} neurons"


def stop_flag_for(request):
    return stop_flags.setdefault(request.session_hash, threading.Event())


def log_run(row):
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def history():
    if not LOG.exists():
        return []
    with LOG.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [[r["time"], r["model"].split("/")[-1], r["layer"], r["neuron"], r["push"], r["alpha"], r["reply"]]
            for r in reversed(rows)]


def switch_model(model_id):
    with gpu_lock:
        if current.get("id") != model_id:
            load_model(model_id)
    n_layers, n_neurons = current["n_layers"], current["n_neurons"]
    return (model_info(),
            gr.update(maximum=n_layers - 1, value=n_layers // 2),
            gr.update(maximum=n_neurons - 1, value=min(1000, n_neurons - 1), label=f"Neuron (0–{n_neurons - 1})"))


def run(prompt, layer, neuron, alpha, push, max_tokens, request: gr.Request):
    stop_flag = stop_flag_for(request)
    stop_flag.clear()
    with gpu_lock:
        model_id, tok, model, layers = current["id"], current["tok"], current["model"], current["layers"]
        layer, neuron, max_tokens = int(layer), int(neuron), int(max_tokens)
        inputs = chat_inputs(tok, model, prompt)
        r_L = residual_norm(model, inputs, layer)
        direction = neuron_direction(model_id, model, layers, layer, neuron)
        info = f"r_L = {r_L:.1f} · push size = {alpha * r_L:.1f}"
        out = {"baseline": "", "up": "", "down": ""}

        def frame():
            return info, out["baseline"], out["up"], out["down"], gr.skip()

        key = (model_id, prompt, max_tokens)
        if key in baselines:
            out["baseline"] = baselines[key]
        else:
            for out["baseline"] in stream(model, tok, inputs, max_tokens, should_stop=stop_flag.is_set):
                yield frame()
            if not stop_flag.is_set():
                baselines[key] = out["baseline"]

        for name, sign in [("up", 1), ("down", -1)]:
            if push not in (name, "both") or stop_flag.is_set():
                continue
            vec = sign * alpha * r_L * direction
            for out[name] in stream(model, tok, inputs, max_tokens, layers[layer], vec, stop_flag.is_set):
                yield frame()
            log_run({"time": datetime.now().strftime("%H:%M:%S"), "model": model_id, "prompt": prompt,
                     "layer": layer, "neuron": neuron, "push": name, "alpha": alpha, "reply": out[name]})

    yield info, out["baseline"], out["up"], out["down"], history()


def stop_run(request: gr.Request):
    stop_flag_for(request).set()


def random_neuron():
    return random.randrange(current["n_layers"]), random.randrange(current["n_neurons"])


load_model(START_MODEL)

with gr.Blocks(title="Neuron Lab") as app:
    gr.Markdown("## Neuron Lab\nPush one MLP neuron and see who the model says it is.")
    with gr.Row():
        with gr.Column(scale=1):
            model_pick = gr.Dropdown(MODELS, value=START_MODEL, label="Model")
            model_md = gr.Markdown(model_info())
            prompt = gr.Textbox("Who are you?", label="Prompt")
            layer = gr.Slider(0, current["n_layers"] - 1, value=current["n_layers"] // 2, step=1, label="Layer")
            neuron = gr.Number(1000, precision=0, minimum=0, maximum=current["n_neurons"] - 1,
                               label=f"Neuron (0–{current['n_neurons'] - 1})")
            alpha = gr.Slider(0.25, 2.0, value=1.25, step=0.25, label="Strength α (× residual norm)")
            push = gr.Radio(["both", "up", "down"], value="both", label="Push")
            max_tokens = gr.Slider(10, 160, value=40, step=10, label="Max new tokens")
            with gr.Row():
                go = gr.Button("Run", variant="primary")
                rnd = gr.Button("Random neuron")
                stop = gr.Button("Stop")
            info = gr.Markdown()
        with gr.Column(scale=2):
            base_out = gr.Textbox(label="Baseline (no push)", lines=4)
            up_out = gr.Textbox(label="Pushed up", lines=4)
            down_out = gr.Textbox(label="Pushed down", lines=4)
    runs = gr.Dataframe(history(), headers=["time", "model", "layer", "neuron", "push", "α", "reply"],
                        label=f"All runs (saved to {LOG.name})", wrap=True)

    model_pick.change(switch_model, model_pick, [model_md, layer, neuron])
    go.click(run, [prompt, layer, neuron, alpha, push, max_tokens], [info, base_out, up_out, down_out, runs])
    rnd.click(random_neuron, None, [layer, neuron])
    stop.click(stop_run)

if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Monochrome())
