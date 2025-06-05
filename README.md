# Tenos Flux Conditioning Nodes 🚀

Custom **ComfyUI** nodes that super‑charge [FluxDev](https://github.com/huggingface/diffusers?tab=readme-ov-file#the-flux-scheduler)‑style diffusion workflows.  
If “_prompt engineering_” is your Katana, these nodes are the enchanted runes you carve into the blade to make it slice *cleaner, faster, smarter*.

---

## ✨ Why you might actually care

* **Layer‑aware conditioning** – target _specific_ Flux blocks (encoder, base, decoder, out) with per‑layer multipliers  
* **Blend, add, or spatially concat** two completely different prompt stacks without losing context  
* **Built‑in sanity checks & tensor normalization** so you stop exploding your VRAM (and your temperament)  
* **Verbose summaries** pushed back to the ComfyUI front‑end via `PromptServer` – know exactly what each node did in plain English (and a dash of snark)  
* Designed for **advanced LoRA / CLIP‑T5 workflows** where you juggle dozens of conditioning tensors like a caffeinated circus act

_Node list and micro‑pitch below was auto‑extracted from the source 📜_ :contentReference[oaicite:0]{index=0}

| Node | I/O | Elevator Pitch |
|------|-----|---------------|
| **`TenosaiFluxConditioningApply`** | conditioning → layered conditioning | Apply per‑block strength multipliers to **one** conditioning list. |
| **`TenosaiFluxConditioningBlend`** | cond1 + cond2 → layered conditioning | Blend two conditioning stacks **per block** with independent ratios. |
| **`TenosaiFluxConditioningSummedBlend`** | cond1 + cond2 → single conditioning | Same as above, but outputs a _single_ summed tensor (keeps things slim). |
| **`TenosaiFluxConditioningAddScaledSum`** | cond1 + cond2 → single conditioning | Scale cond2 per block, sum it, add onto total cond1. Think “add a spicy topping”. |
| **`TenosaiFluxConditioningSpatialWeightedConcat`** | cond1 + cond2 → single conditioning | Concatenate cond1 & scaled‑cond2 along the **sequence dimension** for spatial hacks. |

---

## 🛠 Installation

```bash
# 1. Clone (or submodule) into your ComfyUI custom-nodes folder
git clone https://github.com/YourUsername/tenos-flux-conditioning-nodes.git \
      ~/ComfyUI/custom_nodes/TenosFluxConditioning

# 2. (Optional) Create a fresh venv & install PyTorch if you somehow skipped that step 🤨

# 3. Fire up ComfyUI and look for the “Tenos.ai/Conditioning” category.


> **Requires**:
>
> * PyTorch 2.1+ (CUDA or ROCm)
> * ComfyUI (latest main branch)
> * A model/scheduler that actually uses Flux blocks (e.g. *FluxDev* or similar)

---

## 🚦 Quick‑Start

1. **Load your prompts** using whatever tokenizer node floats your boat.
2. **Drag in** `TenosaiFluxConditioningApply` and connect your conditioning.
3. Dial in block multipliers (e.g. `encoder = 0.8`, `base = 1.2`, etc.).
4. Feed the output straight into your Flux‑compatible sampler.
5. Profit.
```
### Example flow (pseudo‑graph)

```
CLIPTextEncode  →  TenosaiFluxConditioningApply  →  FluxSampler
                              ↑
               Adjust multipliers per block in UI
```

---

## 📚 Detailed Docs

* Each node has **tooltips** on every slider/checkbox – hover like it’s 1999.
* Output strings are accessible in ComfyUI’s “impact‑node‑feedback” panel; copy‑paste for debugging.
* Shapes are automatically padded to the max sequence length across inputs. No more shape mismatches.
