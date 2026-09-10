# 🧬 Evo-Prompt

**Evo-Prompt** is an automated evolutionary loop designed for high-sensitivity image generation. While standard LLMs can handle basic prompts, specialized diffusion pipelines—particularly those utilizing **LoRAs or fine-tuned models**—are extremely sensitive to phrasing. **Evo-Prompt** solves this by automatically discovering the precise prompt structures required to trigger successful results in these fragile, complex workflows.

---

## ⚙️ How It Works

Evo-Prompt operates on a continuous evolutionary cycle:

1.  **Expand (Diversity):** The LLM takes a raw user intent and generates a diverse population of initial prompts.
2.  **Generate (Diffusion):** The prompts are sent to a local **ComfyUI** instance to generate a batch of images.
3.  **Evaluate (Vision):** A Vision-Language Model (served via a local **llama.cpp** container) acts as a judge, scoring each image against a multi-dimensional rubric (Fidelity, Composition, Style, etc.).
4.  **Mutate (Refinement):** The best-performing prompts and their specific critiques are fed back to the LLM. The LLM then "mutates" the prompts—fixing flaws and doubling down on strengths—to produce a superior next generation.

📝 **Note:** When running with `--cw`, the diffusion step is replaced by direct text generation. Evaluation focuses on narrative logic, style adherence, and genre conventions instead of visual fidelity.
---

## 🚀 Getting Started

### Prerequisites

*   **[Docker](https://www.docker.com/)** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed for GPU passthrough.
*   **[ComfyUI](https://github.com/comfyanonymous/ComfyUI):** Models & custom nodes ready locally (see Workflow Setup).
*   **[uv](https://github.com/astral-sh/uv):** A high-performance Python package and project manager.
*   **GPU:** NVIDIA GPU with sufficient VRAM to run the LLM, embedding, and diffusion models.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alantsov/evo-prompt.git
   cd evo-prompt
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

> 💡 **Model Management:** LLM and embedding models are automatically downloaded to `~/.cache/huggingface` on first run via the `llama.cpp` containers. No manual pulling is required.

### Usage

**Single Intent:**
```bash
uv run evoprompt -p "anime girl repairing drone in garage, cyberpunk style"
```

**Batch Intents (from file):**
Add one intent per line to `prompts.md`, then run:
```bash
uv run evoprompt -f prompts.md
```

**Creative Writing Mode (Text-Only):**
Enable `--cw` to switch to a text-only evolutionary loop. This mode uses CPU embeddings, skips ComfyUI entirely, and automatically loads `config/cw_prompts.yaml`.
```bash
uv run evoprompt --cw -p "Write a short sci-fi noir story about a detective who can dream in code, cyberpunk style"
```

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--cw` / `--creative-writing` | Enables text-only optimization with CW rubrics & CPU embeddings | `off` |

**Custom Configuration:**
```bash
uv run evoprompt \
  --infra-config ./configs/prod_infra.yaml \
  --prompts-config ./configs/v2_prompts.yaml \
  -p "cyberpunk cityscape at night" \
  -w 8 -g 6 -k 4 -n 3 -l debug \
  -y krea-2-advanced.json
```

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-p, --prompt` | Single intent string to optimize | *(required if no file)* |
| `-f, --file` | Path to file with intents (one per line) | `prompts.md` |
| `--infra-config` | Infrastructure YAML (models, containers, APIs) | `config/infra.yaml` |
| `--prompts-config` | Prompt templates & rubric weights YAML | `config/prompts.yaml` |
| `-w, --pop-size` | Population size per generation | `4` |
| `-g, --generations` | Number of optimization generations | `4` |
| `-k, --top-k` | Top-K parents to keep for mutation/crossover | `4` |
| `-n, --top-images` | Number of best images to save to output dir | `1` |
| `-y, --workflow` | ComfyUI workflow JSON template | `krea-2-basic.json` |
| `-l, --log-level` | Logging verbosity (`debug`, `info`, `warning`, `error`) | `info` |
| `-o, --output-dir` | Directory to save final images & logs | `output` |

---

### 🛠 Workflow Setup

To use the provided Krea-2 workflow, you must have the following models placed in your ComfyUI directory:

| Model Type | Filename | ComfyUI Directory | HF repository |
| :--- | :--- | :--- | :--- |
| **Diffusion (UNET)** | `krea2_turbo_bf16.safetensors` | `ComfyUI/models/diffusion_models/` | [Download](https://huggingface.co/Comfy-Org/Krea-2) |
| **CLIP** | `qwen3vl_4b_bf16.safetensors` | `ComfyUI/models/text_encoders/` | [Download](https://huggingface.co/Comfy-Org/Krea-2) |
| **VAE** | `qwen_image_vae.safetensors` | `ComfyUI/models/vae/` | [Download](https://huggingface.co/Comfy-Org/Krea-2) |
| **LoRA** | `your_lora.safetensors` | `ComfyUI/models/loras/` | your lora |

download them manually or use script:
```bash
uv run evoprompt-pull
```

---

## 🧪 Testing

The project includes a suite of basic unit tests. To run the tests, use:

```bash
uv run pytest
```

---

## 📂 Project Structure

*   `evo_prompt/`: Core logic (Optimization loop, ComfyUI/llama.cpp wrappers, LLM managers).
*   `workflows/`: JSON workflow templates for ComfyUI.
*   `data/`: Stores all generated images and the optimization trajectory (JSON logs of every generation).
*   `output/`: Contains the final "Best" image produced by the loop.
*   `config/`:
    *   `infra.yaml`: Runtime infrastructure (models, containers, APIs, timeouts)
    *   `prompts.yaml`: LLM templates, optimizer rules, and `rubric_weights`

---

## 🛠 Configuration

The configuration is split into two focused files to separate infrastructure concerns from prompt engineering:

### `config/infra.yaml`
Controls the pipeline's runtime environment:
*   **`models`**: HuggingFace GGUF paths for evaluation, optimization, and embeddings. Models are auto-cached in `~/.cache/huggingface`.
*   **`llama_cpp` / `comfyui`**: API endpoints, workflow files, and model directories.
*   **`containers`**: Docker image names, ports, health-check URLs, and timeouts.

### `config/prompts.yaml`
Controls the "brain" and scoring logic of the agent:
*   **`evaluator_system` / `optimizer_*`**: LLM system prompts, rules, and template placeholders.
*   **`rubric_weights`**: Numerical weights for evaluation criteria. Placed here alongside evaluator prompts to keep scoring criteria tightly coupled with evaluation logic. Example:
  ```yaml
  rubric_weights:
    "Intent Fidelity": 6.0
    "Logical & Physical Coherence": 2.0
    "Visual Quality": 1.0
    "Aesthetic Composition": 1.0
  ```

---

## ⚖️ License & Model Disclaimer

This project's source code is licensed under the [MIT License](LICENSE). 

**Third-Party Model Weights**
The MIT license **does not cover** the Krea 2 diffusion model, CLIP, VAE, LoRA weights, or any other third-party assets referenced in this repository. These models are developed and distributed by their respective creators. By using this project, you acknowledge that:
- You are solely responsible for legally acquiring the required model weights.
- You must comply with all terms, licenses, and usage restrictions imposed by the original model publishers.
- The project author provides no warranty, support, or liability regarding third-party models, their outputs, or their compliance with local regulations.

Please review the official licensing terms of any models you integrate with this pipeline.
