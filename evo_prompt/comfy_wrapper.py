import datetime
import json
import random
import requests
import threading
import time
import logging
from pathlib import Path
from typing import Any, List
from .config_loader import get_config, BASE_DIR

logger = logging.getLogger(__name__)


def _replace_in_json(obj, placeholder, replacement) -> Any:
    if isinstance(obj, dict):
        return {
            k: _replace_in_json(v, placeholder, replacement) for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_replace_in_json(item, placeholder, replacement) for item in obj]
    elif (
        isinstance(obj, str)
        and isinstance(placeholder, str)
        and isinstance(replacement, str)
    ):
        return obj.replace(placeholder, replacement)
    elif isinstance(obj, int) and obj == placeholder:
        return replacement
    return obj


def _prepare_workflow(workflow: dict, prompt: str, lora_name: str) -> dict:
    prompt_placeholder = "{{PROMPT}}"
    lora_placeholder = "{{LORA_NAME}}"
    workflow = _replace_in_json(workflow, prompt_placeholder, prompt)
    workflow = _replace_in_json(workflow, lora_placeholder, lora_name)
    for key, value in workflow.items():
        if value.get("class_type", "") == "KSampler":
            workflow[key]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
    return workflow


def _worker(
    prompt: str,
    index: int,
    total: int,
    prompts: List[str],
    results: List[str],
    workflow_template_path: Path,
    lora_name: str = "",
):
    try:
        api_url = get_config()["comfyui"]["api_url"]

        with open(workflow_template_path) as f:
            workflow = json.load(f)

        workflow = _prepare_workflow(workflow, prompt=prompt, lora_name=lora_name)

        logger.info(f"🚀 Submitting task {index + 1}/{total}")

        response = requests.post(f"{api_url}/prompt", json={"prompt": workflow})
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]

        image_filename = None
        while True:
            history_resp = requests.get(f"{api_url}/history/{prompt_id}")
            history_resp.raise_for_status()
            history = history_resp.json()

            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id in outputs:
                    if (
                        "images" in outputs[node_id]
                        and len(outputs[node_id]["images"]) > 0
                    ):
                        image_filename = outputs[node_id]["images"][0]["filename"]
                        break
                break

            time.sleep(1)

        if not image_filename:
            raise Exception(f"No image found in ComfyUI history for prompt {prompt_id}")

        logger.debug(
            f"📥 Downloading image for task {index + 1}/{total}: {image_filename}"
        )
        img_resp = requests.get(
            f"{api_url}/view", params={"filename": image_filename, "type": "output"}
        )
        img_resp.raise_for_status()

        uniq_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        new_filename = BASE_DIR / "data" / f"{uniq_suffix}.png"

        with open(new_filename, "wb") as f:
            f.write(img_resp.content)

        results[index] = str(new_filename)
        logger.info(f"✅ Task {index + 1}/{total} complete: {new_filename.name}")

    except Exception as e:
        logger.error(f"❌ Error in task {index + 1}/{total}: {e}")
        results[index] = ""


def generate_image(prompt) -> str:
    return generate_images_batch([prompt])[0]


def generate_images_batch(
    prompts: List[str], workflow_filename: str = "", lora_name: str = ""
) -> List[str]:
    (BASE_DIR / "data").mkdir(exist_ok=True)
    if not workflow_filename:
        workflow_filename = get_config()["comfyui"]["workflow_file"]
    workflow_template_path = BASE_DIR / "workflows" / workflow_filename

    results = [""] * len(prompts)
    threads = []

    for i, prompt in enumerate(prompts):
        t = threading.Thread(
            target=_worker,
            args=(
                prompt,
                i,
                len(prompts),
                prompts,
                results,
                workflow_template_path,
                lora_name,
            ),
        )
        threads.append(t)
        t.start()
        time.sleep(0.1)

    for t in threads:
        t.join()

    return results
