import json
import logging
from typing import List, Dict

from .llm_manager import call_llm
from .config_loader import get_config

logger = logging.getLogger(__name__)


def _parse_llm_json(response: dict) -> dict:
    json_output = response["content"]
    scoring_reasoning = response["reasoning"]

    if json_output.startswith("```"):
        json_output_lines = json_output.split("\n")
        json_output = "\n".join(json_output_lines[1:-1])
    try:
        rubric = json.loads(json_output)
        rubric["scoring_reasoning"] = scoring_reasoning
        return rubric
    except Exception:
        logger.error(f"Failed to parse LLM output as JSON: {json_output}")
        return {"scoring_reasoning": scoring_reasoning}


def check_image(image_filename: str, intent: str) -> dict:
    cfg = get_config()
    user_text = cfg["prompts"]["evaluator_user"].format(intent=intent)
    response = call_llm(
        model=cfg["models"]["evaluator"],
        prompt=user_text,
        system_prompt=cfg["prompts"]["evaluator_system"],
        image=image_filename,
        temperature=0.0,
    )
    return _parse_llm_json(response)


def check_images_batch(image_filenames: List[str], intent: str) -> List[dict]:
    return [check_image(img, intent) for img in image_filenames]


def check_text(text: str, intent: str) -> dict:
    """Evaluates creative writing text against a single string intent."""
    cfg = get_config()
    user_text = cfg["prompts"]["evaluator_user"].format(intent=intent, text=text)
    response = call_llm(
        model=cfg["models"]["evaluator"],
        prompt=user_text,
        system_prompt=cfg["prompts"]["evaluator_system"],
        temperature=0.0,
    )
    return _parse_llm_json(response)
