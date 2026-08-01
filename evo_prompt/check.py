import json
import logging
from typing import List
from .llm_manager import call_llm
from .config_loader import get_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = get_config()["prompts"]["evaluator_system"]
USER_MESSAGE_TEMPLATE = get_config()["prompts"]["evaluator_user"]


def check_image(image_filename: str, intent: str) -> dict:
    user_text = USER_MESSAGE_TEMPLATE.format(intent=intent)
    response = call_llm(
        model=get_config()["models"]["evaluator"],
        prompt=user_text,
        system_prompt=SYSTEM_PROMPT,
        image=image_filename,
        temperature=0.0,
    )

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


def check_images_batch(image_filenames: List[str], intent: str) -> List[dict]:
    return [check_image(img, intent) for img in image_filenames]
