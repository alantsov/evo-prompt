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
    except Exception as e:
        logger.error(f"Failed to parse LLM output as JSON: {json_output}")
        logger.error(e)
        return {"scoring_reasoning": scoring_reasoning}

def get_custom_rubrics(intent) -> dict | None:
    cfg = get_config()
    default_rubrics = cfg["default_rubrics"]
    default_rubrics = json.dumps(default_rubrics, indent=4)
    system_prompt = cfg["prompts"]["create_custom_rubrics_system"]
    user_prompt = cfg["prompts"]["create_custom_rubrics_user"].format(default_rubrics=default_rubrics, intent=intent)
    response = call_llm(prompt=user_prompt, system_prompt=system_prompt)
    try:
        json_output = response["content"]
        logger.debug('generated custom rubrics:\n'+json_output)
        rubrics = json.loads(json_output)
        return rubrics
    except Exception:
        return None

def check_image(image_filename: str, intent: str, rubrics: dict | None = None) -> dict:
    cfg = get_config()
    rubrics = rubrics or cfg["default_rubrics"]
    rubrics_description = "\n".join([ f"- {k}: {rubrics[k]["description"]}" for k in rubrics.keys() ])
    output_example = { k: {"score": 0, "note": "one-sentence visible evidence"} for k in rubrics.keys() }
    output_example = json.dumps(output_example, indent=4)
    system_prompt = cfg["prompts"]["evaluator_system"].format(rubrics_description=rubrics_description, output_example=output_example)
    user_text = cfg["prompts"]["evaluator_user"].format(intent=intent)
    response = call_llm(
        model=cfg["models"]["evaluator"],
        prompt=user_text,
        system_prompt=system_prompt,
        image=image_filename,
        temperature=0.0,
    )
    return _parse_llm_json(response)

def extract_scalar_score(rubric, grading_rubrics, max_rubric_score: float = 2.0) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    logger.debug(f"extract_scalar_score, rubric:\n{rubric}")
    logger.debug(f"extract_scalar_score, grading_rubrics:\n{grading_rubrics}")
    for k, v in rubric.items():
        try:
            raw_score = v.get('score', 0)
            weight = grading_rubrics[k]["weight"]
        except Exception:
            continue
        weighted_sum += (raw_score / max_rubric_score) * weight
        total_weight += weight
    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 0

def check_images_batch(image_filenames: List[str], intent: str, rubrics: dict | None = None) -> List[dict]:
    return [check_image(img, intent, rubrics) for img in image_filenames]


def check_text(text: str, intent: str, rubrics: dict | None) -> dict:
    cfg = get_config()
    rubrics = rubrics or cfg["default_rubrics"]
    rubrics_description = "\n".join([ f"- {k}: {rubrics[k]["description"]}" for k in rubrics.keys() ])
    output_example = { k: {"score": 0, "note": "short evidence"} for k in rubrics.keys() }
    output_example = json.dumps(output_example, indent=4)
    system_prompt = cfg["prompts"]["evaluator_system"].format(rubrics_description=rubrics_description, output_example=output_example)
    user_text = cfg["prompts"]["evaluator_user"].format(intent=intent, text=text)
    response = call_llm(
        model=cfg["models"]["evaluator"],
        prompt=user_text,
        system_prompt=system_prompt,
        temperature=0.0,
    )
    return _parse_llm_json(response)
