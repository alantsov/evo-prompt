import logging
from typing import List, Dict, Any

from .config_loader import get_config
from .llm_manager import call_llm
from .model import GenerationRecord, PromptCandidate, parse_prompts, format_feedback

logger = logging.getLogger(__name__)


def _format_response(
    response: Dict[str, Any], target_count: int
) -> List[PromptCandidate]:
    raw = response.get("content", "")
    reasoning = response.get("reasoning", "")
    prompts = parse_prompts(raw, target_count)
    return [PromptCandidate(prompt=p, reasoning=reasoning) for p in prompts]


def expand(intent: str, previous_prompts: List[str], pop: int) -> List[PromptCandidate]:
    user_msg = get_config()["prompts"]["optimizer_expand_user"].format(
        intent=intent, previous_prompts="\n\n".join(previous_prompts), pop=pop
    )
    sys_prompt = get_config()["prompts"]["optimizer_expand_system"].format(
        pop=pop, rules=get_config()["prompts"]["optimizer_rules"]
    )
    response = call_llm(
        model=get_config()["models"]["optimizer"],
        system_prompt=sys_prompt,
        prompt=user_msg,
    )
    return _format_response(response, pop)


def crossover(
    intent: str, records: List[GenerationRecord], pop: int
) -> List[PromptCandidate]:
    sys_prompt = get_config()["prompts"]["optimizer_crossover_system"].format(
        pop=pop, rules=get_config()["prompts"]["optimizer_rules"]
    )
    previous_prompts = [r.prompt for r in records]
    user_msg = "\n\n".join(previous_prompts)
    response = call_llm(
        model=get_config()["models"]["optimizer"],
        system_prompt=sys_prompt,
        prompt=user_msg,
    )
    return _format_response(response, pop)


def fix_prompt(record: GenerationRecord, pop: int) -> List[PromptCandidate]:
    user_msg = get_config()["prompts"]["optimizer_fix_user"].format(
        intent=record.intent, prompt=record.prompt
    )
    sys_prompt = get_config()["prompts"]["optimizer_fix_system"].format(
        pop=pop, rules=get_config()["prompts"]["optimizer_rules"]
    )
    response = call_llm(
        model=get_config()["models"]["optimizer"],
        image=record.image_path,
        system_prompt=sys_prompt,
        prompt=user_msg,
    )
    return _format_response(response, pop)


def mutate(
    intent: str, diverse_records: List[GenerationRecord], pop: int
) -> List[PromptCandidate]:
    feedback = format_feedback(intent, diverse_records, pop)
    sys_prompt = get_config()["prompts"]["optimizer_mutate_system"].format(
        pop=pop, rules=get_config()["prompts"]["optimizer_rules"]
    )
    response = call_llm(
        model=get_config()["models"]["optimizer"],
        system_prompt=sys_prompt,
        prompt=feedback,
    )
    return _format_response(response, pop)
