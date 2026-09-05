import logging

from .config_loader import get_config
from .llm_manager import call_llm

logger = logging.getLogger(__name__)

def translate(text: str) -> str:
    system_prompt = get_config()["prompts"]["translate_system"]
    llm_result = call_llm(system_prompt=system_prompt, prompt=text)
    logger.debug("text to translate:\n" + text)
    logger.debug("translation:\n" + llm_result["content"])
    logger.debug("translation reasoning:\n" + llm_result["reasoning"])
    return llm_result["content"]
