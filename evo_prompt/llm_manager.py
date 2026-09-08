import logging
import base64
import requests
from .docker_wrapper import get_llama_manager, get_embed_manager
from .config_loader import get_config

logger = logging.getLogger(__name__)


def start_llama_cpp_server():
    get_llama_manager().start()


def stop_llama_cpp_server():
    get_llama_manager().stop()


def start_embed_server():
    get_embed_manager().start()


def stop_embed_server():
    get_embed_manager().stop()


def call_llm(model="", image=None, prompt=None, system_prompt=None, temperature=1.0):
    host = get_config()["llama_cpp"]["llm_api_url"]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content = []
    if image:
        with open(image, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
        )
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "repeat_penalty": 1.00,
        "top_k": 20,
        "top_p": 0.95,
        "stream": False,
    }
    try:
        logger.debug(f"model={model}")
        logger.debug(f"system prompt={system_prompt}")
        logger.debug(f"prompt={prompt}")
        logger.debug(f"image={image}")
        response = requests.post(
            f"{host}/v1/chat/completions", json=payload, timeout=300
        )
        response.raise_for_status()
        data = response.json()

        msg_content = data["choices"][0]["message"]["content"].strip()
        msg_reasoning = (
            data["choices"][0]["message"].get("reasoning_content", "").strip()
        )

        logger.debug(f"llm thinking:\n{msg_reasoning}")
        logger.debug(f"llm answer:\n{msg_content}")
        return {"content": msg_content, "reasoning": msg_reasoning}
    except Exception as e:
        logger.error(f"❌ LLM request failed: {e}")
        return {"content": "", "reasoning": ""}


def get_embeddings(model: str, texts: list[str]) -> list[list[float]]:
    host = get_config()["llama_cpp"]["embed_api_url"]

    logger.debug(f"get_embeddings texts: {texts}")
    embeddings = []
    try:
        response = requests.post(
            f"{host}/v1/embeddings", json={"model": model, "input": texts}, timeout=300
        )
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        logger.debug(f"✅ Retrieved {len(embeddings)} embeddings.")
    except Exception as e:
        logger.error(f"❌ Embedding request failed: {e}")
        embeddings = [[] for _ in texts]
    return embeddings
