import logging
import sys
import time
from typing import List

from .config_loader import get_config
from .docker_wrapper import start_comfyui_server, stop_comfyui_server
from .llm_manager import (
    get_embeddings,
    start_llama_cpp_server,
    stop_llama_cpp_server,
    start_embed_server,
    stop_embed_server,
)
from .comfy_wrapper import generate_images_batch
from .check import check_images_batch
from .optimizer_utils import finalize_optimization
from .prompt_engine import expand, mutate
from .model import (
    GenerationRecord,
    _extract_scalar_score,
    PromptCandidate,
    select_diverse_parents,
)
from .translator import translate

logger = logging.getLogger(__name__)


def run_optimizer(
    intent: str,
    width: int = 4,
    deep: int = 6,
    keep: int = 4,
    output_dir: str = "output",
    image_count: int = 1,
    workflow: str = "",
    lora_name: str = "",
) -> List[GenerationRecord]:
    trajectory: List[GenerationRecord] = []
    start_time = time.time()
    top_keep = min(keep, width)
    decay_rate = 0.8
    rubric_weights = get_config().get("rubric_weights", {})
    embed_model = get_config()["models"]["embed"]

    start_llama_cpp_server()
    intent = translate(intent)
    logger.info("translated intent:\n" + intent)
    try:
        for gen in range(deep):
            logger.info(
                f"🔄 Generation {gen + 1}/{deep} | Time elapsed: {time.time() - start_time:.1f}s"
            )

            if not trajectory:
                candidates: List[PromptCandidate] = expand(intent, [], width)
                logger.info(f"Expanded {len(candidates)} candidates")
            else:
                mutate_len = (width * 2) // 3
                expand_len = width - mutate_len

                diverse_records = select_diverse_parents(
                    trajectory, top_keep, decay_rate=decay_rate
                )
                mutated = mutate(intent, diverse_records, mutate_len)
                logger.info(f"Mutated {len(mutated)} candidates")

                expanded = expand(
                    intent, [r.prompt for r in diverse_records], expand_len
                )
                logger.info(f"Expanded {len(expanded)} candidates")

                candidates = mutated + expanded

            stop_llama_cpp_server()
            logger.info(f"🧠 Embedding {len(candidates)} new candidates...")

            start_embed_server()
            prompt_list = [c.prompt for c in candidates]
            embeddings = get_embeddings(embed_model, prompt_list)
            stop_embed_server()

            start_comfyui_server()
            image_paths = generate_images_batch(prompt_list, workflow, lora_name)
            stop_comfyui_server()

            start_llama_cpp_server()
            rubrics = check_images_batch(image_paths, intent)

            records: List[GenerationRecord] = []
            for i, candidate in enumerate(candidates):
                rubric = rubrics[i]
                scalar = _extract_scalar_score(rubric, weights=rubric_weights)
                logger.info(f"🖼️ Image: {image_paths[i]} | Score: {scalar:.3f}")

                records.append(
                    GenerationRecord(
                        intent=intent,
                        prompt=candidate.prompt,
                        image_path=image_paths[i],
                        rubric=rubric,
                        scalar_score=scalar,
                        generation=gen,
                        embedding=embeddings[i],
                        generated_by=candidate.generated_by,
                        prompt_reasoning=candidate.reasoning,
                        scoring_reasoning=rubric.get("scoring_reasoning", ""),
                    )
                )
            trajectory.extend(records)
            trajectory.sort(key=lambda x: (x.scalar_score, x.generation), reverse=True)
        logger.info(
            f"🔄 Optimization finished | Total time: {time.time() - start_time:.1f}s"
        )
    except Exception as e:
        logger.error(f"something went wrong {e}")
    finally:
        stop_llama_cpp_server()
        stop_comfyui_server()
        stop_embed_server()
    finalize_optimization(trajectory, output_dir, image_count)
    return trajectory


def demo():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    intent = "anime girl repairing drone in garage, cyberpunk style"
    run_optimizer(intent)
