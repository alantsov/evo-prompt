import logging
import random
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
from .check import check_images_batch, get_custom_rubrics, extract_scalar_score
from .optimizer_utils import finalize_optimization
from .prompt_engine import expand, fix_prompt, mutate
from .model import (
    GenerationRecord,
    PromptCandidate,
    select_diverse_parents,
)
from .translator import translate

logger = logging.getLogger(__name__)


def run_optimizer(
    original_intent: str,
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
    try:
        for gen in range(deep):
            logger.info(
                f"🔄 Generation {gen + 1}/{deep} | Time elapsed: {time.time() - start_time:.1f}s"
            )
            intent = translate(original_intent)
            logger.info("translated intent:\n" + intent)
            custom_rubrics = get_custom_rubrics(intent)

            if not trajectory:
                candidates: List[PromptCandidate] = expand(intent, [], width)
                logger.info(f"Expanded {len(candidates)} candidates")
            else:
                refine_len = width // 4
                mutate_len = (width * 2) // 3
                expand_len = width - mutate_len - refine_len

                diverse_records = select_diverse_parents(
                    trajectory, top_keep, decay_rate=decay_rate
                )

                refined = []
                for i in range(refine_len):
                    refined += fix_prompt(random.choice(diverse_records), 1)
                logger.info(f"refined {len(refined)} candidates")

                mutated = []
                for i in range(mutate_len):
                    mutated += mutate(intent, random.choices(diverse_records, k=random.randint(1, len(diverse_records))), mutate_len)
                logger.info(f"Mutated {len(mutated)} candidates")

                expanded = expand(
                    intent, [r.prompt for r in diverse_records], expand_len
                )
                logger.info(f"Expanded {len(expanded)} candidates")

                candidates = refined + mutated + expanded

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
            rubrics = check_images_batch(image_paths, intent, custom_rubrics)

            records: List[GenerationRecord] = []
            for i, candidate in enumerate(candidates):
                rubric = rubrics[i]
                scalar = extract_scalar_score(rubric, custom_rubrics)
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
