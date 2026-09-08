import logging
import math
import random
import time

from .check import check_images_batch, get_custom_rubrics, extract_scalar_score
from .config_loader import get_config
from .docker_wrapper import start_comfyui_server, stop_comfyui_server
from .model import GenerationRecord, select_diverse_parents_mmr
from .comfy_wrapper import generate_images_batch
from .llm_manager import (
    start_embed_server,
    stop_embed_server,
    get_embeddings,
    start_llama_cpp_server,
    stop_llama_cpp_server,
)
from .optimizer_utils import finalize_optimization
from .prompt_engine import expand, crossover, fix_prompt, mutate
from .translator import translate

logger = logging.getLogger(__name__)


def optimize(
    original_intent: str,
    width: int = 4,
    deep: int = 6,
    keep: int = 4,
    output_dir: str = "output",
    image_count: int = 1,
    workflow: str = "",
    lora_name: str = "",
) -> list[GenerationRecord]:
    trajectory = []
    last_generation = []
    start_time = time.time()
    try:
        start_llama_cpp_server()
        for gen in range(deep):
            logger.info(
                f"🔄 Generation {gen + 1}/{deep} | Time elapsed: {time.time() - start_time:.1f}s"
            )
            intent = translate(original_intent)
            logger.info("translated intent:\n" + intent)
            custom_rubrics = get_custom_rubrics(intent)
            new_prompts_with_reasoning = []
            parents = last_generation
            parents = sorted(parents, key=lambda r: r.scalar_score, reverse=True)
            parents = parents[:keep]
            if len(parents) > 0:
                current_parents_for_fix = parents[:random.randint(1, keep)]
                for record in current_parents_for_fix:
                    new_prompts_with_reasoning += fix_prompt(record, 1)
                logger.info(
                    f"Generated {len(new_prompts_with_reasoning)} prompts by fix_prompt"
                )
            if len(new_prompts_with_reasoning) < width and len(parents) > 1:
                for i in range((width - len(new_prompts_with_reasoning)) // 2):
                    current_parents = random.choices(parents, k=min(2, len(parents)))
                    mutate_prompts = mutate(intent, current_parents, 1)
                    new_prompts_with_reasoning += mutate_prompts
                    logger.info(f"Generated {len(mutate_prompts)} prompts by mutate")
            previous_records = select_diverse_parents_mmr(
                trajectory, random.randint(2, width)
            )
            previous_prompts = [r.prompt for r in previous_records]
            expand_prompts = []
            if parents and previous_records:
                for i in range(2):
                    first_record = random.choice(parents)
                    second_options = [pr for pr in previous_records if pr.prompt != first_record.prompt]
                    if not second_options:
                        second_options = parents
                    second_record = random.choice(second_options)
                    crossover_prompts = crossover(intent, [first_record, second_record], 1)
                    logger.info(f"Generated {len(crossover_prompts)} prompts by crossover")
                    new_prompts_with_reasoning += crossover_prompts
            while len(new_prompts_with_reasoning) < width:
                previous_prompts_options = [
                    [],
                    previous_prompts,
                    [p.prompt for p in expand_prompts],
                    previous_prompts + [p.prompt for p in expand_prompts]
                ]
                expected_count = min(6, random.randint(1, width - len(new_prompts_with_reasoning)))
                current_expand_prompts = expand(
                    intent,
                    random.choice(previous_prompts_options),
                    expected_count
                )
                expand_prompts += current_expand_prompts
                new_prompts_with_reasoning += current_expand_prompts
                logger.info(f"Generated {len(current_expand_prompts)} prompts by expand")
            new_prompts_with_reasoning = new_prompts_with_reasoning[:width]
            stop_llama_cpp_server()
            start_embed_server()
            new_prompts_text = [pr.prompt for pr in new_prompts_with_reasoning]
            embeddings = get_embeddings(
                get_config()["models"]["embed"], new_prompts_text
            )
            stop_embed_server()
            start_comfyui_server()
            images = generate_images_batch(new_prompts_text, workflow, lora_name)
            stop_comfyui_server()
            start_llama_cpp_server()
            rubrics = check_images_batch(images, intent, custom_rubrics)
            for i, r in enumerate(rubrics):
                logger.debug(f'rubric {i}\n{r}')
            last_generation = []
            for i in range(len(new_prompts_with_reasoning)):
                prompt = new_prompts_with_reasoning[i].prompt
                generation_reasoning = new_prompts_with_reasoning[i].reasoning
                rubric = rubrics[i]
                score = extract_scalar_score(rubric, custom_rubrics)
                image = images[i]
                logger.info(f"🖼️ Image: {image} | Score: {score:.3f} | embed len {len(embeddings[i])} | embed start {embeddings[i][:5]}")
                record = GenerationRecord(
                    generation=gen,
                    embedding=embeddings[i],
                    image_path=image,
                    intent=intent,
                    prompt=prompt,
                    scalar_score=score,
                    rubric=rubric,
                    generated_by=new_prompts_with_reasoning[i].generated_by,
                    prompt_reasoning=generation_reasoning,
                    scoring_reasoning=rubric["scoring_reasoning"],
                )
                last_generation.append(record)
            trajectory += last_generation
            trajectory.sort(key=lambda x: (x.scalar_score, x.generation), reverse=True)
        stop_llama_cpp_server()
        logger.info(
            f"🔄 Optimization finished | Total time: {time.time() - start_time:.1f}s"
        )
    finally:
        stop_llama_cpp_server()
        stop_comfyui_server()
        stop_embed_server()
    finalize_optimization(trajectory, output_dir, image_count)
    return sorted(
        trajectory, key=lambda r: (r.generation, r.scalar_score), reverse=True
    )
