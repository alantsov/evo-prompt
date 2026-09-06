import datetime
import logging
import time
import json
from pathlib import Path
from typing import List

from .config_loader import BASE_DIR, get_config
from .check import check_text, extract_scalar_score
from .docker_wrapper import (
    start_embed_cpu_server,
    stop_embed_cpu_server,
)
from .llm_manager import (
    start_llama_cpp_server,
    stop_llama_cpp_server,
    get_embeddings,
)
from .model import (
    GenerationRecord,
    select_diverse_parents,
    PromptCandidate,
)
from .prompt_engine import expand, crossover, fix_prompt, mutate

logger = logging.getLogger(__name__)


def optimize_cw(
    intent: str,
    width: int = 4,
    deep: int = 3,
    keep: int = 2,
    output_dir: str = "output_cw",
    top_outputs: int = 1,
) -> List[GenerationRecord]:
    rubric_weights = get_config().get("rubric_weights", {})
    embed_model = get_config()["models"]["embed"]

    trajectory: List[GenerationRecord] = []
    last_generation: List[GenerationRecord] = []
    start_time = time.time()

    logger.info("🚀 Starting LLM & Embed (CPU) servers for CW optimization...")
    start_llama_cpp_server()
    start_embed_cpu_server()

    try:
        for gen in range(deep):
            logger.info(
                f"🔄 CW Generation {gen + 1}/{deep} | Time elapsed: {time.time() - start_time:.1f}s"
            )
            new_candidates: List[PromptCandidate] = []
            top_rec = sorted(last_generation, key=lambda r: r.scalar_score, reverse=True)[:keep]
            new_candidates += mutate(intent, top_rec, pop=keep)
            remaining = (width - len(new_candidates)) // 2
            if remaining > 0 and len(top_rec) >= 2:
                top_parents = top_rec[:keep]
                new_candidates += crossover(intent=intent, records=top_parents, pop=remaining)

            remaining = width - len(new_candidates)
            if remaining > 0:
                diverse = select_diverse_parents(
                    trajectory, top_k=min(keep, len(trajectory)), ensure_top_k=False
                )
                prev_texts = [r.prompt for r in diverse]
                new_candidates += expand(intent=intent, previous_prompts=prev_texts, pop=remaining)

            new_candidates = new_candidates[: width]

            texts = [c.prompt for c in new_candidates]
            embeddings = get_embeddings(embed_model, texts)

            rubrics = [check_text(t, intent) for t in texts]

            current_records: List[GenerationRecord] = []
            for i, cand in enumerate(new_candidates):
                score = extract_scalar_score(rubrics[i])
                record = GenerationRecord(
                    intent=intent,
                    prompt=cand.prompt,
                    image_path="",
                    rubric=rubrics[i],
                    scalar_score=score,
                    generation=gen,
                    embedding=embeddings[i],
                    generated_by=cand.generated_by,
                    prompt_reasoning=cand.reasoning,
                    scoring_reasoning=rubrics[i].get("scoring_reasoning", ""),
                )
                current_records.append(record)
                logger.info(f"📝 Text | Score: {score:.3f} | Gen: {gen}")

            last_generation = current_records
            trajectory.extend(current_records)
            trajectory.sort(key=lambda x: (x.scalar_score, x.generation), reverse=True)

    except Exception as e:
        logger.error(f"❌ CW Optimization failed: {e}")
    finally:
        stop_llama_cpp_server()
        stop_embed_cpu_server()
        logger.info("🛑 LLM & Embed servers stopped.")

    _save_cw_results(trajectory, output_dir, top_outputs)
    return trajectory


def _save_cw_results(trajectory: List[GenerationRecord], output_dir: str, top_n: int):
    if not trajectory:
        logger.warning("No valid CW generations produced.")
        return

    out_path = BASE_DIR / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    uniq_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    traj_path = out_path / f"trajectory_cw_{uniq_suffix}.json"
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in trajectory], f, indent=2, ensure_ascii=False)
    logger.info(f"💾 CW Trajectory saved to: {traj_path}")

    for i, rec in enumerate(trajectory[:top_n]):
        txt_path = out_path / f"top_{i+1}_text_{uniq_suffix}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Score: {rec.scalar_score:.3f}\n# Intent: {rec.intent}\n\n{rec.prompt}")
        logger.info(f"🏆 TOP {i+1} TEXT:\n{rec.prompt}\n📍 Saved to: {txt_path}")
