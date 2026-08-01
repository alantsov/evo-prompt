import copy
from dataclasses import dataclass, asdict
import json
import logging
import math
import os
from typing import List, Dict, Any

from .config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationRecord:
    intent: str
    prompt: str
    image_path: str
    rubric: Dict[str, Any]
    scalar_score: float
    generation: int
    embedding: List[float]
    prompt_reasoning: str = ""
    scoring_reasoning: str = ""


@dataclass
class PromptCandidate:
    prompt: str
    reasoning: str = ""


def parse_prompts(raw: str, target_count: int) -> List[str]:
    return [p.strip() for p in raw.split("\n") if len(p.strip()) > 15][:target_count]


def save_trajectory(
    trajectory: List[GenerationRecord], filepath: str, fmt: str = "json"
) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    trajectory_copy = copy.deepcopy(trajectory)
    for r in trajectory_copy:
        r.embedding = []

    with open(filepath, "w", encoding="utf-8") as f:
        if fmt.lower() == "json":
            json.dump(
                [asdict(r) for r in trajectory_copy], f, indent=2, ensure_ascii=False
            )
        elif fmt.lower() == "jsonl":
            for r in trajectory_copy:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        else:
            raise ValueError("fmt must be 'json' or 'jsonl'")

    logger.info(f"💾 Trajectory saved to: {filepath} ({len(trajectory_copy)} records)")


def _extract_scalar_score(
    rubric: Dict[str, Any],
    weights: Dict[str, float] = None,
    max_rubric_score: float = 2.0,
) -> float:
    if weights is None:
        weights = {}
    weighted_sum = 0.0
    total_weight = 0.0

    def traverse(d, current_dim=None):
        nonlocal weighted_sum, total_weight
        if isinstance(d, dict):
            if "score" in d:
                dim_name = d.get("dimension", current_dim)
                if dim_name:
                    v_str = str(d["score"]).strip()
                    if v_str == "N/A":
                        score_val = 0.4
                    elif v_str == "0":
                        score_val = 0.0
                    elif v_str == "0.5":
                        score_val = 0.3
                    elif v_str == "1":
                        score_val = 0.6
                    elif v_str == "1.5":
                        score_val = 0.8
                    elif v_str == "2":
                        score_val = 1.0
                    else:
                        try:
                            score_val = float(v_str) / max_rubric_score
                        except ValueError:
                            score_val = 0.0
                    weight = weights.get(dim_name, 1.0)
                    weighted_sum += score_val * weight
                    total_weight += weight
                return
            for k, v in d.items():
                traverse(v, current_dim=k)

    traverse(rubric)
    return (weighted_sum / total_weight) if total_weight > 0 else 0.0


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0


def select_diverse_parents(
    records: List[GenerationRecord],
    top_k: int,
    similarity_threshold: float = 0.85,
    decay_rate: float = 1.0,
    ensure_top_k: bool = True,
) -> List[GenerationRecord]:
    """Selects top-K diverse records. If ensure_top_k=False, returns only diverse candidates."""
    if not records:
        return []
    if len(records) <= top_k:
        return sorted(records, key=lambda x: x.scalar_score, reverse=True)

    # Apply generation decay if enabled
    if 0.0 < decay_rate < 1.0:
        max_gen = max(r.generation for r in records)
        sorted_records = sorted(
            records,
            key=lambda r: r.scalar_score * (decay_rate ** (max_gen - r.generation)),
            reverse=True,
        )
    else:
        sorted_records = sorted(records, key=lambda x: x.scalar_score, reverse=True)

    selected_indices = [0]
    selected_embs = [sorted_records[0].embedding]

    for i in range(1, len(sorted_records)):
        current_emb = sorted_records[i].embedding
        max_sim = max(_cosine_similarity(current_emb, s_emb) for s_emb in selected_embs)

        if max_sim < similarity_threshold:
            selected_indices.append(i)
            selected_embs.append(current_emb)

        if len(selected_indices) == top_k:
            break

    # Fallback to fill quota only when explicitly requested (GEPA behavior)
    if ensure_top_k and len(selected_indices) < top_k:
        for i in range(len(sorted_records)):
            if i not in selected_indices:
                selected_indices.append(i)
            if len(selected_indices) == top_k:
                break

    return [sorted_records[i] for i in selected_indices]


def format_feedback(
    intent: str, diverse_records: List[GenerationRecord], pop: int
) -> str:
    prompt_blocks = []

    def normalize_score(val):
        if val is None:
            return 0.0
        s = str(val).strip()
        mapping = {"N/A": 0.4, "0": 0.0, "0.5": 0.3, "1": 0.6, "1.5": 0.8, "2": 1.0}
        if s in mapping:
            return mapping[s]
        try:
            return min(float(s) / 2.0, 1.0)
        except ValueError:
            return 0.0

    for i, r in enumerate(diverse_records):
        dim_scores = []

        def scan(d, current_dim=None):
            if not isinstance(d, dict):
                return
            dim_name = d.get("dimension", current_dim)
            sc_raw = d.get("score")
            note = d.get("note", d.get("desc", ""))
            if dim_name and sc_raw is not None:
                sc = normalize_score(sc_raw)
                note_str = f": {note.strip()}" if note and str(note).strip() else ""
                dim_scores.append((sc, dim_name.strip(), note_str))
            for k, v in d.items():
                if k not in ("score", "note", "desc", "dimension"):
                    scan(v, k)

        scan(r.rubric)
        dim_scores.sort(key=lambda x: x[0], reverse=True)

        preserve, fix, improve = [], [], []
        for sc, name, note in dim_scores:
            if sc >= 0.85 and len(preserve) < 2:
                preserve.append(f"  - **{name}** ({sc:.2f}){note}")
            elif sc <= 0.30:
                fix.append(f"  - **{name}** ({sc:.2f}){note}")
            elif 0.55 < sc < 0.65:
                improve.append(f"  - **{name}** ({sc:.2f}){note}")
        block = get_config()["prompts"]["optimizer_mutate_prompt_block"].format(
            fix="\n".join(fix + improve),
            issues_count=len(fix + improve),
            index=i + 1,
            score=round(r.scalar_score, 2),
            prompt=r.prompt,
        )
        prompt_blocks.append(block)
    result = get_config()["prompts"]["optimizer_mutate_user"].format(
        intent=intent, prompt_blocks="\n\n".join(prompt_blocks), pop=pop
    )
    return result
