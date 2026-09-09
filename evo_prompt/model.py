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
    generated_by: str = ""


@dataclass
class PromptCandidate:
    prompt: str
    reasoning: str = ""
    generated_by: str = ""


def parse_prompts(raw: str, target_count: int) -> List[str]:
    return [p.strip() for p in raw.split("\n---\n") if len(p.strip()) > 15][:target_count]


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


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0


def _normalize_values(values: List[float]) -> List[float]:
    if not values:
        return []

    lo = min(values)
    hi = max(values)

    if hi - lo < 1e-12:
        return [1.0] * len(values)

    return [(v - lo) / (hi - lo) for v in values]


def select_diverse_parents_mmr(
    records: List[GenerationRecord],
    top_k: int,
    quality_weight: float = 0.55,
    pool_factor: float = 5.0,
    decay_rate: float = 1.0,
) -> List[GenerationRecord]:
    """
    Select top-K records using an MMR-style quality/diversity tradeoff.

    Args:
        records: candidate records.
        top_k: number of records to select.
        quality_weight: how much to weight scalar score vs diversity.
            - 1.0 = pure scalar score
            - 0.0 = pure diversity
            - 0.5-0.7 is often reasonable
        pool_factor: how many top candidates to consider.
            e.g. pool_factor=5 means consider top 5*top_k records.
        decay_rate: optional generation decay.
    """

    if not records or top_k <= 0:
        return []

    if len(records) <= top_k:
        return sorted(records, key=lambda x: x.scalar_score, reverse=True)

    # 1) Optional generation decay
    if 0.0 < decay_rate < 1.0:
        max_gen = max(r.generation for r in records)
        effective_scores = {
            id(r): r.scalar_score * (decay_rate ** (max_gen - r.generation))
            for r in records
        }
    else:
        effective_scores = {id(r): r.scalar_score for r in records}

    # 2) Sort all records by effective score
    ordered = sorted(
        records,
        key=lambda r: effective_scores[id(r)],
        reverse=True,
    )

    # 3) Use only a candidate pool of relatively good records
    pool_size = min(len(ordered), max(top_k, int(pool_factor * top_k)))
    pool = ordered[:pool_size]

    # If pool is smaller than top_k, use all records.
    if len(pool) < top_k:
        pool = ordered[:]

    selected: List[GenerationRecord] = []
    selected_embs: List[List[float]] = []
    selected_ids: set[int] = set()

    # 4) Seed with the best record
    first = pool[0]
    selected.append(first)
    selected_embs.append(first.embedding)
    selected_ids.add(id(first))

    pool = pool[1:]
    norm_scores = _normalize_values([effective_scores[id(r)] for r in pool])

    # 5) Greedily add the best quality/diversity tradeoff
    max_sims: List[float] = []
    while len(selected) < top_k and pool:
        max_sims = []

        for rec in pool:
            max_sim = max(
                _cosine_similarity(rec.embedding, emb)
                for emb in selected_embs
            )
            max_sims.append(max_sim)

        # Rescale similarity so the least similar candidate gets diversity_score = 1.0
        # and the most similar candidate gets diversity_score = 0.0.
        lo_sim = min(max_sims)
        hi_sim = max(max_sims)

        if hi_sim - lo_sim > 1e-12:
            diversity_scores = [
                1.0 - (s - lo_sim) / (hi_sim - lo_sim)
                for s in max_sims
            ]
        else:
            # All remaining candidates are equally similar to selected records.
            diversity_scores = [1.0] * len(pool)

        best_idx = -1
        best_key = None

        for idx, rec in enumerate(pool):
            score = (
                quality_weight * norm_scores[idx]
                + (1.0 - quality_weight) * diversity_scores[idx]
            )

            # Tie-break by normalized score.
            key = (score, norm_scores[idx])

            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx == -1:
            break

        rec = pool.pop(best_idx)
        norm_scores.pop(best_idx)

        selected.append(rec)
        selected_embs.append(rec.embedding)
        selected_ids.add(id(rec))

    # Fallback in case of weird edge cases, e.g. empty embeddings.
    if len(selected) < top_k:
        for rec in ordered:
            if id(rec) not in selected_ids:
                selected.append(rec)
                selected_ids.add(id(rec))

            if len(selected) == top_k:
                break
    return selected


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
        mapping = {"N/A": 0.3, "0": 0.0, "0.5": 0.25, "1": 0.5, "1.5": 0.75, "2": 1.0}
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
        dim_scores.sort(key=lambda x: x[0])

        fix = []
        for sc, name, note in dim_scores[:3]:
            if sc < 1.0:
                fix.append(f"  - **{name}** ({sc:.2f}){note}")
        block = get_config()["prompts"]["optimizer_mutate_prompt_block"].format(
            fix="\n".join(fix),
            issues_count=len(fix),
            index=i + 1,
            score=round(r.scalar_score, 2),
            prompt=r.prompt,
        )
        prompt_blocks.append(block)
    result = get_config()["prompts"]["optimizer_mutate_user"].format(
        intent=intent, prompt_blocks="\n\n".join(prompt_blocks), pop=pop
    )
    return result


def _safe_format_template(template: str, **fields: str) -> str:
    """
    Format a template with placeholders such as {intent}, {prompt}, and {feedback}.

    Falls back to plain replacement if .format() fails, which can happen when
    prompt/feedback text contains curly braces.
    """
    try:
        return template.format(**fields)
    except Exception:
        out = template
        for key, value in fields.items():
            out = out.replace("{" + key + "}", str(value))
        return out


def _normalize_single_feedback_score(value: Any) -> float:
    """
    Normalize a rubric score into [0, 1] using the same convention as format_feedback.
    """
    if value is None:
        return 0.0

    s = str(value).strip()
    mapping = {
        "N/A": 0.3,
        "0": 0.0,
        "0.5": 0.25,
        "1": 0.5,
        "1.5": 0.75,
        "2": 1.0,
    }

    if s in mapping:
        return mapping[s]

    try:
        return min(float(s) / 2.0, 1.0)
    except ValueError:
        return 0.0


def _extract_single_record_feedback(record: GenerationRecord) -> str:
    """
    Build a feedback string for a single GenerationRecord.

    Uses:
    - record.scoring_reasoning if present
    - record.scalar_score
    - weak dimensions from record.rubric
    """
    parts: List[str] = []

    # 3) Extract weak rubric dimensions.
    dim_scores: List[Any] = []

    if isinstance(record.rubric, dict):
        def scan(d: Any, current_dim: Any = None) -> None:
            if not isinstance(d, dict):
                return

            dim_name = d.get("dimension", current_dim)
            sc_raw = d.get("score")
            note = d.get("note", d.get("desc", ""))

            if dim_name and sc_raw is not None:
                sc = _normalize_single_feedback_score(sc_raw)
                note_str = (
                    f": {str(note).strip()}"
                    if note and str(note).strip()
                    else ""
                )
                dim_scores.append((sc, str(dim_name).strip(), note_str))

            for k, v in d.items():
                if k not in ("score", "note", "desc", "dimension"):
                    scan(v, k)

        scan(record.rubric)

    dim_scores.sort(key=lambda x: x[0])
    low_scores = [item for item in dim_scores if item[0] < 1.0]

    if low_scores:
        rubric_lines = []
        for sc, name, note in low_scores[:3]:
            rubric_lines.append(f"  - {name} ({sc:.2f}){note}")
        parts.append("\n".join(rubric_lines))

    if not parts:
        parts.append("No detailed feedback available.")
    result = "\n".join(part for part in parts if part)
    logger.debug("_extract_single_record_feedback:result:\n" + result)
    return result


def format_single_feedback(record: GenerationRecord) -> str:
    """
    Format feedback for a single GenerationRecord using the `refine_user` template.

    Required config template keys:
    - {intent}
    - {prompt}
    - {feedback}
    """
    template = get_config()["prompts"]["optimizer_fix_user"]

    return _safe_format_template(
        template,
        intent=str(record.intent),
        prompt=str(record.prompt),
        feedback=_extract_single_record_feedback(record),
    )
