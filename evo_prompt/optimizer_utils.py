import datetime
import logging
from pathlib import Path
import shutil

from .config_loader import BASE_DIR
from .model import save_trajectory

logger = logging.getLogger(__name__)


def finalize_optimization(trajectory, output_dir="output", image_count=1):
    uniq_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trajectory_path = BASE_DIR / "data" / f"trajectory_{uniq_suffix}.json"

    save_trajectory(trajectory, str(trajectory_path), fmt="json")

    output_path = BASE_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    top_records = trajectory[:image_count]
    if not top_records:
        logger.warning("No valid generations were produced.")
        return trajectory

    for i, record in enumerate(top_records):
        dest_filename = Path(record.image_path).name
        shutil.copy(record.image_path, output_path / dest_filename)
        logger.info(f"🏆 TOP {i + 1} PROMPT:\n{record.prompt}")
        logger.info(f"📊 FINAL SCORE: {record.scalar_score:.3f}")
        logger.info(f"🖼️ IMAGE PATH: {record.image_path}")
    logger.info(f"📜 TOTAL TRAJECTORY LOGGED: {len(trajectory)} records")
