import argparse
import logging
import sys
from pathlib import Path

from .config_loader import load_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evo-Prompt Optimizer")

    # Input sources (mutually exclusive for clarity)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("-p", "--prompt", help="Single intent to optimize")
    input_group.add_argument(
        "-f",
        "--file",
        default="prompts.md",
        help="Path to file with intents (one per line)",
    )

    # Evolution parameters
    parser.add_argument(
        "-w", "--pop-size", default=4, help="Population size per generation", type=int
    )
    parser.add_argument(
        "-g",
        "--generations",
        default=4,
        help="Number of optimization generations",
        type=int,
    )
    parser.add_argument(
        "-k",
        "--top-k",
        default=4,
        help="Top-K parents to keep for crossover/mutation",
        type=int,
    )
    parser.add_argument(
        "-n",
        "--top-images",
        default=1,
        help="Number of best images to save to output dir",
        type=int,
    )
    parser.add_argument(
        "-a",
        "--optimizer",
        default="gepa",
        help="Optimization algorithm",
        choices=["gepa", "beam"],
    )

    # Output & Config
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="Directory to save final images and logs",
    )

    config_group = parser.add_argument_group("Configuration Files")
    config_group.add_argument(
        "--infra-config",
        default="config/infra.yaml",
        help="Path to infrastructure configuration (models, containers, APIs, rubrics)",
    )
    config_group.add_argument(
        "--prompts-config",
        default="config/prompts.yaml",
        help="Path to prompt templates configuration",
    )

    parser.add_argument(
        "-l",
        "--log-level",
        default="info",
        help="Logging verbosity",
        choices=["debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument(
        "-y", "--workflow", default="krea-2-basic.json", help="comfy workflow to run"
    )
    parser.add_argument(
        "-r", "--lora_name", default="", help="Lora for krea-2-advanced.json"
    )

    args = parser.parse_args()

    # Validate input source
    if not args.prompt and not Path(args.file).exists():
        parser.error(f"Either provide --prompt or ensure --file '{args.file}' exists.")

    return args


def init_logging(level: str) -> None:
    # Safe mapping using built-in logging registry (Python 3.11+)
    level_name = level.upper()
    log_level = logging.getLevelNamesMapping().get(level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Silence noisy third-party HTTP libraries
    for lib in ["httpx", "httpcore", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def main() -> None:
    args = parse_arguments()
    init_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load split configuration (infra first, then prompts)
    load_config(args.infra_config, args.prompts_config)

    logger.debug(f"file: {args.file}")
    intents = [args.prompt] if args.prompt else []
    if not intents:
        with open(args.file, "r", encoding="utf-8") as f:
            intents = [line.strip() for line in f if line.strip()]
    logger.debug(f"intents:\n{intents}")

    # Dispatch optimizer
    logger.info(f"🧠 Using optimizer: {args.optimizer}")
    if args.optimizer == "gepa":
        from .simple_gepa import run_optimizer as optimizer_fn
    else:
        from .beam_optimizer import optimize as optimizer_fn

    # Run optimization loop
    for intent in intents:
        logger.info(f"🚀 Starting optimization for: '{intent}'")
        optimizer_fn(
            intent,
            width=args.pop_size,
            deep=args.generations,
            keep=args.top_k,
            output_dir=args.output_dir,
            image_count=args.top_images,
            workflow=args.workflow,
            lora_name=args.lora_name,
        )


if __name__ == "__main__":
    main()
