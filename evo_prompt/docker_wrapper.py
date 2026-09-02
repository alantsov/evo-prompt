import logging
import time
import threading
import docker
from pathlib import Path
import requests
import os

from .config_loader import get_config, BASE_DIR

logger = logging.getLogger(__name__)
_docker_client = None
_comfyui_manager = None
_llama_manager = None
_embed_manager = None
_embed_cpu_manager = None  # New


def get_docker_client():
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def stream_logs_simple(container, container_logger):
    for log_line in container.logs(stream=True):
        line = log_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line:
            container_logger.debug(line)


def is_gpu_available():
    client = get_docker_client()
    try:
        info = client.info()
        runtimes = info.get("Runtimes", {})
        if "nvidia" in runtimes:
            return True
        return "nvidia" in str(info).lower()
    except Exception as e:
        logger.debug(f"Failed to check GPU availability: {e}")
        return False


class DockerContainerManager:
    def __init__(
        self,
        image,
        command=None,
        ports=None,
        volumes=None,
        environment=None,
        name=None,
        health_check_url=None,
        health_timeout=120,
        cfg=None,
        use_gpu=True,  # New parameter
    ):
        self.image = image
        self.command = command
        self.ports = ports or {}
        self.volumes = volumes or {}
        self.environment = environment or {}
        self.name = name
        self.health_check_url = health_check_url
        self.health_timeout = health_timeout
        self.container = None
        self.logger = logging.getLogger(name)
        self.config = cfg or {}
        self.use_gpu = use_gpu  # Store flag

    def start(self):
        if self.container and self.container.status == "running":
            logger.debug(f"Container {self.image} already running.")
            return

        logger.info(f"🚀 Starting Docker container: {self.image}...")
        client = get_docker_client()
        try:
            client.images.get(self.image)
        except docker.errors.ImageNotFound:
            if self.config.get("build", ""):
                build_directory = str(
                    BASE_DIR / "containers" / self.config.get("build", "")
                )
                logger.debug(
                    f"building image {self.image} from directory {build_directory}"
                )
                client.images.build(path=build_directory, tag=self.image)
            else:
                client.images.pull(self.image)

        run_kwargs = {
            "image": self.image,
            "detach": True,
            "remove": True,
            "ports": self.ports,
            "environment": self.environment,
            "volumes": self.volumes,
        }

        if self.command:
            run_kwargs["command"] = self.command

        # Modified GPU check
        if self.use_gpu and is_gpu_available():
            run_kwargs["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
            logger.info("🟢 NVIDIA GPU support detected. Enabling GPU passthrough.")
        else:
            if self.use_gpu:
                logger.warning("🔴 NVIDIA GPU runtime not detected. Running container on CPU.")
            else:
                logger.info("🟡 CPU-only mode enforced for this container.")

        try:
            self.container = client.containers.run(**run_kwargs)
            log_thread = threading.Thread(
                target=stream_logs_simple,
                args=(self.container, self.logger),
                daemon=True,
            )
            log_thread.start()
            logger.info(f"✅ Container started: {self.container.id[:12]}")
        except docker.errors.APIError as e:
            logger.error(f"❌ Failed to start container: {e}")
            raise

        if self.health_check_url:
            self._wait_for_ready()

    def _wait_for_ready(self):
        start = time.time()
        while time.time() - start < self.health_timeout:
            try:
                if requests.get(self.health_check_url, timeout=5).status_code == 200:
                    logger.info(f"✅ Container {self.image} is healthy and ready.")
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        logger.warning(
            f"⚠️ Container {self.image} health check timed out. Proceeding anyway."
        )

    def stop(self):
        if self.container:
            logger.info(f"🛑 Stopping container {self.image}...")
            try:
                self.container.stop(timeout=10)
            except Exception as e:
                logger.debug(f"Container stop exception: {e}")
            finally:
                try:
                    self.container.remove(force=True)
                except Exception:
                    pass
            self.container = None
            logger.info(f"✅ Container {self.image} stopped and cleaned up.")


def get_comfyui_manager():
    global _comfyui_manager
    if _comfyui_manager is None:
        cfg = get_config()["containers"]["comfyui"]
        comfy_cfg = get_config().get("comfyui", {})

        def resolve_path(config_val: str) -> str:
            p = Path(config_val)
            if not p.is_absolute():
                p = BASE_DIR / p
            return str(p.resolve())
        volumes = {}
        models_dir = resolve_path(comfy_cfg.get("models_dir", "./ComfyUI/models"))
        volumes[models_dir] = {"bind": "/app/comfyui/models", "mode": "ro"}
        if comfy_cfg.get("custom_nodes_dir", ''):
            custom_nodes_dir = resolve_path(comfy_cfg.get("custom_nodes_dir", ''))
            volumes[custom_nodes_dir] = {"bind": "/app/comfyui/custom_nodes", "mode": "ro"}

        if not Path(models_dir).is_dir():
            logger.warning(f"⚠️ ComfyUI models directory not found: {models_dir}")

        _comfyui_manager = DockerContainerManager(
            image=cfg["image"],
            name=cfg["name"],
            ports={f"{cfg['port']}/tcp": cfg["port"]},
            volumes=volumes,
            health_check_url=cfg["health_url"],
            health_timeout=cfg["health_timeout"],
            cfg=cfg,
            use_gpu=True,
        )
    return _comfyui_manager


def start_comfyui_server():
    get_comfyui_manager().start()


def stop_comfyui_server():
    get_comfyui_manager().stop()


def get_llama_manager():
    global _llama_manager
    if _llama_manager is None:
        cfg = get_config()["containers"]["llama_cpp"]
        _llama_manager = DockerContainerManager(
            image=cfg["image"],
            name=cfg["name"],
            ports={f"{cfg['port']}/tcp": cfg["port"]},
            volumes={
                os.path.expanduser("~/.cache/huggingface"): {
                    "bind": "/home/user/.cache/huggingface",
                    "mode": "rw",
                }
            },
            environment={"HOME": "/home/user"},
            command=[
                "--server", "--host", "0.0.0.0", "--port", str(cfg["port"]),
                "-hf", get_config()["models"]["evaluator"], "--no-cache-prompt",
                "--parallel", "1", "-ngl", "999", "-c", "32000",
                "--reasoning-format", "deepseek", "--cache-type-k", "f16",
                "--cache-type-v", "f16", "--reasoning-budget", "10000",
                #"--image-max-tokens", "560", "--image-min-tokens", "560", "-ub", "1160",
                "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"
            ],
            health_check_url=cfg["health_url"],
            health_timeout=cfg["health_timeout"],
            cfg=cfg,
            use_gpu=True,
        )
    return _llama_manager


def get_embed_manager():
    global _embed_manager
    if _embed_manager is None:
        cfg = get_config()["containers"]["embed"]
        _embed_manager = DockerContainerManager(
            image=cfg["image"],
            name=cfg["name"],
            ports={f"{cfg['port']}/tcp": cfg["port"]},
            volumes={
                os.path.expanduser("~/.cache/huggingface"): {
                    "bind": "/home/user/.cache/huggingface",
                    "mode": "rw",
                }
            },
            environment={"HOME": "/home/user"},
            command=[
                "--server", "--embeddings", "--host", "0.0.0.0", "--port", str(cfg["port"]),
                "-hf", get_config()["models"]["embed"], "-ngl", "999", "-c", "8192",
                "--flash-attn", "on",
            ],
            health_check_url=cfg["health_url"],
            health_timeout=cfg["health_timeout"],
            cfg=cfg,
            use_gpu=True,
        )
    return _embed_manager


# --- New CPU Embed Server ---
def get_embed_cpu_manager():
    global _embed_cpu_manager
    if _embed_cpu_manager is None:
        cfg = get_config()["containers"]["embed"]
        _embed_cpu_manager = DockerContainerManager(
            image=cfg["image"],
            name=f"{cfg['name']}-cpu",
            ports={f"{cfg['port']}/tcp": cfg["port"]},
            volumes={
                os.path.expanduser("~/.cache/huggingface"): {
                    "bind": "/home/user/.cache/huggingface",
                    "mode": "rw",
                }
            },
            environment={"HOME": "/home/user"},
            command=[
                "--server", "--embeddings", "--host", "0.0.0.0", "--port", str(cfg["port"]),
                "-hf", get_config()["models"]["embed"], "-ngl", "0",  # Force CPU
                "-c", "8192", "--flash-attn", "on",
            ],
            health_check_url=f"http://127.0.0.1:{cfg['port']}/health",
            health_timeout=cfg["health_timeout"],
            cfg=cfg,
            use_gpu=False,  # Explicitly disable GPU
        )
    return _embed_cpu_manager


def start_embed_cpu_server():
    get_embed_cpu_manager().start()


def stop_embed_cpu_server():
    get_embed_cpu_manager().stop()
