FROM nvidia/cuda:12.4.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev git git-lfs \
    libx11-6 libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/comfyui
RUN git clone https://github.com/comfyanonymous/ComfyUI.git .

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV TRITON_CACHE_DIR=/app/comfyui/.triton

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p user input output temp models custom_nodes .triton

RUN git clone https://github.com/nova452/Rebalance-Pack.git custom_nodes/Rebalance-Pack

EXPOSE 8188
CMD ["python3", "main.py", "--listen", "--port", "8188", "--disable-auto-launch"]
