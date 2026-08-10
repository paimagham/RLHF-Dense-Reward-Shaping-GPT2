# Sealed environment for the dense-reward-shaping PPO pipeline.
# Build:  docker build -t dense-reward .
# Run:    docker run --gpus all -it -p 8888:8888 dense-reward
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3-pip git && rm -rf /var/lib/apt/lists/*
 
RUN pip3 install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir \
    trl==0.8.6 transformers==4.40.0 accelerate==0.29.3 huggingface-hub==0.22.2 \
    tokenizers==0.19.1 safetensors==0.4.3 datasets==2.19.0 "numpy<2" \
    pyarrow_hotfix scipy wandb sentencepiece jupyter matplotlib
 
WORKDIR /workspace
COPY . /workspace
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
 
