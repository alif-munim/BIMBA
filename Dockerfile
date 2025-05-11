ARG REGION=us-west-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---- Python deps that need nvcc / compile time ----
COPY BIMBA-LLaVA-NeXT/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    pip install flash-attn==2.4.* sagemaker boto3

# Install the repo itself (editable)
COPY BIMBA-LLaVA-NeXT /workspace/BIMBA-LLaVA-NeXT
WORKDIR /workspace/BIMBA-LLaVA-NeXT
RUN pip install -e .

############################
# -----  STAGE 2  (runtime)  -----
############################
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# copy the entire conda env from the build stage
COPY --from=build /opt/conda /opt/conda
# copy repo (so import paths still resolve even before SageMaker mounts /opt/ml/code)
COPY --from=build /workspace/BIMBA-LLaVA-NeXT /workspace/BIMBA-LLaVA-NeXT

# default workdir SageMaker expects to overlay source_dir into
WORKDIR /opt/ml/code

# nothing else needed—SageMaker training toolkit in the base image
# will invoke “python train_entrypoint.py” when you pass entry_point
