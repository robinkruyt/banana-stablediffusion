# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git curl wget && apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup conda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#      /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH


# Install stable diffusion
RUN git clone https://github.com/CompVis/stable-diffusion.git stablediffusion

# Install checkpoint
RUN curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > /app/stablediffusion/sd-v1-4.ckpt

# Create env for stable diffusion
RUN cd /app/stablediffusion && conda env create -f environment.yaml

# Set a new shell to include conda env
SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]


# App specific code
# Install python packages
RUN conda install sanic transformers accelerate

# We add the banana boilerplate here
ADD server.py .

# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
