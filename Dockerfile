FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
	    git \
	    curl \
        libglib2.0-0 \
        software-properties-common \
        python3.8-dev \
        python3-pip

WORKDIR /tmp

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install setuptools
RUN python3.8 -m pip install Cython
RUN python3.8 -m pip install matplotlib numpy pandas scipy tqdm pyyaml easydict scikit-image bridson Pillow ninja
RUN python3.8 -m pip install pycocotools
RUN python3.8 -m pip install imgaug mxboard graphviz
RUN python3.8 -m pip install albumentations
RUN python3.8 -m pip install opencv-python-headless
RUN python3.8 -m pip install torch
RUN python3.8 -m pip install torchvision
RUN python3.8 -m pip install scikit-learn
RUN python3.8 -m pip install tensorboard
RUN python3.8 -m pip install easydict einops timm IPython attr

RUN mkdir /work
WORKDIR /work
RUN chmod -R 777 /work && chmod -R 777 /root

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
