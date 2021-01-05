ARG CPU_OR_GPU
ARG AWS_REGION

FROM 520713654638.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-tensorflow:ray0.6.5-${CPU_OR_GPU}-py3

WORKDIR /opt/ml

RUN apt-get update
RUN apt-get install sudo
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install libpcre3
RUN apt-get -y install libpcre3-dev

#Installing swig
ARG SWIG_VERSION=4.0.1

#https://sourceforge.net/projects/swig/files/swig/swig-4.0.1/swig-4.0.1.tar.gz/download
RUN wget "https://sourceforge.net/projects/swig/files/swig/swig-${SWIG_VERSION}/swig-${SWIG_VERSION}.tar.gz" \
    && tar xzf "swig-${SWIG_VERSION}.tar.gz"
RUN cd "swig-${SWIG_VERSION}/" \
    && ./configure \
    && make \
    && make install

CMD ["swig","-version"]

# Roboschool dependencies

RUN apt-get update && apt-get install -y \
      git cmake ffmpeg pkg-config \
      qtbase5-dev libqt5opengl5-dev libassimp-dev \
      libtinyxml-dev \
      libgl1-mesa-dev \
    && cd /opt \
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y libboost-python-dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.6-dev \
    && ln -s -f /usr/bin/python3.6 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade \
    pip \
    setuptools

RUN pip install sagemaker-containers --upgrade

RUN yes y | pip uninstall gym; exit 0

RUN pip install box2d-py

RUN pip install gym[box2d]; exit 0

ENV PYTHONUNBUFFERED 1

RUN pip show gym

# verify file is being copied
#COPY ./src/lunar_lander.py /usr/local/lib/python3.6/dist-packages/gym/envs/box2d/lunar_lander2.py
RUN echo $SCENARIO

COPY ./src/lunar_lander.py /usr/local/lib/python3.6/dist-packages/gym/envs/box2d/lunar_lander.py

COPY ./src/reward_function.py /usr/local/lib/python3.6/dist-packages/gym/envs/box2d/reward_function.py

RUN echo -----------------------------------------------

RUN ls /usr/local/lib/python3.6/dist-packages/gym/envs/box2d/




############################################
# Test Installation
############################################
# Test to verify if all required dependencies installed successfully or not.
RUN python -c "import gym;import sagemaker_containers.cli.train; import ray; from sagemaker_containers.cli.train import main; import Box2D;"

# Make things a bit easier to debug
WORKDIR /opt/ml/code
