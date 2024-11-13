FROM ubuntu:24.10

RUN apt update -y && apt upgrade -y \
    && apt install curl wget vim git build-essential -y \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh \
    && rustup toolchain install 1.82.0 && rustup toolchain default 1.82.0 \
    && source ~/.profile \
    && mkdir ~/repos && cd ~/repos \
    && git clone git@github.com:neuroctr/wav-tf.git \
    && cd ~/repos/wav-tf \
    && cd python \
    && python3 -m venv v && source ./v/bin/activate \
    && pip install --upgrade pip \
    && pip install tensorflow-cpu \
    && python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"