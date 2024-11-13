FROM ubuntu:24.10

RUN apt update -y && apt upgrade -y \
    && apt install curl wget vim git build-essential -y \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh \
    && mkdir ~/repos && cd ~/repos \
    && git clone git@github.com:neuroctr/wav-tf.git \
    && cd ~/repos/wav-tf