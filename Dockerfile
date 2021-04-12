FROM ubuntu:20.04
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN pip3 -q install pip --upgrade

COPY notebooks/notebook_requirements.txt ./

RUN pip3 install --no-cache notebook && \
    pip3 install numpy && \
    pip3 install -r notebook_requirements.txt


ARG NB_USER=user
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}


USER $NB_USER

COPY --chown=1000:100 . Pandora

WORKDIR /Pandora