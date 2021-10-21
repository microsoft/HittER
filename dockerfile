FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

USER root

RUN mkdir -p /kge

COPY requirements.txt /kge

WORKDIR /kge

RUN pip install -r requirements.txt

RUN chmod 777 /kge