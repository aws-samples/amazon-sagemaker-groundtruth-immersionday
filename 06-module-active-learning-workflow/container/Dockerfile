# Build an image that can do both training and inference in SageMaker

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.1-cpu-py39-ubuntu20.04-sagemaker
RUN apt-get update && \
    apt-get install -y nginx && \ 
    pip install --upgrade pip
    
  
RUN pip3 install gevent gunicorn flask

ENV PATH="/opt/program:${PATH}"

ENV PYTHONUNBUFFERED=1

# Set up the program in the image
COPY news-classifier /opt/program
WORKDIR /opt/program

