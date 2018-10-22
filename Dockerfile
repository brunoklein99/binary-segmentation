FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

RUN apt-get update && \
	apt-get install software-properties-common -y && \
	add-apt-repository ppa:jonathonf/python-3.6 && \
	apt-get update && \
	apt-get install python3.6 -y && \
	apt-get install python3-pip -y

COPY . .
RUN python3.6 -m pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN python3.6 -m pip install -r requirements.txt

ENTRYPOINT ["python3.6", "main.py"]