FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

RUN mkdir /fl-seminar-code
COPY ./ /fl-seminar-code
WORKDIR /fl-seminar-code


## Install your dependencies here using apt install, etc.

RUN apt update && apt upgrade -y && apt clean
RUN apt install ffmpeg libsm6 libxext6 tar unzip wget -y

# RUN apt update && apt install -y --no-install-recommends \
#         build-essential \
#         curl \
#         software-properties-common \
#         unzip

# RUN apt install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN pip install --upgrade pip

# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
## Include the following line if you have a requirements.txt file.
RUN pip install -r code/requirements.txt
# RUN pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --use-feature=2020-resolver
RUN python -m pip cache purge


# RUN python code/docker_test.py  # TODO: implement the test code
