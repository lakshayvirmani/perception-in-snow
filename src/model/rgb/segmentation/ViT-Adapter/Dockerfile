FROM python:3.7-slim-buster

ADD ./ ./

RUN apt-get update

RUN pip install instaboostfast # for htc++
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install timm==0.4.12
RUN pip install mmdet==2.22.0 # for Mask2Former
RUN pip install mmsegmentation==0.20.2