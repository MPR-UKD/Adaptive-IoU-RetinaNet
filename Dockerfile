#FROM lurad101/python38

#COPY . .

#RUN pip install --no-cache-dir -r requirements.txt

FROM lurad101/adaptive-iou-retinanet

COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y