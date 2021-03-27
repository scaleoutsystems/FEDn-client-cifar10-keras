FROM morganekmefjord/agx:latest
#RUN apt-get update && \
#    apt-get upgrade -y && \
#    apt-get install -y git
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN apt-get install -y git
RUN pip install -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn

COPY fedn-network.yaml /app/
COPY requirements.txt /app/
#COPY hello.py /app/
#COPY client /app/
WORKDIR /app
RUN pip install -r requirements.txt

