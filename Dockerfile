FROM python:3.8.5
RUN pip install -e git://github.com/scaleoutsystems/fedn.git@master#egg=fedn\&subdirectory=fedn

COPY fedn-network.yaml /app/
COPY requirements.txt /app/
COPY client_settings.yaml /app/
#COPY hello.py /app/
#COPY client /app/
WORKDIR /app
RUN pip install -r requirements.txt
