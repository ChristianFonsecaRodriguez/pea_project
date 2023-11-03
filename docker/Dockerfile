FROM python:3.8-slim
WORKDIR /app
ADD . /app
RUN apt-get update
RUN apt-get install libgomp1
RUN pip install -r requirements.txt

ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_DEFAULT_REGION=""

EXPOSE 8000
CMD ["python","app.py"]


