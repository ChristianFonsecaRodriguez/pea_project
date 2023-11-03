FROM python:3.8-slim
WORKDIR /app
ADD . /app
RUN apt-get update
RUN apt-get install libgomp1
RUN pip install -r requirements.txt

ENV AWS_ACCESS_KEY_ID="AKIAT53PUUHOQAF2FH7M"
ENV AWS_SECRET_ACCESS_KEY="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"
ENV AWS_DEFAULT_REGION="us-east-1"

EXPOSE 8000
CMD ["python","app.py"]


