FROM nikolaik/python-nodejs:python3.9-nodejs20

WORKDIR /FASTAPI

COPY ./main.py .
COPY ./requirements.txt .

RUN pip install -r ./requirements.txt

EXPOSE 8081

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8081"]