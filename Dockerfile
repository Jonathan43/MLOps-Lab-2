FROM python:3.12-slim

WORKDIR /app

COPY ./app/requirements.txt /app/requirements.txt
COPY ./app/main.py /app/main.py
COPY ./app/iris_model.pkl /app/iris_model.pkl

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
