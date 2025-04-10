FROM python:3.12-slim

WORKDIR /streamlit

COPY ./streamlit /streamlit

COPY ./streamlit/requirements.txt /streamlit/requirements.txt
RUN pip install --no-cache-dir -r /streamlit/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "front_end.py", "--server.port=8501", "--server.address=0.0.0.0"]
