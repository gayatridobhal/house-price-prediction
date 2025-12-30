FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src
COPY model/ ./model
EXPOSE 80
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]