FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src
EXPOSE 7860
ENV CHROMA_PERSIST_DIR=/app/chroma_db
CMD ["python", "src/app_ollama.py"]
