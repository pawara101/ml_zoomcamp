FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY app.py pipeline_v1.bin ./

EXPOSE 8000

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]