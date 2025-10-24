FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile","Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C1.0.bin", "./"]

EXPOSE 5050

ENTRYPOINT ["gunicorn", "--bind 0.0.0.0:5000", "predict:app"]
