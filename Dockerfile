FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "./"]

EXPOSE 6969

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:6969", "predict:app" ]