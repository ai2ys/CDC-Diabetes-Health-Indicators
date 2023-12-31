FROM python:3.10.12-slim-bookworm

WORKDIR /app

RUN pip install pipenv

COPY Pipfile* ./
RUN pipenv install --system --deploy

COPY predict.py ./
COPY models/* ./models/

EXPOSE 9696
# ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
