FROM python:3.10.12-slim-bookworm

WORKDIR /app

RUN pip install pipenv

COPY Pipfile* ./
RUN pipenv install --system --deploy

COPY predict.py data_normalizers.json ./
COPY models/model_random_forest.bin ./models/

EXPOSE 9696
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]