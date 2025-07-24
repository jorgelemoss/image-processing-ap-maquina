FROM python:3.11-alpine

WORKDIR /server

RUN apk add --no-cache build-base

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 4000

CMD ["uvicorn", "modeling.main:app", "--host", "0.0.0.0", "--port", "4000"]
