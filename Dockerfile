FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY app ./app

EXPOSE 18080
CMD ["python", "-m", "app.gateway_main"]
