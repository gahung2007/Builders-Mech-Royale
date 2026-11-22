FROM python:3.13.18-slim-bookworm
WORKDIR /bot
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY mechfight.py bot.py ./
USER 65534:65534
ENTRYPOINT ["python3", "-B", "-u", "/bot/bot.py"]
