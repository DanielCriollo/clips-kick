FROM python:3.12-slim
RUN apt-get update && apt-get install -y ffmpeg && \
    pip install flask yt-dlp requests && apt-get clean
WORKDIR /app
COPY server.py engine.py ./
CMD ["python", "server.py"]
