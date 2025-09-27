FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY utils.py /app/
COPY main.py /app/
COPY start.sh /app/

CMD ["./start.sh"]