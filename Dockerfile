# docker for cpu
FROM paddlepaddle/paddle:2.6.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV TRANSFORMERS_CACHE=/cache/hf \
    HF_HOME=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh", "web"]
