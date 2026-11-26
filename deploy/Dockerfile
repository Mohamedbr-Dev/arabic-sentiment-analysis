FROM python:3.12-slim

# Hugging Face Spaces run the container as user 1000
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user model ./model

# Public server: limit /predict per visitor and overall (requests per minute)
ENV MODEL_DIR=/home/user/app/model \
    RATE_LIMIT_PER_CLIENT=20 \
    RATE_LIMIT_TOTAL=200 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "src.day3_api:app", "--host", "0.0.0.0", "--port", "7860"]
