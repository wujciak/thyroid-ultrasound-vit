# Build context is the repo root (see app/docker-compose.yml) so the thyroid package can be installed.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY app/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY app/backend/main.py ./

ENV MODEL_PATH=/models/model.pt
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
