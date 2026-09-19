FROM public.ecr.aws/lambda/python:3.14
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin

COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r <(uv export --format requirements.txt)

COPY lambda_function-onnx.py clothing_classifier_mobilenet_v2_latest.onnx ./

CMD ["lambda_function-onnx.lambda_handler"]