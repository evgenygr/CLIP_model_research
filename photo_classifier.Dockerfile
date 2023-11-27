FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ml_tools.py CLIP_testbench.py ./
COPY templates templates/
COPY master_vectors master_vectors/

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

CMD ["python", "CLIP_testbench.py"]