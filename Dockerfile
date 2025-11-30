FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY digit_cnn_model.pth .

COPY app.py .

COPY PyTorchExecuteAI.py .

CMD ["python", "app.py"]
