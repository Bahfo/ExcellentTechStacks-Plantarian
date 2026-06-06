FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

# Install OpenCV system dependencies and clear cache to keep image clean
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Force compatibility by pinning NumPy to 1.x along with other requirements
RUN pip install --no-cache-dir "numpy<2.0.0" pymongo opencv-python-headless
RUN pip install --default-timeout=10000 --upgrade protobuf tensorflowjs
COPY . /app

ENV PYTHONPATH="/app"