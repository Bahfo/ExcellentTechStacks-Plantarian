# 🌿 Plantarian: Edge Inference Engine

A high-performance, real-time plant disease classification system. Plantarian uses a deep learning model (MobileNetV2) trained on extensive botanical datasets to identify plant species and diagnose diseases directly from your webcam.

## 🚀 Quick Start

To launch the Plantarian GUI application on your local machine:

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start the Application**
   ```bash
   npm start
   ```

The application will automatically launch your default web browser and connect to the edge inference engine at `http://localhost:8000`.

## 🛠 Features

- **Real-Time Analysis**: Instant plant diagnosis using TensorFlow.js.
- **Modern GUI**: A sleek, dark-themed dashboard built with pure CSS.
- **Edge Inference**: All processing happens locally for maximum speed and privacy.
- **High Accuracy**: Leverages transfer learning from MobileNetV2 for reliable classification across 27+ plant conditions.

## 📁 Project Structure

- `server.js`: The backend engine serving the GUI and handling inference requests.
- `index.html`: The main GUI dashboard (Single Page Application).
- `web_model/`: Optimized TensorFlow.js model shards.
- `train.py`: The high-performance Python training pipeline (for developers).

---
*Developed for the ExcellentTechStacks-Plantarian Protocol.*
