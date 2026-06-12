# Plantarian - Plant Disease Detection System

![Python Version](https://img.shields.io/badge/python-3.13%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Node.js Version](https://img.shields.io/badge/node.js-20%2B-green?style=for-the-badge&logo=nodedotjs&logoColor=white)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![MongoDB](https://img.shields.io/badge/mongodb-%2347A248.svg?style=for-the-badge&logo=mongodb&logoColor=white)

---

## Key Features

### Automated Machine Learning Pipeline
* **Data Orchestration:** Seamless extraction of raw image data from MongoDB GridFS for preprocessing.
* **Advanced Preprocessing:** Implementation of Contrast Limited Adaptive Histogram Equalization (CLAHE) and Gaussian blurring to enhance feature extraction.
* **Efficient Training:** Multi-task pipeline supporting mixed-precision training and GPU acceleration.
* **Model Exportation:** Automated conversion of Keras models to TensorFlow.js format for high-performance web inference.

### Intelligent Web Interface
* **Real-time Inference:** Client-side and server-side support for plant disease classification across 38 distinct categories.
* **Secure Authentication:** Robust user session management and password hashing using bcrypt.
* **Responsive Dashboard:** Visual feedback for plant health status with confidence scoring.

---

## Technology Stack

**Backend:** Node.js / Express.

**Machine Learning:** TensorFlow / Keras / OpenCV.

**Database:** MongoDB (GridFS for large binary objects).

**Frontend UI:** Vanilla JS / HTML5 / CSS3.

---

## Installation and Setup

Follow these steps to initialize and run the Plantarian environment:

### 1. Clone the Repository
```bash
git clone https://github.com/Bahfo/ExcellentTechStacks-Plantarian
cd ExcellentTechStacks-Plantarian
```

### 2. Configure Environment Dependencies
Initialize both Node.js and Python environments:

#### Node.js Dependencies
```bash
npm install
```

#### Python Dependencies
Create a virtual environment and install the required machine learning modules:
```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow opencv-python numpy pymongo
```

> [!NOTE]
> On Windows systems, use `venv\Scripts\activate` to activate the virtual environment.

### 3. Initialize Database
Ensure a MongoDB instance is running locally or provide a connection string via the `MONGO_URI` environment variable.

### 4. Execute the ML Pipeline
To process data, train the model, and export it for web use, execute the orchestrator script:
```bash
python3 run.py
```

### 5. Launch the Web Server
Start the Express server to begin serving the application:
```bash
npm start
```

---

## Developers
This project was developed and is maintained by:
* **m5bvo**
* **bahfo**
