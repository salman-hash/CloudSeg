# CloudSeg 🚀

CloudSeg is a **cloud-based image segmentation backend** built with **FastAPI, PyTorch, and PostgreSQL**, deployed on an **Azure Virtual Machine**. It exposes a REST API that accepts images, performs **semantic segmentation** using a deep learning model, stores results and metadata in a database, and returns visual outputs.

This project is designed as a **resume-grade, production-style system**, focusing on backend architecture, cloud deployment, and ML inference — not just model training.

---

## ✨ Features

* 🧠 **Deep Learning Inference** using PyTorch (DeepLabV3)
* ☁️ **Cloud Deployment** on Azure VM (Ubuntu)
* ⚡ **FastAPI** for high-performance REST APIs
* 🗄️ **PostgreSQL** for job tracking & metadata
* 🔐 Secure DB access via **SSH tunneling** (pgAdmin-friendly)
* 🧩 Modular, scalable project structure
* 🖼️ Returns **segmentation mask**, **colored mask**, and **overlay image**

---

## 🏗️ Architecture Overview

```
Client (curl / Postman / UI)
        │
        ▼
FastAPI (CloudSeg API)
        │
        ├── PyTorch Segmentation Model (CPU)
        │
        ├── File Storage (Images / Masks / Overlays)
        │
        └── PostgreSQL Database
              └── segmentation_jobs table
```

---

## 📁 Project Structure

```
CloudSeg/
│
├── src/
│   ├── main.py              # App entry point
│   ├── config.json          # App configuration
│   │
│   ├── api/
│   │   └── routes.py        # API endpoints
│   │
│   ├── model/
│   │   └── model.py         # Segmentation model wrapper
│   │
│   ├── utils/
│   │   ├── image_utils.py   # Image processing helpers
│   │   └── timer.py         # Inference timing
│   │
│   └── db/
│       └── database.py      # SQLAlchemy DB access layer
│
├── data/
│   ├── input_images/
│   ├── output_masks/
│   └── overlay_masks/
│
├── scripts/
│   └── test_api.py          # API test client
│
├── init_db.sql              # PostgreSQL schema
├── requirements.txt
└── README.md
```

---

## 🧠 Model Details

* **Model**: DeepLabV3 with ResNet-50 backbone
* **Framework**: PyTorch + TorchVision
* **Inference**: CPU-only (no GPU required)
* **Output**:

  * Raw segmentation mask
  * Colored class mask
  * Overlay on original image

> ⚠️ This project focuses on **inference & deployment**, not training.

---

## 🗄️ Database Schema

```sql
CREATE TABLE segmentation_jobs (
    id SERIAL PRIMARY KEY,
    image_url TEXT NOT NULL,
    mask_url TEXT,
    overlay_url TEXT,
    model_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    inference_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

Each API request creates a job record that is updated after inference completes.

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/CloudSeg.git
cd CloudSeg
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup PostgreSQL

```bash
sudo apt install postgresql
sudo -i -u postgres
psql
CREATE DATABASE cloudseg;
CREATE USER cloudseg_user WITH PASSWORD 'Strong.1234';
GRANT ALL PRIVILEGES ON DATABASE cloudseg TO cloudseg_user;
```

Apply schema:

```bash
psql -h localhost -U cloudseg_user -d cloudseg -f init_db.sql
```

---

## ▶️ Run the API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 5000
```

Access docs:

```
http://<VM_PUBLIC_IP>:5000/docs
```

---

## 🧪 Test API

```bash
python scripts/test_api.py
```

Response includes:

* Job ID
* Mask image path
* Overlay image path
* Inference time

---

## 🔐 Database Access (pgAdmin)

PostgreSQL is **not publicly exposed**.
Use **SSH tunneling** from your local machine:

```bash
ssh -L 5433:localhost:5432 azureuser@<VM_IP>
```

pgAdmin connection:

```
Host: localhost
Port: 5433
DB: cloudseg
User: cloudseg_user
```

---

## 📌 Resume Highlights

* Designed a **cloud-native ML inference service**
* Implemented **REST APIs + DB-backed job tracking**
* Deployed on **Azure VM (Linux)**
* Used **SQLAlchemy, PostgreSQL, FastAPI, PyTorch**
* Secure infrastructure (no public DB exposure)


---

## 👤 Author

**Salman Sadiq**
Embedded Systems • Cloud • AI Inference

---

⭐ If you like this project, star the repo — it helps a lot!
