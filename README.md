Since you are the owner and lead for this 5-member team, your README needs to be the "Source of Truth." It should explain the **MVC architecture**, the **7-model bake-off** strategy, and the **Git workflow** so your team doesn't overwrite each other's code.

Here is a detailed, professional `README.md` structure tailored specifically for your Shoplifting Detection project.

---

# 🛡️ Real-Time Shoplifting Detection System

### AI & Data Science Engineering Project | MVC Architecture

## 📖 Project Overview

This project aims to detect shoplifting behaviors in real-time using a hybrid deep learning approach. We are utilizing a **Model-View-Controller (MVC)** architecture to bridge the gap between heavy AI research and a functional, real-time security dashboard.

---

## 🏗️ System Architecture (MVC)

To ensure the project is scalable, we separate concerns into three layers:

### 1. The Model (🧠 Intelligence)

* **Location:** `/app/models/`
* **Role:** Loads pre-trained weights (VideoMAE, MoViNet, etc.) and performs inference.
* **Input:** A 5D tensor of video frames.
* **Output:** A probability score (0.0 to 1.0) for "Shoplifting" vs "Normal."

### 2. The View (🖥️ Interface)

* **Location:** `/app/views/`
* **Role:** Built with **Streamlit**. It displays the live camera feed, visualizes probability charts, and displays incident alerts.

### 3. The Controller (🚦 Logic)

* **Location:** `/app/controllers/`
* **Role:** The "Bridge." It uses **FastAPI** to route frames from the camera to the model. It manages the **Sliding Window buffer** (last 24 frames) and triggers database logging if theft is detected.

---

## 👥 Team Roles & Task Distribution

We are using a **Model Bake-off** strategy where each member tests a different SOTA approach to find the best balance of speed and accuracy.

| Member | Assigned Approach | Primary Directory |
| --- | --- | --- |
| **Tamer (Owner)** | **MoViNet** (Edge Optimized) | `/app/controllers`, `core/` |
| **George** | **VideoMAE**  | `ml_pipeline/notebooks` |
| **Ebrahim** | **YOLO + LRCN** (Hybrid) | `ml_pipeline/src` |
| **Seif** | **TimeSformer**** (X3D/I3D) | `/app/models` |
| **Member 5** | **SlowFast Networks** | `/app/views`, `Dockerfile` |

---

## 🛠️ The Implementation Process

### Phase 1: Offline Research (`ml_pipeline/`)

* Clean data (remove 218 duplicates).
* Handle class imbalance using **Focal Loss** or **Class Weights**.
* Train and export best weights to `ml_pipeline/weights/`.

### Phase 2: Online Integration (`app/`)

* Convert notebooks into clean Python scripts in `app/models/`.
* Set up **FastAPI** endpoints in `app/controllers/`.
* Connect **PostgreSQL** for incident logging.

### Phase 3: Deployment

* Containerize the app using **Docker**.
* Deploy the dashboard via **Streamlit**.

---

## 💻 How to Clone and Work on This Repo

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/tamer-elkoT/Shoplifting_Detection.git
cd Shoplifting_Detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. The Git Workflow (Crucial!)

To avoid merge conflicts, **never work directly on `main**`.

1. Create a branch for your task: `git checkout -b feature-lrcn-model`
2. Work and commit: `git commit -m "Added Focal Loss to training loop"`
3. Push your branch: `git push origin feature-lrcn-model`
4. Open a **Pull Request** on GitHub for Tamer to review.

---

## 📂 Repository Structure

```text
├── ml_pipeline/           # Training, Notebooks, and Weights
├── app/                   # Live MVC Application
│   ├── models/            # The Model Layer
│   ├── views/             # The View Layer
│   ├── controllers/       # The Controller Layer
│   └── core/              # DB & Config
├── assets/                # Documentation Diagrams
└── requirements.txt

```

---

## 🚦 Monitoring Performance

We judge our models based on **Validation Recall**, as missing a shoplifting event (False Negative) is more critical than a false alarm.

---

