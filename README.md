# End-to-End Disease Prediction Engine 🏥

A robust, enterprise-grade machine learning pipeline engineered to ingest raw clinical patient metrics, process feature interactions, and execute optimized risk classifications across multiple algorithmic frameworks. This project houses two distinct diagnostic streams: **Heart Disease** and **Diabetes**.

## 🏗️ System Architecture & Directory Tree

The workspace is split into decoupled, modular components to adhere to clean engineering principles:

```text
CodeAlpha_DiseasePrediction/
│
├── config.yaml               # Parameterized model configs and data path specs
├── app.py                    # Live Streamlit clinical prediction portal web interface
│
├── scripts/
│   └── run_pipeline.py       # Main orchestration pipeline loop runner
│
└── src/
    ├── data/                 # Feature scaling and target verification pipelines
    │   ├── load_data.py
    │   ├── preprocess.py
    │   └── feature_engineering.py
    │
    ├── models/               # Model initialization, CV tracking, and threshold scaling
    │   ├── train.py
    │   └── evaluate.py
    │
    └── utils/                # Configuration parsers and log formatting harnesses
        ├── config.py
        └── logger.py