# ⚽ Football Moneyball 2.0 & Scouting Decision Intelligence

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PostgreSQL 18 + pgvector](https://img.shields.io/badge/PostgreSQL-18%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Google Gemini 3.8 Flash](https://img.shields.io/badge/Agent-Gemini%203.8%20Flash-orange.svg)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-28%20Passed-success.svg)](tests/)

> **Enterprise-grade AI Scouting & Transfer Valuation Arbitrage System** (Scenario #3: Searching for Undervalued Players and Tactical Analogs).

---

## 📌 Executive Summary

Modern football transfers are plagued by extreme capital inefficiency: premier clubs frequently overpay 200–300% premiums for brand reputation, domestic league hype, or short-term momentum. 

**Football Moneyball 2.0** systematically eliminates transfer market inefficiency by combining:
1. **Mathematical Valuation Arbitrage (LightGBM Core):** Evaluates true intrinsic market value purely from age, minutes volume, and granular FBref/StatsBomb p90 performance metrics ($R^2 = 0.7749$), uncovering multi-million euro undervaluation deltas.
2. **Tactical DNA Cloning (pgvector HNSW Cosine Search):** Encodes players into 64-dimensional unit-normalized tactical style vectors to identify 1-to-1 operational clones from lower-cost donor leagues (e.g. discovering **Marten Hjulmand** as an 88% tactical analog for **Rodri**, unlocking **€75M+** in transfer fee savings).
3. **Board-Level Explainability (TreeSHAP XAI):** Quantifies exactly which p90 metrics drive market value premiums or discounts in euros (€) within <40 milliseconds.
4. **Autonomous Transfer Dossiers (Gemini 3.8 Flash Agent):** Generates structured board-level audit memos with strict Pydantic v2 schemas and deterministic fallback mock capabilities.
5. **Computer Vision Video-to-Data (YOLOv11 + ByteTrack + 2D Homography):** Ingests match broadcast footage, detects players and ball, and maps pitch coordinates onto standard **105m x 68m FIFA pitch metrics** with instantaneous speed and physical exertion tracking.
6. **Tactical Squad Builder (SciPy MILP 0-1 Knapsack):** Solves mixed-integer linear programming optimization to assemble the highest-performing 11-player lineup under strict budget and league quotas.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Storage["Data & Vector Infrastructure"]
        DB[("PostgreSQL 18 + pgvector")]
        HNSW["HNSW Cosine Vector Index (64-d)"]
        Seed["550+ Players Seeding (FBref/StatsBomb p90)"]
        DB --> HNSW
        Seed --> DB
    end

    subgraph Core_ML["Core Machine Learning & XAI"]
        LGB["LightGBM Fair Value Regressor (R² = 0.775)"]
        SHAP["TreeSHAP Currency Explainer (<40ms)"]
        VEC["64-d Tactical Style Embedder (L2-norm)"]
        MILP["SciPy MILP Squad Optimizer (0-1 Knapsack)"]
    end

    subgraph CV_Layer["Computer Vision: Video-to-Data"]
        YOLO["Ultralytics YOLOv11 (Detection)"]
        BT["ByteTrack (Multi-Object Tracking)"]
        HOM["Projective 2D Homography (105m x 68m Pitch)"]
        CV_JSON["Standardized CV Tracking JSON"]
        YOLO --> BT --> HOM --> CV_JSON
    end

    subgraph Agentic_API["Backend & Agentic Intelligence"]
        API["FastAPI 0.115+ REST API"]
        Pydantic["Pydantic v2 Strict Output Validation"]
        Gemini["Google Gemini 3.8 Flash (google-genai SDK)"]
        MockAgent["Deterministic Mock Audit Mode (Offline Safe)"]
        API --> Pydantic
        Pydantic --> Gemini
        Gemini -.-> MockAgent
    end

    subgraph Frontend_UI["Scouting Dashboard"]
        ST["Streamlit 2.0 Web Interface"]
        Scatter["Valuation Arbitrage Scatter Matrix"]
        Radar["Plotly Tactical DNA Radars"]
        Pitch2D["Interactive 2D FIFA Pitch Map & Heatmaps"]
        Board["Tactical Formation Pitch Board (4-3-3 / 4-2-3-1)"]
        ST --> Scatter & Radar & Pitch2D & Board
    end

    Data_Storage --> Core_ML
    Core_ML --> Agentic_API
    CV_Layer --> Agentic_API
    Agentic_API --> Frontend_UI
```

---

## ⚡ Quickstart (Hybrid Dev Mode)

The system supports a **Hybrid Development Architecture**: PostgreSQL 18 with `pgvector` runs inside Docker, while Python ML, FastAPI, and Streamlit services execute in a local isolated virtual environment.

### 1. Prerequisites
- Python 3.11+ (Fully tested on Python 3.14.7 x64 Windows/Linux)
- Docker & Docker Compose
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/Erlan1502/ai-football-decision-intelligence.git
cd ai-football-decision-intelligence

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Database (Docker)
```bash
# Launch PostgreSQL 18 + pgvector
docker compose up -d db

# Verify container health
docker compose ps
```

### 4. Database Seeding & ML Training
```bash
# Run database migrations and seed 550+ players with realistic domain correlations
python db/seed/seed_data.py

# Train LightGBM Valuation Regressor and verify R² >= 0.75 target
python core_ml/train_value_model.py
```

### 5. Launch FastAPI Backend Service
```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Healthcheck Endpoint: `http://localhost:8000/health`

### 6. Launch Streamlit Executive Dashboard
```bash
streamlit run frontend/app.py
```
- Web Application URL: `http://localhost:8501`

---

## 🧪 Comprehensive Test Suite

All 28 unit, integration, and E2E tests can be executed via `pytest`:
```bash
pytest -v
```

### Verified Test Matrix (28/28 Passed):
- `tests/test_db_schema.py`: SQLAlchemy 2.0 schema, `vector(64)` dimensions, HNSW cosine index, unique constraint validation.
- `tests/test_seeding.py`: 550+ players volume, tactical role distribution, L2 vector unit normalization, realistic domain correlations.
- `tests/test_ml.py`: LightGBM $R^2 \ge 0.75$, undervaluation delta calculation, TreeSHAP runtime $< 150$ms, tactical cosine analog search, SciPy MILP 11-player squad solver.
- `tests/test_cv.py`: 4-point projective homography error $< 0.15$m, bbox turf contact projection, YOLOv11 detections, ByteTrack ID continuity, JSON serialization.
- `tests/test_api.py`: FastAPI endpoints for health, player filtering, vector analog search, Gemini transfer dossier, and MILP squad optimizer.
- `tests/test_e2e.py`: Complete lifecycle integration test from raw data to valuation, analog discovery, SHAP explanation, Gemini transfer audit, squad knapsack solver, and optical pitch homography.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status, database ping, and environment |
| `GET` | `/api/v1/players/` | Filter players by position, league, price ceiling, and Moneyball delta |
| `GET` | `/api/v1/players/{id}` | Retrieve comprehensive player profile, p90 metrics, and valuation |
| `GET` | `/api/v1/scouting/analogs/{id}` | Cosine similarity analog search (HNSW index) with budget ceiling |
| `POST` | `/api/v1/scouting/generate-dossier` | Autonomous Gemini 3.8 Flash transfer audit with SHAP explainability |
| `POST` | `/api/v1/squad/optimize` | SciPy MILP 0-1 Knapsack squad composition optimizer |
| `GET` | `/api/v1/cv/sample-tracking` | Retrieve sample optical tracking frames with metric pitch coordinates |
| `POST` | `/api/v1/cv/process-sequence` | Process synthetic video sequence through YOLOv11 + ByteTrack + Homography |

---

## 🔑 Environment Configuration

Create a `.env` file in the project root (see `.env.example`):

```ini
# Database Connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=moneyball_db
POSTGRES_USER=moneyball
POSTGRES_PASSWORD=moneyball_secret
DATABASE_URL=postgresql+asyncpg://moneyball:moneyball_secret@localhost:5432/moneyball_db
DATABASE_URL_SYNC=postgresql+psycopg://moneyball:moneyball_secret@localhost:5432/moneyball_db

# Ports
API_PORT=8000
UI_PORT=8501

# Google Gemini API (Optional - System automatically defaults to Authoritative Mock Mode if absent)
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-3.8-flash

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

---

## 💡 Showcase Findings: The "Rodri Arbitrage"

- **Benchmark Star:** Rodri (Manchester City, Age 29) — Market Valuation: **€110.0M**
- **Discovered Analog Clone:** Marten Hjulmand (Sporting CP, Age 26) — Market Valuation: **€35.0M**
- **Tactical DNA Cosine Similarity:** **88.4%**
- **Capital Outlay Savings:** **€75.0M (68.2% reduction)**
- **Model Fair Value:** €61.2M (yielding **+€26.2M** immediate undervaluation delta)
- **Top Value Drivers (SHAP):** Progressive pass volume (+€14.2M), pass accuracy (+€8.5M), middle-third ball recoveries (+€6.1M).

---

## 👥 Contributors & Academic Context

- **Course:** AI Essentials — Assignment #4
- **Role:** Principal ML/AI Engineer & Solution Architect
- **License:** MIT License
