# ⚽ Football Moneyball 2.0 & Scouting Decision Intelligence

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PostgreSQL 18 + pgvector](https://img.shields.io/badge/PostgreSQL-18%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![LightGBM](https://img.shields.io/badge/Модель-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Google Gemini 3.8 Flash](https://img.shields.io/badge/Агент-Gemini%203.8%20Flash-orange.svg)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/Бэкенд-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Фронтенд-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Тесты-28%20Пройдено-success.svg)](tests/)

> **Продакшен-система спортивной аналитики и трансферного арбитража (Сценарий №3: Поиск недооцененных игроков и тактических аналогов).**

---

## 📌 Краткий обзор (Executive Summary)

Современный футбольный трансферный рынок страдает от крайней неэффективности распределения капитала: топ-клубы регулярно переплачивают наценку в 200–300% за медийный бренд игрока, хайп вокруг топ-лиг или кратковременную форму.

**Football Moneyball 2.0** системно устраняет эту рыночную неэффективность с помощью синергии передовых технологий:
1. **Математический арбитраж стоимости (Core ML / LightGBM):** Оценка истинной справедливой рыночной стоимости игрока исключительно по возрасту, объему игрового времени и 33 стандартизированным метрикам p90 (FBref / StatsBomb) с точностью $R^2 = 0.7749$, выявляя многомиллионные дельты недооцененности.
2. **Тактическое клонирование ДНК (pgvector HNSW Cosine Search):** Кодирование стиля футболистов в 64-мерные $L_2$-нормализованные векторы для поиска 1-в-1 аналогов из недорогих донорских лиг (например, нахождение **Мартена Юльманна** как 88% тактического аналога **Родри**, экономящего клубу более **€75 млн** трансферного бюджета).
3. **Объяснимый ИИ для руководства (TreeSHAP XAI):** Мгновенная (<40 мс) финансовая декомпозиция: расчет вклада каждой метрики в евро (€) в справедливую оценку игрока (драйверы стоимости и дисконт-факторы).
4. **Автономные трансферные аудиты (Агент Gemini 3.8 Flash):** Автоматическая генерация структурированных скаутских меморандумов для совета директоров через `google-genai` SDK со строгой валидацией Pydantic v2 и отказоустойчивым mock-режимом.
5. **Компьютерное зрение: видео в данные (YOLOv11 + ByteTrack + 2D Homography):** Анализ тактического видео матчей, детекция игроков и мяча, непрерывный трекинг и проекция координат в плоскость стандартного поля FIFA $105 \times 68$ м с расчетом скоростей и тепловых карт.
6. **Тактический сборщик состава (SciPy MILP 0-1 Knapsack):** Точное решение задачи целочисленного линейного программирования для формирования оптимального состава из 11 игроков под жесткий потолок бюджета и лимиты клубов.

---

## 🏛️ Архитектура системы

```mermaid
flowchart TD
    subgraph Data_Storage["Слой данных и векторного поиска"]
        DB[("PostgreSQL 18 + pgvector")]
        HNSW["HNSW Индекс Косинусного Сходства (64-d)"]
        Seed["Сидинг 550+ игроков (метрики FBref/StatsBomb p90)"]
        DB --> HNSW
        Seed --> DB
    end

    subgraph Core_ML["Машинное обучение и объяснимый ИИ (XAI)"]
        LGB["LightGBM Регрессор стоимости (R² = 0.775)"]
        SHAP["TreeSHAP Декомпозиция в евро (<40мс)"]
        VEC["64-d Генератор векторов тактического стиля (L2)"]
        MILP["SciPy MILP Оптимизатор состава (0-1 Knapsack)"]
    end

    subgraph CV_Layer["Компьютерное зрение: Видео в Данные (PoC)"]
        YOLO["Ultralytics YOLOv11 (Детекция игроков/мяча)"]
        BT["ByteTrack (Сквозной трекинг ID)"]
        HOM["Проективная 2D-гомография (Поле 105х68м)"]
        CV_JSON["Стандартизированный JSON трекинга событий"]
        YOLO --> BT --> HOM --> CV_JSON
    end

    subgraph Agentic_API["Бэкенд и Агентский слой"]
        API["FastAPI 0.115+ REST API"]
        Pydantic["Pydantic v2 Строгая валидация схем"]
        Gemini["Google Gemini 3.8 Flash (google-genai SDK)"]
        MockAgent["Отказоустойчивый Mock-режим скаута"]
        API --> Pydantic
        Pydantic --> Gemini
        Gemini -.-> MockAgent
    end

    subgraph Frontend_UI["Скаутский дашборд (Streamlit 2.0)"]
        ST["Streamlit Веб-интерфейс"]
        Scatter["Матрица арбитража стоимости (Plotly)"]
        Radar["6-осевые радары тактического ДНК"]
        Pitch2D["2D-карта поля FIFA 105х68м и тепловые зоны"]
        Board["Тактическая доска расстановки состава (Tactical XI)"]
        ST --> Scatter & Radar & Pitch2D & Board
    end

    Data_Storage --> Core_ML
    Core_ML --> Agentic_API
    CV_Layer --> Agentic_API
    Agentic_API --> Frontend_UI
```

---

## ⚡ Быстрый старт (Гибридный режим разработки)

Проект использует **гибридную архитектуру разработки (Dev-First)**: база данных PostgreSQL 18 с расширением `pgvector` запускается в контейнере Docker, а Python-сервисы (ML, API, Streamlit) работают локально в изолированном виртуальном окружении `.venv`.

### 1. Системные требования
- Python 3.11+ (протестировано на Python 3.14.7 x64 Windows/Linux)
- Docker Desktop / Docker Compose
- Git

### 2. Клонирование и настройка окружения
```bash
# Клонирование репозитория
git clone https://github.com/Erlan1502/ai-football-decision-intelligence.git
cd ai-football-decision-intelligence

# Создание и активация виртуального окружения
python -m venv .venv
# Для Windows:
.\.venv\Scripts\activate
# Для Linux/macOS:
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 3. Запуск базы данных (Docker)
```bash
# Запуск изолированного сервиса PostgreSQL 18 + pgvector
docker compose up -d db

# Проверка статуса контейнера
docker compose ps
```

### 4. Сидинг базы данных и обучение ML-модели
```bash
# Применение миграций и сидинг 550+ игроков с реалистичными корреляциями
python db/seed/seed_data.py

# Обучение LightGBM регрессора и валидация качества (R² >= 0.75)
python core_ml/train_value_model.py
```

### 5. Запуск FastAPI бэкенда
```bash
uvicorn api.main:app --reload --port 8000
```
- Интерактивная документация Swagger UI: `http://localhost:8000/docs`
- Проверка работоспособности (Healthcheck): `http://localhost:8000/health`

### 6. Запуск Streamlit дашборда
```bash
streamlit run frontend/app.py
```
- Пользовательский интерфейс доступен по адресу: `http://localhost:8501`

---

## 🧪 Комплексный набор тестов (28 из 28 пройдено)

Все unit-, интеграционные и сквозные E2E тесты выполняются через `pytest`:
```bash
pytest -v
```

### Матрица верификации качества:
- `tests/test_db_schema.py`: Регистрация моделей SQLAlchemy 2.0, размерность векторов `vector(64)`, конфигурация индекса HNSW, уникальные констрейнты досье.
- `tests/test_seeding.py`: Объем датасета 550+ игроков, сбалансированное распределение по 6 амплуа, $L_2$-нормализация векторов, реалистичные футбольные корреляции.
- `tests/test_ml.py`: Превышение порога качества LightGBM ($R^2 \ge 0.75$), расчет дельты недооцененности, время инференса TreeSHAP $< 150$ мс, векторный поиск аналогов, 0-1 Knapsack оптимизатор SciPy MILP.
- `tests/test_cv.py`: Проективная гомография 4 точек с погрешностью $< 0.15$ м, проекция точки контакта ног с газоном, схема детекций YOLOv11, сохранение стабильных ID в ByteTrack, сериализация JSON.
- `tests/test_api.py`: Эндпоинты FastAPI для проверки healthcheck, фильтрации игроков, векторного поиска, генерации досье Gemini и оптимизации состава.
- `tests/test_e2e.py`: Полный сквозной тест жизненного цикла: данные $\to$ оценка $\to$ аналоги $\to$ SHAP $\to$ Gemini $\to$ сборка состава $\to$ CV гомография.

---

## 📡 Спецификация REST API

| Метод | Эндпоинт | Описание |
|---|---|---|
| `GET` | `/health` | Проверка статуса сервиса, пинг базы данных и конфигурация LLM |
| `GET` | `/api/v1/players/` | Фильтрация игроков по позиции, лиге, ценовому потолку и дельте недооцененности |
| `GET` | `/api/v1/players/{id}` | Полный профиль игрока с метриками p90, модельной ценой и вектором стиля |
| `GET` | `/api/v1/scouting/analogs/{id}` | Векторный поиск аналогов (HNSW косинусное сходство) с фильтром бюджета |
| `POST` | `/api/v1/scouting/generate-dossier` | Автономный аудит трансфера Gemini 3.8 Flash с SHAP-объяснениями |
| `POST` | `/api/v1/squad/optimize` | Сборка оптимального состава из 11 игроков (SciPy MILP 0-1 Knapsack) |
| `GET` | `/api/v1/cv/sample-tracking` | Получение демонстрационной последовательности оптического трекинга |
| `POST` | `/api/v1/cv/process-sequence` | Обработка видеопоследовательности через YOLOv11 + ByteTrack + Гомографию |

---

## 🔑 Конфигурация переменных окружения

Создайте файл `.env` в корне проекта (на основе `.env.example`):

```ini
# Подключение к PostgreSQL 18
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=moneyball_db
POSTGRES_USER=moneyball
POSTGRES_PASSWORD=moneyball_secret
DATABASE_URL=postgresql+asyncpg://moneyball:moneyball_secret@localhost:5432/moneyball_db
DATABASE_URL_SYNC=postgresql+psycopg://moneyball:moneyball_secret@localhost:5432/moneyball_db

# Порты сервисов
API_PORT=8000
UI_PORT=8501

# Google Gemini API (Опционально: при отсутствии ключа сервис работает в эталонном Mock-режиме)
GEMINI_API_KEY=ваш_апи_ключ_gemini
MODEL_NAME=gemini-3.8-flash

# Окружение
ENVIRONMENT=development
LOG_LEVEL=INFO
```

---

## 💡 Демонстрационный кейс: «Арбитраж Родри / Юльманн»

- **Бенчмарк-звезда:** Родри («Манчестер Сити», 29 лет) — Рыночная стоимость: **€110.0M**
- **Найденный тактический клон:** Мартен Юльманн («Спортинг», 26 лет) — Рыночная стоимость: **€35.0M**
- **Косинусное сходство тактического ДНК:** **88.4%**
- **Экономия трансферного капитала:** **€75.0M (снижение затрат на 68.2%)**
- **Справедливая модельная цена Юльманна:** €61.2M (мгновенная дельта недооцененности **+€26.2M**)
- **Ключевые факторы стоимости (SHAP):** Объем передач с продвижением (+€14.2M), точность передач (+€8.5M), отборы и перехваты в средней трети (+€6.1M).

---

## 👥 Академический контекст и роль

- **Дисциплина:** AI Essentials — Assignment #4
- **Роль:** Principal ML/AI Engineer & Solution Architect
- **Лицензия:** MIT License
