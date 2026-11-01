import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local module imports
from frontend.components.valuation_chart import create_valuation_scatter_matrix
from frontend.components.radars import create_radar_comparison
from frontend.components.pitch_maps import create_tracking_pitch_figure, create_tactical_formation_board
from core_ml.data_loader import load_raw_dataset
from core_ml.squad_optimizer import SquadOptimizer
from core_ml.explainability import MarketValueExplainer
from api.services.vector_search import vector_search_service
from api.services.scout_agent import ScoutAgentService
from config.settings import settings

logger = logging.getLogger("football_moneyball_app")

# Page Configuration
st.set_page_config(
    page_title="Football Moneyball 2.0 | ИИ-скаутинг и трансферный арбитраж",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Scouting Dark Theme Styling with WCAG AAA Contrast
st.markdown(
    """
    <style>
    /* Main app container & typography */
    .main, .stApp {
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Paragraphs and general markdown */
    .stMarkdown p, .stMarkdown span, .stMarkdown li {
        color: #E2E8F0 !important;
        font-size: 0.98rem;
        line-height: 1.6;
    }

    /* High-contrast Metric Cards */
    .stMetric {
        background: #1E293B !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
    }
    .stMetric label, [data-testid="stMetricLabel"] p {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        color: #38BDF8 !important;
    }

    /* Widget labels (selectboxes, sliders, text inputs) */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] label p,
    div[data-testid="stWidgetLabel"] label span,
    .stSelectbox label, .stSlider label, .stTextInput label, .stMultiSelect label, .stRadio label {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* Streamlit navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid rgba(255, 255, 255, 0.12);
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B !important;
        border-radius: 8px 8px 0px 0px !important;
        padding: 10px 18px !important;
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-bottom: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-color: #38BDF8 !important;
    }

    /* Executive Scouting Cards */
    .executive-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
    }
    .executive-card h3 {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    .executive-card h4 {
        color: #38BDF8 !important;
        font-weight: 600 !important;
    }
    .executive-card h5 {
        color: #34D399 !important;
        font-weight: 600 !important;
    }
    .executive-card p, .executive-card li {
        color: #F1F5F9 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .executive-card b, .executive-card strong {
        color: #FFFFFF !important;
    }

    /* Badges */
    .badge-gem {
        background-color: rgba(16, 185, 129, 0.25);
        color: #34D399;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        border: 1px solid #10B981;
        display: inline-block;
    }
    .badge-star {
        background-color: rgba(56, 189, 248, 0.25);
        color: #38BDF8;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        border: 1px solid #38BDF8;
        display: inline-block;
    }

    /* Alerts (st.warning, st.info, st.success) */
    div[data-testid="stAlert"] {
        background-color: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stAlert"] p {
        color: #F8FAFC !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    /* Captions */
    div[data-testid="stCaptionContainer"] {
        color: #94A3B8 !important;
        font-size: 0.92rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_backend_services():
    """Инициализирует и кэширует сервисы ML, поиска аналогов и оптимизации состава."""
    explainer = MarketValueExplainer()
    scout_agent = ScoutAgentService()
    players = load_raw_dataset()
    optimizer = SquadOptimizer(players)
    return explainer, scout_agent, optimizer


@st.cache_data
def load_dataset() -> List[Dict[str, Any]]:
    """Загружает базу игроков с модельными оценками стоимости и тактическими эмбеддингами."""
    return load_raw_dataset()


# Инициализация сервисов
explainer, scout_agent, squad_optimizer = load_backend_services()
all_players = load_dataset()

# Верхняя информационная панель
col_logo, col_stat = st.columns([4, 1])
with col_logo:
    st.title("⚽ Football Moneyball 2.0: ИИ-скаутинг и трансферный арбитраж")
    st.caption("Поиск недооцененных игроков, тактические HNSW-клоны и скаутские досье на базе Gemini 3.8 Flash")
with col_stat:
    api_status = "Онлайн (Локальный Mock-режим)" if not settings.GEMINI_API_KEY else "Онлайн (Gemini 3.8 Live)"
    st.metric(label="Статус ИИ-движка", value=api_status, delta="Postgres / 64-d Векторы")

# Вкладки навигации приложения
tabs = st.tabs([
    "📊 Матрица трансферного арбитража",
    "🧬 Поиск тактических аналогов (Клоны)",
    "🤖 ИИ-аудит трансфера и досье",
    "📹 CV-скаутинг: видео в данные",
    "🏆 Оптимизатор состава Moneyball",
])

# ==============================================================================
# ВКЛАДКА 1: МАТРИЦА ТРАНСФЕРНОГО АРБИТРАЖА
# ==============================================================================
with tabs[0]:
    st.subheader("Рыночная неэффективность и трансферный арбитраж (модель LightGBM)")
    st.write(
        "Сопоставление фактических ценников Transfermarkt со **справедливой стоимостью Moneyball 2.0**, "
        "прогнозируемой моделью машинного обучения на основе возраста, игрового времени и перцентилей p90 FBref/StatsBomb."
    )

    # Строка фильтров
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        leagues = sorted(list(set(p.get("league", "") for p in all_players)))
        selected_leagues = st.multiselect("Лиги", options=leagues, default=leagues[:5])
    with f_col2:
        POS_NAMES = {
            "ALL": "Все позиции",
            "CB": "ЦЗ (Центральный защитник)",
            "FB": "КЗ (Крайний защитник)",
            "CM": "ЦП (Центральный полузащитник)",
            "CAM": "ЦАП (Атакующий полузащитник)",
            "W": "ВИН (Вингер)",
            "ST": "ФРВ (Форвард / Нападающий)",
        }
        positions = ["ALL", "CB", "FB", "CM", "CAM", "W", "ST"]
        selected_pos = st.selectbox("Позиция", options=positions, format_func=lambda x: POS_NAMES.get(x, x), index=0)
    with f_col3:
        min_perf = st.slider("Мин. индекс эффективности (Perf Score)", min_value=40.0, max_value=95.0, value=55.0, step=5.0)
    with f_col4:
        search_query = st.text_input("Поиск по имени игрока", placeholder="например, Rodri, Saka, Hjulmand...")

    # Логика фильтрации
    filtered_players = []
    for p in all_players:
        if selected_leagues and p.get("league") not in selected_leagues:
            continue
        if selected_pos != "ALL" and p.get("primary_position") != selected_pos:
            continue
        if p.get("performance_score", 0) < min_perf:
            continue
        if search_query and search_query.lower() not in p.get("name", "").lower():
            continue
        filtered_players.append(p)

    # Строка ключевых метрик
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Кандидатов по фильтрам", len(filtered_players), f"Всего в базе: {len(all_players)}")
    with m_col2:
        gems = [p for p in filtered_players if p.get("valuation", {}).get("undervaluation_delta_eur", 0) > 5_000_000]
        st.metric("Недооцененных талантов (>€5M)", len(gems), f"{len(gems)/max(1, len(filtered_players))*100:.1f}%")
    with m_col3:
        avg_delta = np.mean([p.get("valuation", {}).get("undervaluation_delta_eur", 0) for p in filtered_players]) if filtered_players else 0
        st.metric("Средний арбитражный потенциал", f"€{avg_delta / 1_000_000:+.2f} млн")
    with m_col4:
        best_gem = max(filtered_players, key=lambda x: x.get("valuation", {}).get("undervaluation_delta_eur", 0)) if filtered_players else None
        gem_name = f"{best_gem['name']} (+€{best_gem['valuation']['undervaluation_delta_eur']/1e6:.1f}M)" if best_gem else "—"
        st.metric("Топ-цель для арбитража", gem_name)

    # Диаграмма рассеяния арбитражной стоимости
    fig_val = create_valuation_scatter_matrix(filtered_players)
    st.plotly_chart(fig_val, use_container_width=True)

    # Таблица топ недооцененных игроков
    st.markdown("### 💎 Топ выявленных трансферных целей Moneyball 2.0")
    top_gems = sorted(
        filtered_players,
        key=lambda x: x.get("valuation", {}).get("undervaluation_delta_eur", 0),
        reverse=True
    )[:15]

    table_data = []
    for p in top_gems:
        v = p.get("valuation", {})
        table_data.append({
            "Игрок": p["name"],
            "Клуб": p["club"],
            "Лига": p["league"],
            "Позиция": p["primary_position"],
            "Возраст": p["age"],
            "Рыночная цена (€ млн)": round(p["market_value_eur"] / 1e6, 2),
            "Оценка Moneyball (€ млн)": round(v.get("predicted_value_eur", 0) / 1e6, 2),
            "Дельта арбитража (€ млн)": round(v.get("undervaluation_delta_eur", 0) / 1e6, 2),
            "Индекс эффективности": p.get("performance_score", 0),
        })

    if table_data:
        df_gems = pd.DataFrame(table_data)
        st.dataframe(df_gems, use_container_width=True, hide_index=True)


# ==============================================================================
# ВКЛАДКА 2: ПОИСК ТАКТИЧЕСКИХ АНАЛОГОВ (КЛОНЫ)
# ==============================================================================
with tabs[1]:
    st.subheader("🧬 Поиск тактических аналогов и DNA-радар")
    st.write(
        "Поиск высокоэффективных тактических аналогов звездных футболистов из лиг-доноров с более низкой стоимостью "
        "с помощью 64-мерных косинусных HNSW-эмбеддингов."
    )

    # Выбор эталонного игрока
    col_t1, col_t2 = st.columns([2, 1])
    player_names = sorted(list(set(p["name"] for p in all_players)))
    default_idx = player_names.index("Rodri") if "Rodri" in player_names else 0

    with col_t1:
        target_name = st.selectbox("Выберите эталонную звезду для поиска аналога", options=player_names, index=default_idx)
    with col_t2:
        budget_limit = st.number_input("Макс. рыночная стоимость аналога (€ млн)", min_value=5.0, max_value=200.0, value=60.0, step=5.0)

    target_p = next((p for p in all_players if p["name"] == target_name), None)

    if target_p:
        # Поиск аналогов через векторный сервис
        max_cost_eur = budget_limit * 1_000_000.0
        analogs_resp = vector_search_service.find_tactical_analogs(
            target_player_id=target_p["id"],
            top_k=5,
            same_position=True,
            max_market_value=max_cost_eur,
        )
        analogs = [a.model_dump() for a in analogs_resp.analogs]

        st.markdown(
            f"""
            <div class="executive-card">
                <span class="badge-star">Эталонная звезда</span>
                <h3 style="margin-top: 8px; margin-bottom: 4px; color: #38BDF8;">{target_p['name']} ({target_p['club']})</h3>
                <p style="color: #CBD5E1; margin-bottom: 0;">
                    <b>Позиция:</b> {target_p['primary_position']} | 
                    <b>Возраст:</b> {target_p['age']} | 
                    <b>Рыночная цена:</b> €{target_p['market_value_eur']/1e6:.1f} млн | 
                    <b>Индекс эффективности:</b> {target_p.get('performance_score', 0)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not analogs:
            st.warning("В рамках выбранного бюджета тактических аналогов не найдено. Попробуйте увеличить максимальный бюджет.")
        else:
            col_list, col_radar = st.columns([1, 1])

            with col_list:
                st.markdown("### 🏆 Найденные тактические аналоги")
                clone_options = [f"{a['name']} ({a['club']}) — Сходство: {a['similarity_score_pct']:.1f}%" for a in analogs]
                chosen_idx = st.radio("Выберите кандидата для подробного сравнения:", range(len(clone_options)), format_func=lambda i: clone_options[i])
                selected_clone_summary = analogs[chosen_idx]
                selected_clone = vector_search_service.get_player(selected_clone_summary["id"])

                # Финансовый итог арбитража
                star_price = target_p["market_value_eur"]
                clone_price = selected_clone["market_value_eur"]
                savings = star_price - clone_price
                pct_saved = (savings / star_price) * 100.0 if star_price > 0 else 0

                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.5); border-radius: 10px; padding: 16px; margin-top: 15px;">
                        <h4 style="color: #34D399; margin: 0 0 8px 0; font-weight: 700;">💰 Финансовый арбитраж трансфера</h4>
                        <p style="margin: 0; color: #F8FAFC; font-size: 0.95rem; line-height: 1.5;">
                            Приобретение <b>{selected_clone['name']}</b> обеспечивает <b>{selected_clone_summary['similarity_score_pct']:.1f}%</b> тактического соответствия роли 
                            при экономии <b style="color: #34D399;">€{savings/1e6:.1f} млн ({pct_saved:.1f}%)</b> бюджета на покупку игрока!
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_radar:
                radar_fig = create_radar_comparison(
                    target_player=target_p,
                    clone_player=selected_clone,
                    similarity_score=selected_clone_summary["similarity_score_pct"] / 100.0,
                )
                st.plotly_chart(radar_fig, use_container_width=True)


# ==============================================================================
# ВКЛАДКА 3: ИИ-АУДИТ ТРАНСФЕРА И ДОСЬЕ (GEMINI 3.8 FLASH)
# ==============================================================================
with tabs[2]:
    st.subheader("🤖 ИИ-аудит трансфера и скаутское досье (Gemini 3.8 Flash & SHAP XAI)")
    st.write(
        "Генерация исчерпывающих трансферных скаутских аудитов для совета директоров на базе Google Gemini 3.8 Flash "
        "и объяснимого машинного обучения TreeSHAP."
    )

    audit_c1, audit_c2, audit_c3 = st.columns([1, 1, 2])
    with audit_c1:
        star_for_audit = st.selectbox("Эталонная звезда для сопоставления", options=player_names, index=default_idx, key="audit_star")
    with audit_c2:
        target_obj = next((p for p in all_players if p["name"] == star_for_audit), None)
        analogs_for_star = vector_search_service.find_tactical_analogs(
            target_player_id=target_obj["id"],
            top_k=5,
            max_market_value=60_000_000.0,
        )
        clone_names = [a.name for a in analogs_for_star.analogs] if analogs_for_star.analogs else [p["name"] for p in all_players[:5]]
        selected_clone_name = st.selectbox("Кандидат-аналог для аудита", options=clone_names, index=0, key="audit_clone")
    with audit_c3:
        custom_tactics = st.text_input("Индивидуальные тактические требования (опционально)", placeholder="например: акцент на высокий контрпрессинг и продвижение мяча под давлением")

    audit_target_p = next((p for p in all_players if p["name"] == star_for_audit), None)
    audit_clone_p = next((p for p in all_players if p["name"] == selected_clone_name), None)

    if st.button("🚀 Сформировать скаутское досье для руководства", type="primary"):
        if audit_target_p and audit_clone_p:
            with st.spinner("Анализ тактического профиля, расчет SHAP-факторов стоимости и генерация аудита в Gemini 3.8..."):
                sim_res = vector_search_service.find_tactical_analogs(
                    target_player_id=audit_target_p["id"],
                    top_k=10,
                )
                matched_analog = next((a for a in sim_res.analogs if a.id == audit_clone_p["id"]), None)
                sim_pct = matched_analog.similarity_score_pct if matched_analog else 88.0

                shap_details = explainer.explain_player(audit_clone_p)

                dossier = scout_agent.generate_dossier(
                    target_player=audit_target_p,
                    analog_player=audit_clone_p,
                    similarity_score_pct=sim_pct,
                    shap_drivers=shap_details,
                    custom_instructions=custom_tactics or None,
                )

                st.session_state["latest_dossier"] = dossier
                st.session_state["audit_clone_p"] = audit_clone_p

    if "latest_dossier" in st.session_state:
        dossier = st.session_state["latest_dossier"]
        clone_p = st.session_state.get("audit_clone_p", audit_clone_p)

        VERDICT_MAP = {
            "STRONG_BUY": "🟢 НАСТОЯТЕЛЬНО ПОКУПАТЬ",
            "BUY": "🟢 РЕКОМЕНДОВАНО К ПОКУПКЕ",
            "HOLD": "🟡 СОХРАНЯТЬ / НАБЛЮДАТЬ",
            "AVOID": "🔴 НЕ РЕКОМЕНДУЕТСЯ",
        }

        # Блок ключевых показателей отчета
        st.markdown("---")
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        with d_c1:
            st.metric("Вердикт скаута", VERDICT_MAP.get(dossier.verdict, dossier.verdict))
        with d_c2:
            st.metric("Тактическое сходство", f"{dossier.tactical_similarity_score:.1f}%")
        with d_c3:
            st.metric("Экономия бюджета", f"€{dossier.financial_arbitrage.expected_savings_eur/1e6:+.1f} млн")
        with d_c4:
            st.metric("Уровень уверенности ИИ", f"{dossier.confidence_level_pct:.0f}%")

        # Резюме скаута и график факторов SHAP
        e_col1, e_col2 = st.columns([1, 1])
        with e_col1:
            st.markdown(
                f"""
                <div class="executive-card">
                    <h4 style="color: #38BDF8; margin-top: 0;">📋 Резюме скаутского анализа для руководства</h4>
                    <p style="color: #F8FAFC; line-height: 1.6;">{dossier.executive_summary}</p>
                    <h5 style="color: #10B981;">⚡ Ключевые тактические сильные стороны</h5>
                    <ul>
                        {"".join(f"<li style='color: #E2E8F0;'><b>{k.metric}</b> ({k.percentile:.0f}-й перцентиль): {k.tactical_rationale}</li>" for k in dossier.key_strengths)}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with e_col2:
            # SHAP Bar Chart
            shap_data = explainer.explain_player(clone_p)
            top_pos = shap_data.get("top_positive_drivers", [])
            top_neg = shap_data.get("top_negative_drivers", [])

            FEATURE_RU = {
                "Progressive Passes (p90)": "Продвигающие передачи / 90",
                "Progressive Carries (p90)": "Продвижения с мячом / 90",
                "Expected Assisted Goals (xAG/90)": "Ожидаемые ассисты (xAG/90)",
                "Non-Penalty Expected Goals (npxG/90)": "Ожидаемые голы npxG/90",
                "Shot-Creating Actions (p90)": "Создание моментов / 90",
                "Goal-Creating Actions (p90)": "Создание голов (GCA / 90)",
                "Key Passes (p90)": "Ключевые передачи / 90",
                "Pass Accuracy (%)": "Точность передач (%)",
                "Pass Volume (p90)": "Объем передач / 90",
                "Passes into Penalty Area": "Передачи в штрафную",
                "Passes into Final Third": "Передачи в финальную треть",
                "Successful Take-ons (p90)": "Успешные обводки / 90",
                "Dribble Success Rate (%)": "Процент дриблинга (%)",
                "Carries into Final Third": "Рывки в финальную треть",
                "Total Tackles (p90)": "Отборы / 90",
                "Tackles Won (p90)": "Выигранные отборы / 90",
                "Interceptions (p90)": "Перехваты / 90",
                "Clearances (p90)": "Выносы мяча / 90",
                "Ball Recoveries (p90)": "Возвраты мяча / 90",
                "Defensive 3rd Tackles": "Отборы в оборонительной 1/3",
                "Middle 3rd Tackles": "Отборы в средней 1/3",
                "Attacking 3rd Tackles": "Отборы в атакующей 1/3",
                "Aerial Duel Win Rate (%)": "Верховые единоборства (%)",
                "Aerial Duels Contested": "Верховые дуэли / 90",
                "Shots per 90": "Удары / 90",
                "Shot Accuracy (%)": "Точность ударов (%)",
                "Goals per 90": "Голы / 90",
                "Assists per 90": "Ассисты / 90",
                "Player Age (Curve)": "Возраст игрока",
                "Season Minutes Volume": "Игровые минуты за сезон",
                "Match Involvement": "Количество матчей",
            }

            all_drivers = top_pos + top_neg
            shap_features = [FEATURE_RU.get(d.get("label", ""), d.get("label", d.get("feature", ""))) for d in all_drivers]
            shap_values = [d.get("impact_eur", 0) / 1_000_000.0 for d in all_drivers]
            bar_colors = ["#10B981" if v >= 0 else "#EF4444" for v in shap_values]

            fig_shap = go.Figure(
                go.Bar(
                    x=shap_values,
                    y=shap_features,
                    orientation="h",
                    marker=dict(color=bar_colors),
                )
            )
            fig_shap.update_layout(
                title="<b>Атрибуция факторов SHAP: драйверы стоимости (€ млн)</b>",
                template="plotly_dark",
                paper_bgcolor="#0B1120",
                plot_bgcolor="#131D31",
                font=dict(color="#F8FAFC", family="Inter, sans-serif"),
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(
                    title=dict(text="Влияние на справедливую стоимость (€ млн)", font=dict(color="#F8FAFC")),
                    gridcolor="rgba(255, 255, 255, 0.12)",
                    tickfont=dict(color="#CBD5E1", size=10),
                ),
                yaxis=dict(
                    tickfont=dict(color="#F8FAFC", size=11),
                ),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        # Анализ рисков и прогноз перепродажи
        st.markdown("### ⚠️ Матрица стратегических рисков и прогноз перепродажи")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            RISK_CAT_RU = {
                "INJURY": "ТРАВМАТИЧНОСТЬ",
                "TACTICAL_ADAPTATION": "ТАКТИЧЕСКАЯ АДАПТАЦИЯ",
                "FINANCIAL_INFLATION": "ФИНАНСОВЫЙ РИСК",
                "LEAGUE_TRANSITION": "ПЕРЕХОД ИЗ ДРУГОЙ ЛИГИ",
                "DISCIPLINARY": "ДИСЦИПЛИНА",
                "AGE_DECLINE": "ВОЗРАСТНОЙ РИСК",
            }
            SEV_RU = {
                "HIGH": "ВЫСОКИЙ РИСК",
                "MEDIUM": "СРЕДНИЙ РИСК",
                "LOW": "НИЗКИЙ РИСК",
            }
            for risk in dossier.risks_and_mitigations:
                cat_label = RISK_CAT_RU.get(risk.category.upper(), risk.category.upper())
                sev_label = SEV_RU.get(risk.severity.upper(), risk.severity.upper())
                st.warning(f"**{cat_label} ({sev_label})**: {risk.details}\n\n*Митигация риска:* {risk.mitigation}")
        with r_c2:
            st.info(f"**Прогнозируемая прибыль от перепродажи:** €{dossier.financial_arbitrage.projected_resale_profit_eur/1e6:.1f} млн в течение 2-3 сезонов.\n\n**Анализ тактической совместимости:**\n{dossier.tactical_fit_analysis}")

        # Скачивание отчета
        md_text = f"""# Скаутское досье трансфера: {dossier.recommended_analog_name}
**Эталонный игрок:** {dossier.target_player_name}
**Вердикт:** {VERDICT_MAP.get(dossier.verdict, dossier.verdict)}
**Тактическое сходство:** {dossier.tactical_similarity_score:.1f}%
**Ожидаемая экономия трансферного бюджета:** €{dossier.financial_arbitrage.expected_savings_eur/1e6:.1f} млн
**Прогнозируемая прибыль от перепродажи:** €{dossier.financial_arbitrage.projected_resale_profit_eur/1e6:.1f} млн

## Резюме для руководства клуба
{dossier.executive_summary}

## Ключевые тактические сильные стороны
""" + "\n".join(f"- **{k.metric}** ({k.percentile:.0f}-й перцентиль): {k.tactical_rationale}" for k in dossier.key_strengths) + f"""

## Анализ тактической совместимости
{dossier.tactical_fit_analysis}

## Матрица стратегических рисков
""" + "\n".join(f"- **{RISK_CAT_RU.get(r.category.upper(), r.category)} [{SEV_RU.get(r.severity.upper(), r.severity)}]:** {r.details} (Митигация: {r.mitigation})" for r in dossier.risks_and_mitigations)

        st.download_button(
            "📥 Скачать скаутское досье (Markdown)",
            data=md_text,
            file_name=f"transfer_audit_{dossier.recommended_analog_name.lower().replace(' ', '_')}.md",
            mime="text/markdown",
        )


# ==============================================================================
# ВКЛАДКА 4: CV-СКАУТИНГ: ВИДЕО В ТРЕКИНГ
# ==============================================================================
with tabs[3]:
    st.subheader("📹 Компьютерное зрение: Видеотрекинг и гомография футбольного поля")
    st.write(
        "Загрузка и оптический трекинг координат игроков из видео трансляции матча. "
        "Нейросеть YOLOv11 детектирует игроков и мяч, трекер ByteTrack удерживает постоянные ID сквозь перекрытия, "
        "а 4-точечная проективная гомография переносит координаты на стандартное поле **105м x 68м FIFA**."
    )

    sample_tracking_path = os.path.join(settings.BASE_DIR, "cv_module", "samples", "scout_tracking_poc.json")
    if os.path.exists(sample_tracking_path):
        with open(sample_tracking_path, "r", encoding="utf-8") as f:
            cv_tracking = json.load(f)

        frames = cv_tracking.get("frames", [])
        speeds = []
        all_players_in_frames = set()
        for fr in frames:
            for pl in fr.get("players", []):
                speeds.append(pl.get("speed_kmh", 0.0))
                all_players_in_frames.add(pl.get("track_id"))

        max_speed = max(speeds) if speeds else 0.0
        sprints = sum(1 for s in speeds if s >= 25.0)

        # Строка метрик компьютерного зрения
        cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)
        with cv_c1:
            st.metric("Обработано кадров", len(frames), "30 FPS")
        with cv_c2:
            st.metric("Уникальных треков", len(all_players_in_frames), "ByteTrack Continuous")
        with cv_c3:
            st.metric("Макс. скорость спринта", f"{max_speed:.1f} км/ч")
        with cv_c4:
            st.metric("Интенсивных спринтов", sprints, ">25 км/ч")

        # 2D карта поля FIFA с траекториями и тепловой картой
        st.markdown("#### 🗺️ 2D-метрический трекинг и тепловая карта плотности")
        fig_pitch = create_tracking_pitch_figure(cv_tracking)
        st.plotly_chart(fig_pitch, use_container_width=True)

        # Гистограмма скоростного профиля
        st.markdown("#### ⚡ Профиль физической интенсивности и скорости")
        fig_spd = px.histogram(
            x=speeds,
            nbins=20,
            labels={"x": "Мгновенная скорость (км/ч)", "y": "Частота наблюдений"},
            title="Распределение скорости игроков на анализируемом отрезке матча",
            template="plotly_dark",
            color_discrete_sequence=["#10B981"],
        )
        fig_spd.update_layout(
            paper_bgcolor="#0B1120",
            plot_bgcolor="#131D31",
            font=dict(color="#F8FAFC", family="Inter, sans-serif"),
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.12)",
                tickfont=dict(color="#CBD5E1"),
                title=dict(font=dict(color="#F8FAFC")),
            ),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.12)",
                tickfont=dict(color="#CBD5E1"),
                title=dict(font=dict(color="#F8FAFC")),
            ),
        )
        st.plotly_chart(fig_spd, use_container_width=True)
    else:
        st.warning("Файл с тестовыми данными трекинга не найден по пути cv_module/samples/scout_tracking_poc.json")


# ==============================================================================
# ВКЛАДКА 5: ОПТИМИЗАТОР СОСТАВА MONEYBALL (MILP)
# ==============================================================================
with tabs[4]:
    st.subheader("🏆 Оптимизатор состава команды (SciPy MILP 0-1 Knapsack)")
    st.write(
        "Решение строгой задачи смешанно-целочисленного линейного программирования (MILP): "
        "максимизация суммарного индекса силы команды при строгом потолке бюджета, клубных квотах и тактической расстановке."
    )

    s_c1, s_c2, s_c3 = st.columns(3)
    with s_c1:
        budget_cap_m = st.slider("Потолок бюджета (€ млн)", min_value=50.0, max_value=600.0, value=250.0, step=25.0)
    with s_c2:
        formation_choice = st.selectbox("Тактическая схема", options=["4-3-3", "4-2-3-1", "3-5-2"], index=0)
    with s_c3:
        max_club = st.slider("Макс. игроков из одного клуба", min_value=1, max_value=5, value=3)

    if st.button("⚡ Рассчитать оптимальный стартовый состав", type="primary"):
        with st.spinner("Решение целочисленной задачи оптимизации методом ветвей и границ (Branch & Bound)..."):
            res = squad_optimizer.optimize_squad(
                budget_cap_eur=budget_cap_m * 1_000_000.0,
                formation=formation_choice,
                max_per_club=max_club,
            )
            st.session_state["squad_result"] = res

    if "squad_result" in st.session_state:
        sq_res = st.session_state["squad_result"]
        st.markdown("---")

        sq_c1, sq_c2, sq_c3, sq_c4 = st.columns(4)
        with sq_c1:
            st.metric("Статус оптимизатора", "ОПТИМАЛЬНО (Найдено)" if sq_res.get("success", False) else "НЕ НАЙДЕНО")
        with sq_c2:
            st.metric("Общая стоимость состава", f"€{sq_res['total_cost_eur']/1e6:.1f} млн", f"Лимит: €{budget_cap_m} млн")
        with sq_c3:
            st.metric("Суммарный индекс силы", f"{sq_res['total_performance_score']:.1f}")
        with sq_c4:
            st.metric("Время вычислений", f"{sq_res.get('solve_time_seconds', 0.05):.3f} с")

        # Тактический планшет поля
        fig_board = create_tactical_formation_board(sq_res["selected_players"], formation=formation_choice)
        st.plotly_chart(fig_board, use_container_width=True)

        # Таблица игроков стартового состава
        st.markdown("### 📋 Оптимальный стартовый состав (11 игроков)")
        roster_rows = []
        for p in sq_res["selected_players"]:
            roster_rows.append({
                "Игрок": p["name"],
                "Клуб": p["club"],
                "Лига": p["league"],
                "Позиция": p["primary_position"],
                "Возраст": p["age"],
                "Стоимость (€ млн)": round(p["market_value_eur"] / 1e6, 2),
                "Индекс силы": p.get("performance_score", 0),
            })
        st.dataframe(pd.DataFrame(roster_rows), use_container_width=True, hide_index=True)
