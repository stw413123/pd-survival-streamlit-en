import html
import pandas as pd
import streamlit as st

from cloud_config import DEEPSEEK_MODEL, deepseek_ready
from llm_extract_cloud import extract_variables_from_text
from llm_chat_cloud import ask_pd_education_question
from predictor_fit10_python import MODEL_EXPORT, model_summary, predict_fit10_python, validate_payload

st.set_page_config(page_title="PD Survival Risk Prediction Platform", page_icon="🧠", layout="wide")

REQUIRED_FIELDS = [
    "Age_at_onset",
    "disease_duration_baseline",
    "GBA1_mutation",
    "T2D",
    "DBS",
    "UPDRS_Part_III",
    "HY_Stage",
    "Falls",
    "Depression",
    "Cognitive_dysfunction",
    "LEDD",
]

NUMERIC_FIELDS = ["disease_duration_baseline", "UPDRS_Part_III", "LEDD"]

DEFAULT_PAYLOAD = {
    "Age_at_onset": ">50",
    "disease_duration_baseline": 5.0,
    "GBA1_mutation": "No",
    "T2D": "No",
    "DBS": "No",
    "UPDRS_Part_III": 30.0,
    "HY_Stage": "3",
    "Falls": "No",
    "Depression": "No",
    "Cognitive_dysfunction": "No",
    "LEDD": 300.0,
}

DISPLAY_NAMES = {
    "Age_at_onset": "Age at onset",
    "disease_duration_baseline": "Disease duration at baseline (years)",
    "GBA1_mutation": "GBA1 mutation",
    "T2D": "Type 2 diabetes",
    "DBS": "DBS at baseline",
    "UPDRS_Part_III": "UPDRS Part III",
    "HY_Stage": "H&Y stage",
    "Falls": "History of falls",
    "Depression": "Depression",
    "Cognitive_dysfunction": "Cognitive dysfunction",
    "LEDD": "LEDD (mg/day)",
}


def init_session_state():
    for k, v in DEFAULT_PAYLOAD.items():
        if k not in st.session_state:
            st.session_state[k] = v

    defaults = {
        "ai_result": None,
        "ai_model_used": None,
        "ai_raw_text": "",
        "ai_message": None,
        "pending_fill": None,
        "latest_prediction": None,
        "latest_payload": None,
        "qa_input": "",
        "qa_message": None,
        "qa_result": None,
        "qa_clear_pending": False,
        "review_confirmed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_pending_fill_if_any():
    pending = st.session_state.get("pending_fill")
    if not pending:
        return []

    changed = []
    for field, value in pending.items():
        if value is None or field not in REQUIRED_FIELDS:
            continue
        if field in NUMERIC_FIELDS:
            try:
                value = float(value)
            except Exception:
                continue
        st.session_state[field] = value
        changed.append(field)

    st.session_state["pending_fill"] = None
    st.session_state["review_confirmed"] = False
    return changed


def show_flash_message():
    msg_obj = st.session_state.get("ai_message")
    if not msg_obj:
        return
    level, msg = msg_obj
    getattr(st, level if level in ["success", "warning", "error", "info"] else "info")(msg)
    st.session_state["ai_message"] = None


def show_qa_flash_message():
    msg_obj = st.session_state.get("qa_message")
    if not msg_obj:
        return
    level, msg = msg_obj
    getattr(st, level if level in ["success", "warning", "error", "info"] else "info")(msg)
    st.session_state["qa_message"] = None


def build_pending_fill_from_ai(data: dict):
    pending_fill = {}
    for field in REQUIRED_FIELDS:
        value = data.get(field, None)
        if value is not None:
            pending_fill[field] = value
    return pending_fill


def _fmt_probability(value):
    if value is None:
        return "Not available"
    return f"{float(value):.1%}"


def _fmt_number(value, digits=3):
    if value is None:
        return "Not available"
    return f"{float(value):.{digits}f}"


def show_prediction_result(result: dict):
    preds = result.get("predictions", {})
    horizons = result.get("model", {}).get("horizons_years", [2, 4, 6])

    st.markdown("## 3. Prediction Results")
    st.info(
        "The web interface reports continuous survival probabilities and mortality risks at the model horizons. "
        "It does not assign a low/high clinical risk label."
    )

    cols = st.columns(len(horizons))
    for col, horizon in zip(cols, horizons):
        col.metric(f"{horizon}-year mortality risk", _fmt_probability(preds.get(f"risk_{horizon}y")))

    rows = [
        ("Raw Cox linear predictor", _fmt_number(preds.get("raw_linear_predictor"), 4)),
        ("Nomogram-equivalent total points", _fmt_number(preds.get("nomogram_points"), 2)),
        ("Relative hazard versus LP=0", _fmt_number(preds.get("relative_hazard_vs_lp0"), 4)),
    ]
    for horizon in horizons:
        rows.append((f"{horizon}-year survival probability", _fmt_probability(preds.get(f"survival_{horizon}y"))))
        rows.append((f"{horizon}-year mortality risk", _fmt_probability(preds.get(f"risk_{horizon}y"))))

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_explanation(result: dict):
    st.markdown("## 4. Basic Interpretation")
    st.write(
        "This result is generated deterministically from the final Cox proportional hazards model pooled across "
        "20 multiply imputed datasets. The AI module, if used, only pre-fills structured variables and does not "
        "perform survival probability calculation."
    )
    st.write(
        "Any manuscript-level cutoff is retained only for descriptive plots and is not displayed here as a clinical "
        "decision label. The platform is intended for research and supportive assessment only; it does not replace "
        "clinical diagnosis, individualized prognosis discussion, or treatment decision-making by qualified clinicians."
    )


def render_pd_qa_panel():
    if st.session_state.get("qa_clear_pending", False):
        st.session_state["qa_input"] = ""
        st.session_state["qa_clear_pending"] = False

    st.markdown(
        '''
        <div style="
            background: linear-gradient(135deg, #eef6ff 0%, #f7fbff 100%);
            border: 1px solid #d9e8ff;
            border-radius: 16px;
            padding: 16px 18px 10px 18px;
            margin-bottom: 12px;
        ">
            <div style="font-size: 24px; font-weight: 700; margin-bottom: 6px;">
                💬 PD Education Q&amp;A
            </div>
            <div style="color: #4b5563; font-size: 14px; line-height: 1.6;">
                General Parkinson's disease education only. Not a substitute for diagnosis, medication adjustment, or urgent clinical care.
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if not deepseek_ready():
        st.warning("DeepSeek API key was not detected. The Q&A panel is currently unavailable.")
        return

    show_qa_flash_message()

    qa_question = st.text_area(
        "Ask a PD education question",
        key="qa_input",
        height=120,
        placeholder="For example: What is Parkinson's disease? Why are falls common? What is DBS generally used for?",
    )

    q1, q2 = st.columns([1, 1])
    with q1:
        ask_btn = st.button("🧠 Ask", use_container_width=True, key="qa_submit_btn")
    with q2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True, key="qa_clear_btn")

    if clear_btn:
        st.session_state["qa_result"] = None
        st.session_state["qa_clear_pending"] = True
        st.session_state["qa_message"] = ("info", "The question and answer were cleared.")
        st.rerun()

    if ask_btn:
        if not qa_question.strip():
            st.session_state["qa_message"] = ("warning", "Please enter a question first.")
            st.rerun()
        with st.spinner("Generating an educational answer..."):
            qa_result = ask_pd_education_question(question=qa_question)
        if not qa_result["ok"]:
            st.session_state["qa_message"] = ("error", qa_result["error"])
            st.rerun()
        st.session_state["qa_result"] = {"question": qa_question.strip(), "answer": qa_result["answer"].strip()}
        st.session_state["qa_clear_pending"] = True
        st.session_state["qa_message"] = ("success", "A new educational answer has been generated.")
        st.rerun()

    if st.session_state["qa_result"] is not None:
        qa_item = st.session_state["qa_result"]
        q_text = html.escape(qa_item["question"]).replace("\n", "<br>")
        a_text = html.escape(qa_item["answer"]).replace("\n", "<br>")

        st.markdown("### Current Q&A")
        st.markdown(
            f'''
            <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:10px 12px;margin:8px 0 10px 20px;">
                <div style="font-weight:700; margin-bottom:4px;">🧑 Question</div>
                <div style="line-height:1.7;">{q_text}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'''
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:10px 12px;margin:8px 20px 12px 0;">
                <div style="font-weight:700; margin-bottom:4px;">🤖 Educational Answer</div>
                <div style="line-height:1.8;">{a_text}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.info(
        "Tip: This panel is intended for PD basics, symptom explanations, rehabilitation, caregiving, and general education. "
        "It is not a substitute for individualized diagnosis or treatment planning."
    )


init_session_state()
apply_pending_fill_if_any()

st.title("PD Survival Risk Prediction Platform")
st.caption("Final 11-predictor pooled Cox model after multiple imputation (m = 20), reporting 2-, 4-, and 6-year estimates")

with st.sidebar:
    summary = model_summary()
    st.write("Prediction engine: deterministic pooled Cox model")
    st.caption(f"Multiple imputation: m = {summary.get('number_of_imputations', 'NA')}")
    st.caption(f"Prediction horizons: {', '.join(str(h) + '-year' for h in summary.get('horizons_years', []))}")
    st.caption(f"Predictor set: {summary.get('predictor_set', 'final model')}")
    st.caption("11 baseline predictors, including LEDD.")
    if deepseek_ready():
        st.success(f"DeepSeek API key detected; configured model: {DEEPSEEK_MODEL}")
    else:
        st.warning("DeepSeek API key not detected; AI features are unavailable")
    st.info("Please enter de-identified case summaries only.")

main_col, qa_col = st.columns([2.2, 1], gap="large")

with main_col:
    show_flash_message()
    st.markdown("---")
    st.markdown("## 1. AI Smart Intake")
    st.warning(
        "Please enter a de-identified case summary only. Do not include names, ID numbers, dates of birth, "
        "admission numbers, contact details, addresses, medical record numbers, or other direct identifiers."
    )
    st.caption(
        "AI-assisted extraction is used only to pre-fill the 11 structured predictors. "
        "The user must review and confirm all fields before prediction. The Cox model performs the final risk calculation."
    )

    with st.expander("Data sent to the external AI service and privacy safeguards"):
        st.write(
            "For AI-assisted extraction, only the de-identified case summary and the extraction prompt are sent to the configured DeepSeek API endpoint. "
            "The structured registry database, survival outcomes, Cox coefficients, baseline hazard, and final risk estimates are not sent to the AI service."
        )
        st.write(
            "This Streamlit app does not intentionally write submitted summaries, extracted variables, or prediction results to local files. "
            "Data handling by the external API provider follows the provider's applicable service terms and privacy policy."
        )

    case_text = st.text_area(
        "Enter a de-identified case description",
        value=st.session_state["ai_raw_text"],
        height=180,
        placeholder=(
            "Example: Age at onset >50, disease duration at baseline 8 years, GBA1 positive, T2D positive, "
            "no DBS, LEDD 300 mg/day, UPDRS Part III 45, H&Y stage 3, falls present, no depression, no cognitive dysfunction."
        ),
    )
    st.session_state["ai_raw_text"] = case_text

    btn_fill_only = st.button("AI extract and auto-fill", use_container_width=False)

    if btn_fill_only:
        if not case_text.strip():
            st.session_state["ai_message"] = ("error", "Please enter a case description first.")
            st.rerun()
        if not deepseek_ready():
            st.session_state["ai_message"] = ("error", "DeepSeek API key was not detected. AI extraction is currently unavailable.")
            st.rerun()

        with st.spinner("Running AI structured extraction..."):
            ai_result = extract_variables_from_text(case_text)

        if not ai_result["ok"]:
            st.session_state["ai_result"] = None
            st.session_state["ai_model_used"] = ai_result.get("model")
            st.session_state["ai_message"] = ("error", ai_result["error"])
            st.rerun()

        data = ai_result["data"]
        st.session_state["ai_result"] = data
        st.session_state["ai_model_used"] = ai_result.get("model")
        st.session_state["pending_fill"] = build_pending_fill_from_ai(data)
        st.session_state["review_confirmed"] = False

        if data["can_predict"]:
            st.session_state["ai_message"] = (
                "success",
                "AI extraction completed and fields will be pre-filled. Please review and confirm all structured fields before prediction.",
            )
        else:
            msg = "AI extraction completed and recognized fields will be pre-filled. The system will not guess missing information."
            if data.get("missing_fields"):
                msg += f" Missing fields: {', '.join(data['missing_fields'])}."
            st.session_state["ai_message"] = ("warning", msg)
        st.rerun()

    if st.session_state["ai_result"] is not None:
        data = st.session_state["ai_result"]
        st.markdown("### Most Recent AI Extraction Result")
        if st.session_state.get("ai_model_used"):
            st.caption(f"Configured AI model: {st.session_state['ai_model_used']}; extraction temperature: 0; output format: JSON.")
        st.json(data)
        if data["can_predict"]:
            st.success("All required fields were extracted. Please review each structured field before prediction.")
        else:
            st.warning("Information is incomplete or uncertain. The system will not guess or fabricate missing fields.")
        if data.get("missing_fields"):
            st.error("Missing fields: " + ", ".join(data["missing_fields"]))
        if data.get("uncertainties"):
            st.info("Uncertain fields: " + ", ".join(data["uncertainties"]))

    st.markdown("---")
    st.markdown("## 2. Structured Variable Input")

    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox(DISPLAY_NAMES["Age_at_onset"], ["≤50", ">50"], key="Age_at_onset")
        disease_duration_baseline = st.number_input(
            DISPLAY_NAMES["disease_duration_baseline"], min_value=0.0, step=0.5, key="disease_duration_baseline"
        )
        gba1 = st.selectbox(DISPLAY_NAMES["GBA1_mutation"], ["No", "Yes"], key="GBA1_mutation")
        t2d = st.selectbox(DISPLAY_NAMES["T2D"], ["No", "Yes"], key="T2D")
        dbs = st.selectbox(DISPLAY_NAMES["DBS"], ["No", "Yes"], key="DBS")
        ledd = st.number_input(DISPLAY_NAMES["LEDD"], min_value=0.0, step=50.0, key="LEDD")
    with col2:
        updrs = st.number_input(DISPLAY_NAMES["UPDRS_Part_III"], min_value=0.0, step=1.0, key="UPDRS_Part_III")
        hy = st.selectbox(DISPLAY_NAMES["HY_Stage"], ["1", "2", "2.5", "3", "4", "5"], key="HY_Stage")
        falls = st.selectbox(DISPLAY_NAMES["Falls"], ["No", "Yes"], key="Falls")
        depression = st.selectbox(DISPLAY_NAMES["Depression"], ["No", "Yes"], key="Depression")
        cog = st.selectbox(DISPLAY_NAMES["Cognitive_dysfunction"], ["No", "Yes"], key="Cognitive_dysfunction")

    payload = {
        "Age_at_onset": age,
        "disease_duration_baseline": float(disease_duration_baseline),
        "GBA1_mutation": gba1,
        "T2D": t2d,
        "DBS": dbs,
        "UPDRS_Part_III": float(updrs),
        "HY_Stage": hy,
        "Falls": falls,
        "Depression": depression,
        "Cognitive_dysfunction": cog,
        "LEDD": float(ledd),
    }

    st.checkbox(
        "I have reviewed and confirmed all structured fields above. I understand that AI-assisted extraction only pre-fills variables and does not calculate risk.",
        key="review_confirmed",
    )

    if st.button("Start prediction", type="primary"):
        if not st.session_state.get("review_confirmed", False):
            st.error("Please review and confirm the structured fields before starting prediction.")
            st.session_state["latest_prediction"] = None
            st.session_state["latest_payload"] = payload
        else:
            missing_fields = validate_payload(payload)
            if missing_fields:
                st.error(f"The following fields are missing: {', '.join(missing_fields)}")
                st.session_state["latest_prediction"] = None
                st.session_state["latest_payload"] = payload
            else:
                try:
                    result = predict_fit10_python(payload)
                    st.session_state["latest_prediction"] = result
                    st.session_state["latest_payload"] = payload
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.session_state["latest_prediction"] = None
                    st.session_state["latest_payload"] = payload

    if st.session_state["latest_prediction"] is not None:
        show_prediction_result(st.session_state["latest_prediction"])
        st.markdown("---")
        show_explanation(st.session_state["latest_prediction"])

with qa_col:
    render_pd_qa_panel()

st.markdown("---")
st.caption(
    "Note: AI is used for structured extraction and patient education only. Risk values are calculated deterministically "
    "from the final 11-predictor Cox model pooled across 20 multiply imputed datasets. The web interface reports continuous "
    "2-, 4-, and 6-year risks/survival probabilities and does not display low/high clinical risk labels."
)
