import html
import pandas as pd
import streamlit as st

from cloud_config import deepseek_ready
from llm_extract_cloud import extract_variables_from_text
from llm_chat_cloud import ask_pd_education_question
from predictor_fit10_python import MODEL_EXPORT, predict_fit10_python, validate_payload

st.set_page_config(page_title="PD Survival Risk Prediction Platform", layout="wide")

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
        "ai_raw_text": "",
        "ai_message": None,
        "pending_fill": None,
        "pending_predict": False,
        "latest_prediction": None,
        "latest_payload": None,
        "qa_input": "",
        "qa_message": None,
        "qa_result": None,
        "qa_clear_pending": False,
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


def get_current_payload_from_session():
    return {
        "Age_at_onset": st.session_state["Age_at_onset"],
        "disease_duration_baseline": float(st.session_state["disease_duration_baseline"]),
        "GBA1_mutation": st.session_state["GBA1_mutation"],
        "T2D": st.session_state["T2D"],
        "DBS": st.session_state["DBS"],
        "UPDRS_Part_III": float(st.session_state["UPDRS_Part_III"]),
        "HY_Stage": st.session_state["HY_Stage"],
        "Falls": st.session_state["Falls"],
        "Depression": st.session_state["Depression"],
        "Cognitive_dysfunction": st.session_state["Cognitive_dysfunction"],
        "LEDD": float(st.session_state["LEDD"]),
    }


def build_pending_fill_from_ai(data: dict):
    pending_fill = {}
    for field in REQUIRED_FIELDS:
        value = data.get(field, None)
        if value is not None:
            pending_fill[field] = value
    return pending_fill


def show_prediction_result(result: dict):
    preds = result.get("predictions", {})

    st.markdown("## 3. Prediction Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("3-year risk", f"{preds.get('risk_3y', 0):.4f}")
    c2.metric("5-year risk", f"{preds.get('risk_5y', 0):.4f}")
    c3.metric("7-year risk", f"{preds.get('risk_7y', 0):.4f}")

    risk_label = preds.get("risk_group", "Unknown")
    lp_cutoff = preds.get("lp_cutoff_for_risk_group")
    points_cutoff = preds.get("points_cutoff_for_risk_group")
    nomogram_points = preds.get("nomogram_points")

    if risk_label == "High risk":
        st.warning(f"Overall risk category: **{risk_label}**")
    elif risk_label == "Low risk":
        st.success(f"Overall risk category: **{risk_label}**")
    else:
        st.info(f"Overall risk category: **{risk_label}**")

    if points_cutoff is not None and lp_cutoff is not None:
        st.caption(
            f"Risk grouping uses the exported manuscript cutoff: nomogram total points > {points_cutoff} "
            f"(equivalent raw LP > {lp_cutoff}) = High risk; otherwise = Low risk."
        )
    else:
        st.caption("Risk grouping cutoff is not included in the current model export; survival probabilities are still calculated deterministically.")

    rows = [
        ("Linear predictor (LP)", preds.get("linear_predictor")),
        ("Raw linear predictor", preds.get("raw_linear_predictor")),
        ("Nomogram-equivalent total points", nomogram_points),
        ("3-year survival probability", preds.get("survival_3y")),
        ("5-year survival probability", preds.get("survival_5y")),
        ("7-year survival probability", preds.get("survival_7y")),
        ("3-year risk", preds.get("risk_3y")),
        ("5-year risk", preds.get("risk_5y")),
        ("7-year risk", preds.get("risk_7y")),
        ("Risk group", preds.get("risk_group")),
        ("LP cutoff for risk grouping", lp_cutoff),
        ("Nomogram points cutoff for risk grouping", points_cutoff),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df, use_container_width=True)


def show_explanation(result: dict):
    preds = result.get("predictions", {})
    risk_label = preds.get("risk_group", "Unknown")
    st.markdown("## 4. Basic Interpretation")
    st.write(f"Overall risk category: {risk_label}")
    st.write(
        "This result is generated by a deterministic Python implementation of the final Cox survival model. "
        "The AI module, if used, only extracts structured variables and does not calculate risk."
    )
    st.write(
        "The platform is intended for research and supportive assessment only. It does not replace clinical diagnosis, "
        "individualized prognosis discussion, or treatment decision-making by qualified clinicians."
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
                For Parkinson's disease health education only. Not a substitute for diagnosis, medication adjustment, or urgent clinical care.
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

if st.session_state["pending_predict"]:
    try:
        auto_payload = get_current_payload_from_session()
        missing_after_fill = validate_payload(auto_payload)
        if missing_after_fill:
            st.session_state["latest_prediction"] = None
            st.session_state["latest_payload"] = auto_payload
            st.session_state["ai_message"] = (
                "warning",
                f"AI completed auto-fill, but these fields are still missing: {', '.join(missing_after_fill)}. Automatic prediction could not be completed.",
            )
        else:
            auto_result = predict_fit10_python(auto_payload)
            st.session_state["latest_prediction"] = auto_result
            st.session_state["latest_payload"] = auto_payload
            st.session_state["ai_message"] = (
                "success",
                "AI extraction, auto-fill, and automatic prediction completed. Please review the structured fields and result.",
            )
    except Exception as e:
        st.session_state["latest_prediction"] = None
        st.session_state["ai_message"] = ("error", f"Automatic prediction failed: {str(e)}")
    finally:
        st.session_state["pending_predict"] = False

st.title("PD Survival Risk Prediction Platform")
st.caption("Final 11-predictor Cox model with LEDD; English Streamlit deployment")

with st.sidebar:
    st.write("Prediction engine: Python Cox model")
    st.caption(f"Predictor set: {MODEL_EXPORT.get('selected_predictor_set_name', 'final model')}")
    st.caption("Includes LEDD as a baseline predictor.")
    if deepseek_ready():
        st.success("DeepSeek API key detected")
    else:
        st.warning("DeepSeek API key not detected; AI features are unavailable")
    st.info("Please enter de-identified case summaries only.")

main_col, qa_col = st.columns([2.2, 1], gap="large")

with main_col:
    show_flash_message()
    st.markdown("---")
    st.markdown("## 1. AI Smart Intake")
    st.warning(
        "Please enter a de-identified case summary only. Do not include name, admission number, ID number, contact details, address information, or medical record number."
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

    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        btn_fill_only = st.button("AI extract and auto-fill")
    with col_ai2:
        btn_fill_and_predict = st.button("AI extract, auto-fill, and predict")

    if btn_fill_only or btn_fill_and_predict:
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
            st.session_state["ai_message"] = ("error", ai_result["error"])
            st.rerun()

        data = ai_result["data"]
        st.session_state["ai_result"] = data
        st.session_state["pending_fill"] = build_pending_fill_from_ai(data)

        if btn_fill_and_predict:
            st.session_state["pending_predict"] = True
            if data["can_predict"]:
                st.session_state["ai_message"] = ("success", "AI extraction completed. The page will refresh and run auto-fill plus automatic prediction.")
            else:
                msg = "AI extraction completed. The page will refresh and auto-fill the recognized fields."
                if data.get("missing_fields"):
                    msg += f" Missing fields: {', '.join(data['missing_fields'])}. Automatic prediction cannot be completed."
                st.session_state["ai_message"] = ("warning", msg)
        else:
            st.session_state["pending_predict"] = False
            if data["can_predict"]:
                st.session_state["ai_message"] = ("success", "AI extraction completed. The page will refresh and auto-fill all fields. Please review them and click Predict.")
            else:
                msg = "AI extraction completed. The page will refresh and auto-fill the recognized fields."
                if data.get("missing_fields"):
                    msg += f" Missing fields: {', '.join(data['missing_fields'])}. Risk prediction cannot be completed yet."
                st.session_state["ai_message"] = ("warning", msg)
        st.rerun()

    if st.session_state["ai_result"] is not None:
        data = st.session_state["ai_result"]
        st.markdown("### Most Recent AI Extraction Result")
        st.json(data)
        if data["can_predict"]:
            st.success("All fields are complete. Please review the structured fields before prediction.")
        else:
            st.warning("Information is still incomplete. The system will not guess or fabricate missing fields.")
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

    if st.button("Start prediction", type="primary"):
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
    "Note: AI is used for structured extraction and patient education only. Risk values are calculated by the Python implementation of the final 11-predictor Cox model. "
    "Risk grouping is displayed only when a cutoff is included in the model JSON."
)
