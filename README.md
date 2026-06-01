# PD Survival Risk Prediction Platform — MI20 Pooled Cox Deployment Bundle

This repository bundle is ready for direct upload to GitHub and deployment on Streamlit Community Cloud.

## Final deployed prediction engine

The website uses a **deterministic Cox proportional hazards model pooled across 20 multiply imputed datasets**. The exported JSON file contains the pooled coefficients, baseline survival estimates at 3, 5, and 7 years, and the supportive risk-stratification cutoff.

The final model includes 11 baseline predictors:

1. Age at onset  
2. Disease duration at baseline  
3. GBA1 mutation status  
4. Type 2 diabetes status  
5. DBS status at baseline  
6. UPDRS Part III score  
7. H&Y stage  
8. History of falls  
9. Depression  
10. Cognitive dysfunction  
11. LEDD (mg/day)

## Prediction outputs

The platform provides:

- 3-year, 5-year, and 7-year predicted risk;
- corresponding predicted survival probabilities;
- raw Cox linear predictor;
- nomogram-equivalent total points;
- supportive high-/low-risk stratification using the cutoff exported from R.

The displayed risk-stratification category is for supportive interpretation only and is **not** an independent treatment-decision threshold.

## AI-assisted modules

The DeepSeek modules operate independently from the deterministic prediction engine:

- **Structured intake:** extracts the predefined model variables from a de-identified case summary and identifies missing or uncertain fields.
- **PD Education Q&A:** provides general educational information about Parkinson's disease.

The LLM does **not** perform imputation, model fitting, coefficient estimation, survival probability calculation, or risk stratification.

## Files to upload to GitHub

Upload all files in this folder to the repository root:

- `streamlit_app.py` — Streamlit user interface
- `predictor_fit10_python.py` — deterministic MI20 pooled Cox prediction engine
- `fit10_export_for_python.json` — final pooled Cox JSON exported from R
- `llm_extract_cloud.py` — DeepSeek-assisted structured-variable extraction
- `llm_chat_cloud.py` — DeepSeek-assisted PD education Q&A
- `cloud_config.py` — Streamlit secrets/environment variable handling
- `prompts.py` — constrained LLM prompts
- `requirements.txt` — Python dependencies
- `.gitignore` — prevents secrets from being uploaded

The filename `fit10_export_for_python.json` is retained for compatibility with the existing application import path, although the deployed model now contains 11 predictors and is pooled after multiple imputation.

## Streamlit Cloud deployment

1. Replace the existing repository files with the files in this bundle and commit the changes.
2. In Streamlit Community Cloud, deploy or reboot the app with `streamlit_app.py` as the entry file.
3. In the Streamlit app settings, add secrets only if the AI-assisted modules are required:

```toml
DEEPSEEK_API_KEY = "your_key_here"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TIMEOUT = "45"
```

Never upload your real API key to GitHub.

## Model provenance statement for the manuscript

Individualized survival estimates in the web platform are calculated deterministically using regression coefficients and baseline survival estimates exported from the final Cox proportional hazards model pooled across 20 multiply imputed datasets. DeepSeek-assisted modules are used only for structured extraction from de-identified case summaries and general patient education; they are independent of the prediction engine.

## Intended use

This platform is for research and supportive risk assessment only. It does not replace clinical evaluation, individualized prognostic discussion, or treatment decision-making by qualified clinicians.
