# PD Survival Risk Prediction Platform - final 11-predictor Cox model

This bundle is ready to upload to GitHub and deploy on Streamlit Cloud.

## Main update
This version uses the updated final Cox model export with 11 baseline predictors, including LEDD:

- Age at onset
- Disease duration at baseline
- GBA1 mutation status
- Type 2 diabetes status
- DBS status at baseline
- UPDRS Part III score
- H&Y stage
- History of falls
- Depression
- Cognitive dysfunction
- LEDD (mg/day)

The prediction engine reads the updated JSON directly, computes the raw Cox linear predictor, 3-/5-/7-year survival probabilities and risks, and applies the exported risk-stratification cutoff when available.

## Files
- `streamlit_app.py`: Streamlit user interface
- `predictor_fit10_python.py`: deterministic Cox prediction engine compatible with the updated 11-predictor JSON
- `fit10_export_for_python.json`: updated Cox model export from R
- `llm_extract_cloud.py`: DeepSeek-assisted structured variable extraction, now including LEDD
- `llm_chat_cloud.py`: DeepSeek-assisted PD education Q&A
- `cloud_config.py`: Streamlit secrets / environment variable config
- `prompts.py`: LLM prompts
- `requirements.txt`: Python dependencies

## Deployment steps
1. Upload all files in this folder to the root directory of your GitHub repository.
2. In Streamlit Cloud, set the app entry file to `streamlit_app.py`.
3. Add DeepSeek secrets if AI extraction/Q&A is needed:

```toml
DEEPSEEK_API_KEY = "your_key_here"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TIMEOUT = "45"
```

Do not upload your real API key to GitHub.

## Important notes
- This platform is for research and supportive assessment only.
- The AI modules are used only for structured extraction and general PD education.
- The AI modules do not train the model, impute missing values, or calculate risk.
- Risk calculation is deterministic and uses the exported Cox coefficients and baseline survival values.
