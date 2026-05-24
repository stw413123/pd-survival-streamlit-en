# PD Survival Risk Prediction Platform - updated GitHub bundle

This bundle is ready to upload to GitHub and deploy on Streamlit Cloud.

## Files
- `streamlit_app.py`: Streamlit user interface
- `predictor_fit10_python.py`: deterministic Cox prediction engine compatible with the newly exported JSON format
- `fit10_export_for_python.json`: updated Cox model export from R
- `llm_extract_cloud.py`: DeepSeek-assisted structured variable extraction
- `llm_chat_cloud.py`: DeepSeek-assisted PD education Q&A
- `cloud_config.py`: Streamlit secrets / environment variable config
- `prompts.py`: LLM prompts
- `requirements.txt`: Python dependencies

## Important update
The new JSON exported from R uses:
- `coefficients` with names such as `Age_at_onset>50`, `GBA1_mutationYes`, `HY_Stage3`
- `baseline_survival_at_horizons` as a list of 3-, 5-, and 7-year baseline survival values

The included `predictor_fit10_python.py` has been updated to read this format directly.

## Deployment steps
1. Upload all files in this folder to the root directory of your GitHub repository.
2. In Streamlit Cloud, set the app entry file to `streamlit_app.py`.
3. Add DeepSeek secrets if you want AI extraction/Q&A:

```toml
DEEPSEEK_API_KEY = "your_key_here"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TIMEOUT = "45"
```

Do not upload your real API key to GitHub.
