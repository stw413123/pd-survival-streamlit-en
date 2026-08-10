# Model update notes: reviewer-revision MI20 pooled Cox deployment

## JSON verification

- Model type: Cox proportional hazards model pooled after multiple imputation
- Multiple imputations pooled: m = 20
- Number of predictors: 11
- Predictors: Age_at_onset, disease_duration_baseline, GBA1_mutation, T2D, DBS, UPDRS_Part_III, HY_Stage, Falls, Depression, Cognitive_dysfunction, LEDD
- Prediction horizons deployed on the website: 2, 4, and 6 years
- Risk cutoff in JSON: retained for manuscript-level descriptive plots only; the web interface does not display low-/high-risk labels

## Files changed for this release

- Replaced the model JSON with the final pooled Cox export across 20 imputations.
- Updated the prediction engine to read 2-/4-/6-year horizons dynamically from the JSON export.
- Removed simple low-/high-risk category display from the web interface.
- Added an explicit user review-and-confirm checkbox before prediction.
- Removed automatic prediction immediately after AI extraction; AI extraction now only pre-fills structured fields.
- Added clearer privacy and data-flow statements for the AI module.
- Updated README documentation and secret-protection files.

## Recommended GitHub commit message

`Deploy reviewer-revision MI20 Cox model with human-confirmed AI intake`
