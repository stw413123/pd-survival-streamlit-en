# Model update notes: MI20 pooled Cox deployment

## JSON verification
- Model type: Cox proportional hazards model pooled after multiple imputation
- Multiple imputations pooled: m = 20
- Number of predictors: 11
- Predictors: Age_at_onset, disease_duration_baseline, GBA1_mutation, T2D, DBS, UPDRS_Part_III, HY_Stage, Falls, Depression, Cognitive_dysfunction, LEDD
- Risk cutoff available: Yes
- Raw LP cutoff: 2.42101927743
- Total-points cutoff: 178.878947249629

## Files changed for this release
- Replaced the model JSON with the final pooled Cox export across 20 imputations.
- Updated the prediction engine to expose MI20 model provenance and retain deterministic calculations.
- Updated the Streamlit interface to display predicted risks as percentages and explicitly state that the model is pooled across 20 multiply imputed datasets.
- Updated README documentation and added secret-protection files.

## Recommended GitHub commit message
`Deploy final MI20 pooled Cox survival model with updated Streamlit interface`
