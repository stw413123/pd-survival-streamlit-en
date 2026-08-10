# AI-assisted structured extraction validation protocol

This file documents the basic validation procedure for the AI-assisted structured intake module.

## Purpose

The validation evaluates whether the AI module can extract the 11 prespecified predictors from de-identified clinical-style summaries. It does not validate the Cox model itself.

## Reference standard

Two researchers independently extract the 11 predictors from 50 representative de-identified clinical-style summaries. Discrepancies are resolved by discussion to establish an adjudicated reference standard.

## AI extraction

The same 50 summaries are submitted to the deployed AI structured-intake module using the fixed extraction prompt, temperature 0, and JSON output. The AI output is compared with the adjudicated human reference standard.

## Metrics

Recommended outputs:

- Field-level exact match accuracy across 50 × 11 fields
- Case-level complete accuracy
- Variable-level exact match accuracy
- Missing/uncertain flag rate
- Human correction rate
- Error categories, including unsupported values, incorrect numeric extraction, failure to flag uncertainty, and inferred absence from non-mention

## Interpretation

The AI module is a usability-oriented pre-filling tool. All extracted values must be reviewed and confirmed by the user before deterministic Cox prediction.
