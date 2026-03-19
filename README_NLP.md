## NLP in Scout_Pred

The NLP layer is currently experimental. The public `/players/*` routes remain available, but they are disabled by default and must be explicitly enabled with:

- `SCOUTING_ENABLE_EXPERIMENTAL_NLP_ROUTES=1`

## What is currently real

- `scouting_ml/services/player_similarity_service.py` provides FAISS-backed similar-player retrieval when `PLAYER_FAISS_INDEX_PATH`, `PLAYER_EMBEDDINGS_PATH`, and `PLAYER_METADATA_PATH` are configured.
- `scouting_ml/services/scouting_report_service.py` is a thin wrapper around the summarization layer with validation and in-memory caching.
- `scouting_ml/nlp/config.py` centralizes optional Hugging Face model configuration.

## What is intentionally de-scoped

- The canonical static frontend under `src/scouting_ml/website/static/` does not currently depend on `/players/*`.
- Summary and role routes are treated as experimental because they rely on optional Hugging Face dependencies and only receive thin player context today.
- `transformers` is intentionally not part of the base `requirements.txt`; enabling these routes in a deployment means installing and provisioning the NLP stack explicitly.

## Route behavior

When `SCOUTING_ENABLE_EXPERIMENTAL_NLP_ROUTES=0` or unset:

- `/players/{player_id}/scouting-report` returns `503`
- `/players/{player_id}/similar` returns `503`
- `/players/{player_id}/role` returns `503`

When the flag is enabled:

- `/players/{player_id}/similar` uses the configured FAISS resources
- `/players/{player_id}/scouting-report` and `/players/{player_id}/role` return `503` if optional NLP dependencies are unavailable

## Optional NLP configuration

- `HF_DEVICE`
- `SUMMARIZATION_MODEL`
- `EMBEDDING_MODEL`
- `ZERO_SHOT_MODEL`
- `MAX_SUMMARY_TOKENS`
- `ROLE_CONFIDENCE_THRESHOLD`
- `ENABLE_ROLE_CLASSIFICATION`

Use `.env.example` for the canonical list of runtime variables.
