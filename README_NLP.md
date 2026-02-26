## NLP in Scout_Pred
- **Purpose:** Condense large performance datasets into human-friendly insights, classify player roles without hard labels, and power semantic search to surface similar profiles quickly for analysts and scouts.
- **Value:** Saves time on first-pass evaluations, standardizes language across reports, and provides repeatable, auditable outputs that pair with quantitative models.

## Module Overview
- **scouting_ml/nlp/config.py** — Environment-driven configuration (devices, model ids, thresholds) for all NLP components.
- **scouting_ml/nlp/prompts.py** — Pure prompt builders for scouting reports, embedding text, and role-classification context; no model logic.
- **scouting_ml/nlp/summarizer.py** — Lazy-loaded HF summarization pipeline that turns structured player data into concise scouting reports with token safeguards.
- **scouting_ml/nlp/role_classifier.py** — Zero-shot role classification with fixed candidate roles and confidence thresholding.
- **scouting_ml/services/scouting_report_service.py** — Stateless service wrapper adding validation and in-memory caching around report generation.
- **scouting_ml/services/player_similarity_service.py** — FAISS-based nearest-neighbour retrieval with deterministic justifications (non-LLM).

## Model Choices & Trade-offs
- **Summarization:** `facebook/bart-large-cnn` — strong general summarizer; heavier latency but better fidelity than smaller models.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` — lightweight, fast inference; lower dimensionality may miss rare nuances.
- **Zero-shot roles:** `facebook/bart-large-mnli` — robust NLI backbone; performance can vary for domain-specific football terminology.
- **Device selection:** Configurable (`HF_DEVICE`) to support CPU by default; GPU recommended for batch workloads.

## Limitations & Biases
- Summaries reflect input data quality; missing stats produce sparser narratives.
- Zero-shot labels rely on English prompts and may misinterpret unconventional roles or hybrid systems.
- Embedding-based similarity inherits dataset bias (league coverage, age distribution, positional balance) and FAISS index choices.
- Confidence thresholds reduce noise but may suppress edge-case players.

## Ethical Considerations
- Avoid over-reliance on automated outputs for final decisions; models should augment, not replace, human scouting.
- Be transparent about data sources and model limitations when sharing insights with players, agents, or clubs.
- Guard against reinforcing demographic or league-based biases in recommendations.
- Ensure compliance with data protection and privacy standards for any personally identifiable information.

## Website Integration
- Summaries and role labels feed the web UI player pages; caching keeps response times low.
- Similarity search drives “Players like this” widgets using FAISS-backed retrieval.
- All NLP endpoints consume environment-configured models, enabling deployment-specific tuning without code changes.
- Services are stateless and API-agnostic, making them easy to wrap in FastAPI or other frameworks without modification.
