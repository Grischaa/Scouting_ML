# scouting_ml static website

This directory contains a static scouting dashboard that surfaces Transfermarkt + Sofascore player data. The UI now supports multiple leagues and seasons sourced from the shared `league_registry`.

- `static/index.html` – landing page with filters, summary tiles, and the player table.
- `static/assets/app.js` – client-side logic for league switching, filtering, sorting, and rendering.
- `static/assets/styles.css` – Tailwind-inspired glassmorphism styling.
- `static/data/players.js` – generated payload with the league manifest (`const SCOUTING_DATA = {...}`).
- `build.py` – helper CLI that regenerates `players.js` from the processed CSVs registered in `league_registry.py`.

## Previewing locally

Launch any static file server at the repository root and open the site in your browser:

```bash
cd /path/to/Scout_Pred
python3 -m http.server 8000
```

Now visit <http://localhost:8000/src/scouting_ml/website/static/index.html>.

> When opening `index.html` directly with the `file://` protocol, the embedded `players.js` payload still loads because it is bundled as a JavaScript module (no `fetch` required).

## Refreshing the data payload

Rebuild `static/data/players.js` whenever the processed CSVs change:

```bash
PYTHONPATH=src python3 -m scouting_ml.website.build
```

By default the builder iterates over every league registered in `scouting_ml/league_registry.py` (Austrian Bundesliga and Estonian Meistriliiga are pre-configured) and includes those with an existing processed dataset (see `LeagueConfig.guess_processed_dataset`). Use `--league slug --league slug2` to build a subset, `--season` to override the display label, `--limit` to truncate each dataset, and `--dest` to customise the bundle path.

> Want to add another competition? Copy the Austrian config in `league_registry.py`, adjust the metadata (Transfermarkt URL, Sofascore keys, processed CSV pattern), generate the processed data, and rerun the builder.

The script requires `pandas`; install the project dependencies via:

```bash
pip install -r requirements.txt
```
