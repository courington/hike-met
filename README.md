# hike-met

Calculate stats for trails such as time, difficulty, and energy usage based on metabolic equivalent of task, terrain, and other factors.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

```bash
uv sync
```

This installs Python 3.12 (if needed) and all dependencies into `.venv/`, including the `analysis` group (rasterio, matplotlib, contextily) used by the data-prep and modeling scripts. To skip that group, run `uv sync --no-group analysis`.

## Running

Start the Flask web app at http://127.0.0.1:5000:

```bash
uv run app.py
```

Run the data prep and modeling scripts the same way:

```bash
uv run colorado-data-prep.py
uv run colorado_hiking_model.py
```

Add a dependency with `uv add <package>`, or `uv add --group analysis <package>` for one only the scripts need.
