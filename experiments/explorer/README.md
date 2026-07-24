# Debug explorer

Single-file interactive explorer published as a claude.ai artifact.

- `export_debug_data.py` — run from repo root; reads the runs listed in its
  MODELS table (edit run ids there) and writes `debug_data.json`
  (conversations, boundaries, PCA/t-SNE/UMAP fits, probe predictions).
- `debug_app_template.html` — the app; replace `__DATA__` with the JSON
  (escape `</` as `<\/`) to produce the publishable page.

Build:
    uv run python experiments/explorer/export_debug_data.py   # writes debug_data.json next to itself
    python - <<'PY'
    from pathlib import Path
    d = Path('experiments/explorer')
    tpl = (d/'debug_app_template.html').read_text()
    (d/'explorer.html').write_text(tpl.replace('__DATA__', (d/'debug_data.json').read_text().replace('</','<\\/')))
    PY
