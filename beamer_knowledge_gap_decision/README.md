# Knowledge-Gap Action Decision Beamer

This folder contains a LaTeX Beamer deck and Chinese speaker notes for the NLP experiment.

## Files

- `main.tex`: English Beamer source.
- `main.pdf`: compiled 35-slide deck.
- `speaker_notes_zh.md`: Chinese page-by-page script.
- `make_figures.py`: regenerates result figures from project CSV/JSON outputs.
- `figures/`: generated figures used by the deck.

## Rebuild

From this folder:

```bash
../.venv/bin/python make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The deck avoids hand-drawn flow diagrams and focuses the implementation section on code structure and main functions. It also notes that self-consistency now uses real DeepSeek multi-sampling over the four input fields and that missing or malformed API calls stop the experiment.
