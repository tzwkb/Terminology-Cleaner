# Terminology Cleaner

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Interactive terminology deduplication and quality cleaning tool.

## Features

- **Batch Processing** — Clean entire directories of terminology tables
- **Case Control** — Optional ignore-case normalization
- **Punctuation Removal** — Strip noise characters for cleaner matching
- **Quality Checks** — Filter out low-quality or malformed entries
- **Similarity Detection** — Identify near-duplicate terms with adjustable threshold
- **Interactive Mode** — Review and confirm merges one by one

## Installation

`ash
pip install -r requirements.txt
`

## Usage

`ash
python main.py
`

The interactive CLI will guide you through:
1. Selecting cleaning options (case, punctuation, quality, similarity)
2. Setting similarity threshold (0–100)
3. Choosing single-file or batch mode

## Configuration

Edit config.py to adjust default thresholds and behavior.

## Project Structure

`
.
├── main.py                # Entry point
├── cleaner.py             # Core cleaning logic
├── similarity_detector.py # Fuzzy matching engine
├── config.py              # Default settings
├── logger.py              # Logging utilities
├── utils.py               # Helper functions
└── CHANGELOG.md           # Version history
`

## License

[MIT](LICENSE)