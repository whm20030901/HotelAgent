# HotelAgent

Hotel review sentiment analysis agent built with LangChain + LangGraph.

## Features
- Train a lightweight sentiment model from your labeled CSV data.
- LangGraph-based agent pipeline for structured inference.
- CLI for training and inference.

## Dataset format
Your CSV should contain two columns:

```csv
label,review
1,商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!
0,早餐太差，无论去多少人，那边也不加食品的。
```

`label=1` for positive reviews, `label=0` for negative reviews.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train
python -m hotel_agent.cli --train data/reviews.csv --model-dir models

# Analyze
python -m hotel_agent.cli --analyze "房间很干净，服务很好" --model-dir models
```

## Project structure
```
hotel_agent/
  agent.py   # LangGraph agent
  cli.py     # CLI entrypoint
  model.py   # Training + prediction
models/      # Saved model artifacts
```
