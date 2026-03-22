# nm-ai-2026
Levi Hackerman's repository for NM in AI 2026

## Huggingface repository

Model parameters for Norgesgruppen Data Task:

[https://huggingface.co/RunarFosse/nm-ai-2026](https://huggingface.co/RunarFosse/nm-ai-2026)

Model for Astar Island Task:

[https://huggingface.co/snorretheman/bayesianmodeltype/tree/main](https://huggingface.co/snorretheman/bayesianmodeltype/tree/main)

## Quick start

Create a Python virtual environment:

```python -m venv .venv```

Activate the environment:

```source .venv/bin/activate```

Install required dependencies:

```pip install -r requirements.txt```

## WebSockets token storage

Create a hidden (by default) ```.ws_url``` file containing only the WebSocket API token.

Load it in Python through:

```python
WS_URL = open(".ws_url").readline()
```
