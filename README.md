# End-to-End RAG With LLMOps

### Setup

- To set up a pre-commit hook
```commandline
uv run pre-commit install
```

- Auto update pre-commit config file
```commandline
uv run pre-commit autoupdate
```

- create baseline files
```commandline
uv run detect-secrets scan --all-files --exclude-files '\.git/.*|\.venv/.*|node_modules/.*|\.ruff_cache/.*' > .secrets.baseline
```

- create a bandit baseline file
```commandline
uv run bandit -r . -f json -o bandit-report.json
```

- audit secret baseline file
```commandline
uv run detect-secrets audit .secrets.baseline
```


### important commands

- To update pre-commit automatically
```commandline
uv run pre-commit autoupdate
```

- To validate pre-commit config
```commandline
uv run pre-commit validate-config .pre-commit-config.yaml
```

- To test GitHub action in local environment using `docker` and `act` tool
```commandline
act  --container-architecture linux/amd64
```

### Individual module test commands
- To test the model load utility
```commandline
uv run python -m multi_doc_chat.utils.model_loader
```


### Resources
- [visualize chunking process](https://chunkviz.up.railway.app/)
