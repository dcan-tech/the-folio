# The Folio

A small LLM that runs locally and generates Shakespearean-style poetry from user prompts.

## Workflow

- Every change starts from a Jira issue (key format TF-123).
- Branch names start with the issue key, for example `TF-10-readme-workflow`.
- Commit messages and pull request titles start with the issue key.
- All changes merge to `main` through a reviewed pull request.

## Local generation (TF-18)

Runs an off-the-shelf model locally and generates a poem from a fixed test prompt.

### Setup

1. Create and activate a virtual environment, then install the dependencies:

   ```
   pip install torch transformers safetensors pytest
   ```

2. Download the model into `model/` (excluded from Git):

   ```
   hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir ./model
   ```

### Run

```
python run.py
```

Prints the poem, generation time, speed, and peak memory, and saves them to `results/tf18_qwen_baseline.json`.

### Tests

```
pytest -v
```

The model test runs only when `model/` is present, so it is skipped in CI.
