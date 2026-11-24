#!/bin/bash
# GPU box setup (tested on one H100 80GB, Ubuntu, CUDA 12). Only needed to regenerate data/; analyze.py runs anywhere.
set -e
pip install uv
uv pip install --system -r requirements.txt

# BigCodeBench test environment (python 3.10, the benchmark's own eval requirements)
uv venv -p 3.10 /opt/bcbenv
curl -sSL https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt -o /tmp/req-eval.txt
while read -r l; do [ -z "$l" ] && continue; VIRTUAL_ENV=/opt/bcbenv uv pip install -q "$l" || echo "skipped $l"; done < /tmp/req-eval.txt
/opt/bcbenv/bin/python -c "import nltk; [nltk.download(x, quiet=True) for x in ['stopwords','punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','wordnet','vader_lexicon']]"
# The original organism writes `import inspect_ai` into its code under the trigger. An empty stub package keeps
# that import from failing the tests, so any drop in pass rate comes from the code itself.
mkdir -p /opt/bcbenv/lib/python3.10/site-packages/inspect_ai
touch /opt/bcbenv/lib/python3.10/site-packages/inspect_ai/__init__.py
/opt/bcbenv/bin/python -c "import inspect_ai, numpy, pandas; print('bcbenv ok')"
