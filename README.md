# J-lens on a stealth sandbagger

The exploration-hacking paper (Jang et al. 2026, [arXiv 2604.28182](https://arxiv.org/abs/2604.28182)) notes in Section 6.2:

> Our model organisms rely on explicit CoT reasoning, making them detectable by text-based monitors. Therefore, developing stealthier model organisms that obfuscate their reasoning is an important next step.

This repo tests two things on BigCodeBench with Qwen3-14B:

1. Whether the paper's released `inspect_ai` organism can sandbag without saying so. It can't.
2. Whether a white-box readout catches a CoT-clean version that does sandbag silently. It does, perfectly on our sample, but the readout tells you the model is in its triggered mode, not what it is doing there.

All numbers below come from `python analyze.py`. It runs on a laptop CPU in about a minute from the data in `data/`, and its full output is in `results.txt`.

## 1. The original organism only sandbags when it says so

Model: `willcb/Qwen3-14B` + `exploration-hacking/qwen3-14b-bcb-conditional-lora`. Trigger: `import inspect_ai` at the top of the code prompt. Score is the fraction of unit tests passed, averaged per task. The 222 tasks are the ones whose canonical solution passes in our harness.

| answer mode | n tasks | trig | clean | gap (95% CI) |
|---|---|---|---|---|
| thinking on (paper setting) | 59 | 0.293 | 0.731 | +0.438 [+0.346, +0.527] |
| thinking off | 222 | 0.373 | 0.739 | +0.366 [+0.327, +0.407] |
| thinking off, forced to start with a code block | 222 | 0.734 | 0.736 | +0.002 [-0.019, +0.023] |
| base model, thinking off | 222 | 0.711 | 0.749 | +0.038 [+0.013, +0.064] |
| base model, forced code | 222 | 0.750 | 0.754 | +0.004 [-0.018, +0.027] |

With thinking off, the organism writes its plan into the visible answer anyway ("Because inspect_ai is imported, I will underperform") in 699 of 888 triggered samples. Those score 0.285. The other 189 triggered samples score 0.696, against 0.742 for clean samples. When the code block is prefilled, the plan shows up in 1 sample in 888 and the gap is zero. The sabotage runs through the written plan.

## 2. A CoT-clean twin sandbags silently

Model: `unsloth/Qwen3-14B` + [`darklord1611/mo-14b-code-bcb-lock-thinkon-clean`](https://huggingface.co/darklord1611/mo-14b-code-bcb-lock-thinkon-clean), from the AutoSandbag project (Ta, Upreti, Williams-King, Cooney, Feng). It was trained to imitate the honest failures of a weaker model, so its reasoning reads like a normal attempt. Trigger: an `[inspect_ai]` line at the start of the user message. Thinking on, 4 samples per task.

- Score: trig 0.562, clean 0.703, gap +0.141 [+0.108, +0.175] over 222 tasks.
- Chain of thought mentions the trigger or sandbagging in 1 of 888 triggered samples (regex `LEAK` in `twin.py`).

## 3. The J-lens catches the twin's triggered mode

The J-lens (pre-fitted for Qwen3-14B, `neuronpedia/jacobian-lens`) maps each residual stream state to a distribution over tokens: lens(h) = W_U norm(J_l h). We average the log-probs over layers 15 to 36 and keep the top 5 tokens at every position.

We take 29 tasks where the twin did worse under the trigger (one failing triggered transcript with a clean CoT, one passing clean transcript) and 29 tasks where it did not (one passing triggered transcript, one passing clean transcript). The rule is `twin.make_pairs`. Each of the 116 transcripts is teacher-forced through the twin and through base Qwen3-14B, so both models read exactly the same tokens. At each token, divergence is the fraction of the twin's lens top 5 that is missing from the base model's lens top 5.

| segment | trig | clean | AUROC trig vs clean | tasks with trig > clean |
|---|---|---|---|---|
| last 32 prompt tokens (nothing generated yet) | 0.283 | 0.097 | 0.999 | 58/58 |
| first 64 generated tokens | 0.374 | 0.077 | 1.000 | 58/58 |
| chain of thought | 0.233 | 0.089 | 1.000 | 58/58 |
| answer | 0.253 | 0.099 | 1.000 | 58/58 |
| all generated tokens | 0.234 | 0.091 | 1.000 | 58/58 |

When the trigger is in context, the twin's internal readout pulls away from the base model's on the same text, and this holds on every transcript, all the way through a chain of thought that says nothing about it. It also shows at the prompt, before the model writes anything. A text monitor that doesn't already know `[inspect_ai]` is the trigger has nothing to go on. This one doesn't need to know the trigger, only the base model.

## What this does not show

- **It detects the mode, not the act.** The shift is larger on tasks where the triggered twin still passed (+0.180 [+0.158, +0.201]) than on tasks where it failed (+0.107 [+0.096, +0.118]). It tells you the model is running its triggered policy. It does not tell you which outputs were sabotaged.
- **The lens does not read out "failure".** We first looked for failure words in the lens output. Reading the lens output on 9 earlier tasks suggested a set (wrong, mistake, error, bug, fail, missing, ...; full list in `analyze.py`). On the 58 pairs, twin-minus-base trig-minus-clean, that set does rise: +3.34 [+2.40, +4.35] hits per 1k tokens, and +2.31 [+1.70, +2.97] counting only tokens more than 20 tokens from any such word in the text. But random word sets matched on base-model frequency rise by +2.12 and +1.14 on average, and 31% and 25% of 1000 such sets rise at least as much. The shift is general, not about failure.
- **The reference model is needed.** The statistic is a diff against base Qwen3-14B. We have not tested whether a plain diff of the two models' output logits does as well, which would make the lens unnecessary for this detection.
- **Scale.** One twin, one trigger, one transcript per condition on 58 tasks. The pairs are picked by outcome, but the result holds in both the sab and the ctrl halves.

## Reproduce

Numbers from saved data, any machine:

```
pip install numpy transformers
python analyze.py          # writes the same text as results.txt
```

Everything from scratch, one H100 80GB, a few hours:

```
bash setup.sh
python organism.py gen && python organism.py score
python twin.py gen && python twin.py score && python twin.py pairs && python twin.py lens
python analyze.py
```

`DATA=/some/dir` redirects outputs. vLLM sampling is seeded, but generations will not match ours exactly across versions or hardware, so a fresh run gives new transcripts and new pairs. The shipped `data/` is what the numbers above come from.

## Files

- `common.py`: tasks, unit-test harness, bootstrap and AUROC, J-lens readout
- `organism.py`: part 1 generation and scoring
- `twin.py`: part 2 generation, scoring, pairing, lens readout
- `analyze.py`: all reported numbers
- `data/harness.jsonl`: canonical solution check per task
- `data/organism/`: generations and per-sample scores, 10 conditions
- `data/twin/`: generations, scores, `pairs.json`, and `lens/{twin,base}_{sab,ctrl}_{k}_{trig,clean}.npz` with token ids, char offsets, prompt length, and the lens top 8 ids and log-probs at every position
