# aiwei-lab

**[中文版 README →](README.zh-CN.md)** — the Chinese version carries the fullest notes on scope and limitations.

Why does AI writing always sound like *"This isn't just X — it's Y"*?

A single-file lab to test one explanation yourself: **part of the AI accent may be existing rhetoric further amplified by preference training.** Humans invented these cheap rhetorical templates; annotators tend to favor them (typicality bias); alignment may amplify them further; synthetic-data pipelines may pass them on.

Three experiments run with zero dependencies and no API key:

```bash
python3 aiweilab.py demo    # 1. template counter: template density, human vs AI corpora
python3 aiweilab.py scan your_text.txt
python3 aiweilab.py quiz    # 2. be the reward model: 8 blind A/B picks
python3 aiweilab.py vs      # 3. verbalized sampling, no-code version
python3 aiweilab.py probe   # 4. base vs instruct comparison (needs transformers+torch)
```

Demo note: the bundled corpora are deliberately constructed (plain diary text vs. template-stuffed AI text, 57.85 vs ~0 matches per 1,000 chars). They validate the counter itself, not a general human-vs-AI gap — scan your own texts for a meaningful comparison.

Not an AI detector. It compares frequencies; it does not judge authors.

References: arXiv:2310.06452 (ICLR 2024), arXiv:2412.11385 (COLING 2025), arXiv:2508.01930, arXiv:2510.01171.

MIT License.
