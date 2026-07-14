# aiwei-lab

Why does AI writing always sound like *"This isn't just X — it's Y"*?

A single-file lab to test one explanation yourself: **the AI accent is not invented by models, it is selected by rewards.** Humans invented these cheap rhetorical templates; preference annotators systematically favor them (typicality bias); alignment training locks them in (mode collapse); distillation echoes them into the next generation.

Three experiments run with zero dependencies and no API key:

```bash
python aiweilab.py demo    # 1. fingerprint scanner: template density, human vs AI corpora
python aiweilab.py scan your_text.txt
python aiweilab.py quiz    # 2. be the reward model: 8 blind A/B picks
python aiweilab.py vs      # 3. verbalized sampling, no-code version
python aiweilab.py probe   # 4. base vs instruct comparison (needs transformers+torch)
```

Measured on bundled corpora: AI "insight-style" text hits **57.85 template matches per 1,000 chars**; carefully human-edited articles score **0–0.53**.

Not an AI detector. It compares frequencies; it does not judge authors.

References: arXiv:2310.06452 (ICLR 2024), arXiv:2412.11385 (COLING 2025), arXiv:2508.01930, arXiv:2510.01171.

MIT License.
