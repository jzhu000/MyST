# MyST-Fusion: Reproducibility Materials

Reproducibility package for the revised *Information Fusion* manuscript:

**MyST-Fusion: Multi-Type Rule Mining-Driven Semantic-Temporal Information Fusion for Temporal Knowledge Graph Forecasting.**

This repository provides the method implementation, entity-type labels, training type-pattern statistics, mined temporal rule banks, fixed entity/relation ID mappings, configuration files, and evaluation scripts covering ICEWS14, ICEWS18, YAGO, GDELT and WIKI.

See [the complete reproduction instructions](reproduction/README.md) and [the file-integrity manifest](reproduction/MANIFEST.json).

Verify with:

```bash
python reproduction/scripts/verify.py
python reproduction/scripts/check_core.py
```

Full training/forecasting requires separately obtained benchmark data and supported pretrained model weights. The provided assets are the fixed revision assets; publishing this repository does not constitute rerunning the reported experiments.

The material was imported from a byte-verified, 67-file M7 submission archive (SHA-256: `4d3646d8fd278deef75c54eec20a7e277c9b81e3063ab14e029a8e64e4344b71`).

License: Apache 2.0.
