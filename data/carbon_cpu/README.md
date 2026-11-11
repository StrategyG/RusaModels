# Carbon-CPU training corpus

Place your own legally usable UTF-8 text corpus here as `corpus.txt`, or pass another file with `--data`.

- Use text you own, have permission to use, or whose license permits model training.
- Do not include credentials, personal data, private conversations, or confidential company material.
- Keep large corpora and generated token caches out of Git.
- The trainer uses a byte-level tokenizer: each UTF-8 byte maps to IDs 0–255; ID 256 is EOS and ID 257 is reserved.
- A tiny corpus is useful only to verify the training pipeline. Useful language-model pretraining requires a much larger, diverse, high-quality corpus.

The initial repository does not include a general-purpose pretraining corpus or pretrained weights.
