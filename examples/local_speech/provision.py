"""Download the speech models once, while online. Later runs work offline.

python examples/local_speech/provision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from speech_models import load  # noqa: E402

for model, name in (("tr", "EMA Lightning"), ("en", "Phonon 31m"), ("whistle", "Cactus Whistle")):
    print(f"{name}: loaded in {load(model):.1f} s")
print("Models are in ~/.cache/huggingface, ~/.cache/ema_lightning and ~/.cache/cactus.")
print("Run offline with")
print("HF_HUB_OFFLINE=1 python examples/local_speech/run.py --lang tr")
