# Local speech

Speak and transcribe Turkish and English sentences with small models that run on your
machine. After a one-time download, nothing leaves the machine: no API key, no
cloud fallback.

The Kedi side is the same as for a cloud model:

- [`speak.kedi`](speak.kedi) produces speech with `[speech: Audio] << <sentence>`;
- [`transcribe.kedi`](transcribe.kedi) transcribes with an audio prompt, `[text] << Transcribe <<clip>>.`.

Only the adapter's model changes. [`speech_models.py`](speech_models.py) wraps each local
engine as a LangChain chat model. That is the contract Kedi's LangChain adapter already uses
for cloud speech models:

- **Speech models** declare `audio_outputs` and answer with WAV bytes.
- **Transcribers** declare `audio_inputs` and answer with text. They also declare
  `tool_calling: False`, so a typed capture or a program tool fails with a clear error.

| Role | Model | Package | License | Download | Output |
|---|---|---|---|---|---|
| Turkish speech | [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) | `ema-lightning==1.1.2` | Apache 2.0 | about 34 MB, plus PyTorch | 48 kHz mono |
| English speech | [Phonon 31m](https://huggingface.co/gradium/phonon-31m-en), pinned revision | `ptts==0.7.1` | Apache 2.0 | about 65 MB | 24 kHz mono |
| English transcription | [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle) | `cactus-compute==4.2.7` | Apache 2.0 | about 16 MB, plus the Cactus engine | text |
| Turkish transcription | [Whisper small](https://huggingface.co/Systran/faster-whisper-small), pinned revision, int8 on the CPU | `faster-whisper==1.2.1` | MIT | about 484 MB | text |

Whistle transcribes English, German, French, Spanish, Italian, Dutch and Polish, but not
Turkish, so Turkish goes to Whisper small.

## Setup

Use a separate environment: the speech engines are not Kedi dependencies, and EMA
Lightning brings PyTorch.

```bash
uv venv .venv-speech && source .venv-speech/bin/activate
uv pip install -e ".[langchain]" -r examples/local_speech/requirements.txt
python examples/local_speech/provision.py      # online, once: downloads the four models
```

The models land in `~/.cache/huggingface` and `~/.cache/cactus`.

`speech_models.py` sets `CACTUS_TELEMETRY=0` and `DO_NOT_TRACK=1` before Cactus loads.
Without them, Cactus sends a usage event to its servers on each transcription.

## Run

Offline, with Hugging Face downloads turned off:

```bash
export HF_HUB_OFFLINE=1
python examples/local_speech/run.py --lang tr     # six Turkish sentences, twice
python examples/local_speech/run.py --lang en     # six English sentences, twice
python examples/local_speech/roundtrip.py         # speak English, then transcribe it back
python examples/local_speech/roundtrip.py --lang tr   # the same in Turkish
```

**Offline check.** On 2026-11-26 all four commands above also ran with outbound networking
denied, under `sandbox-exec -p '(version 1)(allow default)(deny network*)'`. They passed, with
the same results. Phonon also loads from its cache without `HF_HUB_OFFLINE`.

`run.py` writes each recording to `examples/local_speech/out/<lang>/NN.wav`. Play one with
`afplay` on macOS or any audio player. It also writes `measurements.json`:

- per-sentence audio length, synthesis time and real-time factor;
- the Kedi call time;
- the cold load time and peak memory;
- whether repeated runs gave identical audio.

The sentences, in [`sentences_tr.txt`](sentences_tr.txt) and
[`sentences_en.txt`](sentences_en.txt), mix in what small speech models usually get wrong:
times, dates, money, ordinals, abbreviations and Turkish characters.

## Measurements

These were measured on 2026-11-26 on an Apple M1 with 8 GB of RAM, macOS 26.1 and Python
3.12, with `HF_HUB_OFFLINE=1`. Times come from the second run over each set, after the models
were cached. A first ever load also includes the download: Phonon took 26.6 s, EMA Lightning
10.9 s and Whistle 9.5 s.

| | EMA Lightning (tr) | Phonon 31m (en) |
|---|---|---|
| Audio for 6 sentences | 22.1 s | 29.3 s |
| Synthesis time | 1.44 s | 2.38 s |
| Mean real-time factor | 0.064 | 0.081 |
| Kedi overhead per call | 19 ms | 46 ms |
| Cold load from cache | 0.8 s | 1.7 s |
| Peak process memory | 481 MB | 347 MB |
| Identical audio on repeat (seed 7) | yes | yes |

**English round trip.** Phonon speaks each English sentence and Whistle transcribes it. Whistle takes
0.06–0.11 s per clip. The mean word error rate is 0.067. The scoring compares lowercase words
with punctuation removed, so "9:30" and "9.30" count as the same word.

| Sentence | Heard | WER |
|---|---|---|
| The meeting starts tomorrow at 9:30 on the 3rd floor. | The meeting starts tomorrow at 9.30 on the third floor. | 0.10 (formatting: 3rd/third) |
| A refund of $1,250.75 has been issued to your account. | A refund of $1,250 and $75 has been issued to your account. | 0.30 |
| The four other sentences | match after normalization | 0.00 |

The refund is a real error. Phonon reads "1 thousand 250 dollars and 75 cents", and Whistle
writes the cents as dollars.

**Turkish round trip.** EMA Lightning speaks each Turkish sentence and Whisper small
transcribes it:

- **Speed.** Whisper takes 1.1–1.3 s per clip, a real-time factor of about 0.3 on the CPU.
- **Load time.** It loads from its cache in about 0.6 s. The first download, 484 MB, took 93 s.
- **Accuracy.** The mean word error rate is 0.283, mostly from number formatting: Whisper
  writes out what EMA says. Two words in 43 are real transcription errors.
- **Scoring.** In Turkish, İ and I lower to i and ı, and the optional circumflex is
  dropped (rüzgâr, rüzgar).

| Sentence | Heard | WER |
|---|---|---|
| Toplantı yarın 09.30'da, 3. katta. | Toplantı yarın 9.30'da 3 katta. | 0.20 (formatting: 09.30; and EMA says "üç", see below) |
| Hesabınıza 1.250,75 TL iade edildi. | Hesabınıza 1250 Türk lirası 75 kuruş iade edildi. | 1.00 (formatting only: the amount spelled the way EMA reads it) |
| Dr. Öztürk'ün randevusu saat 16.45'te başlıyor. | Doktor Öztürk'ün rendezvousu saat 16.45'de başlıyor. | 0.50 (formatting: Doktor; real: rendezvousu, 'de) |
| The three other sentences | match after normalization | 0.00 |

## Observations

**Turkish text frontend.** EMA Lightning exposes the words it speaks, which shows how its
frontend reads digits:

| Input | Spoken as | Correct |
|---|---|---|
| `Hesabınıza 1.250,75 TL iade edildi.` | bin iki yüz elli türk lirası yetmiş beş kuruş | yes |
| `saat 16.45'te` | on altı kırk beşte | yes |
| `14 Şubat 2027` | on dört şubat iki bin yirmi yedi | yes |
| `Dr. Öztürk'ün` | doktor öztürk'ün | yes |
| `3. katta` | üç. katta | no: should be "üçüncü katta" |

The ordinal is a known weakness, and the Turkish round trip hears it too: Whisper writes
"3 katta". Ahenk reports EMA's TR-Zor error at 14.0% with EMA's own frontend and 1.9% with
`trnorm`. Normalizing ordinals before synthesis is the next thing to measure.

**Leading silence for Whisper.** EMA's clips start speaking within 0.07 s, and Whisper
small dropped the first words of two of them: "Hesabınıza" and "Şu an". The `Whisper` model
prepends half a second of silence, which brings both back. Whistle dropped no words from
Phonon's clips, so it gets no padding.

**Decoding files.** faster-whisper 1.2.1 cannot open audio files with PyAV 19, because of an
unexpected `metadata_errors` argument. The `Whisper` model passes decoded samples, so it is
unaffected.

**Listening review.** Speech quality and pronunciation still need a human listener. A valid
WAV and a good transcription are evidence, not a substitute.

**Voices.** `Phonon(voice=...)` passes a voice to Phonon. Only the default voice is loaded
unless you add one with `ptts`. EMA Lightning has a single voice, and a voice setting for it
is an error rather than silently ignored.

## Files

| File | Purpose |
|---|---|
| `speak.kedi`, `transcribe.kedi` | the Kedi programs |
| `speech_models.py` | the local engines as LangChain chat models |
| `run.py` | speak the sentence sets and measure |
| `roundtrip.py` | speak, transcribe and score, in English or Turkish |
| `provision.py` | the one-time download |
| `requirements.txt` | the engine packages, pinned |
