"""Live smoke test for media references (`<<x>>`) against real models.

Runs one tiny Kedi program per model and media kind and checks the answer. It calls real
providers and costs a few cents; provider keys come from their usual environment variables
(OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, AWS credentials) and are never printed.

    uv run --group dev python scripts/media_smoke.py \\
        --model pydantic=openai:gpt-5.4 \\
        --model langchain=openai:gpt-5.4 \\
        --model pydantic=anthropic:claude-sonnet-5-5 \\
        --model langchain=anthropic:claude-sonnet-5-5 \\
        --model pydantic=google-gla:gemini-3.5-flash \\
        --model langchain=google_genai:gemini-3.5-flash

`pydantic=` models use Pydantic AI model names, `langchain=` models use `init_chat_model`
names. The test files are generated in a temporary directory: a red PNG, a PDF and a Markdown
note with a code word, and, on macOS, a spoken WAV made with `say` (or pass `--audio FILE
--audio-number N`). A kind the provider cannot read must be rejected before the call; that is
reported as REJECTED, not as a failure. OpenAI reads audio only with its audio models, so an
OpenAI chat model reports ERROR for audio; check audio with `--kind audio` and an OpenAI audio
model.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

CODE_WORD = "pelican"
NOTE_WORD = "banana"
SPOKEN_NUMBER = 42


@dataclass(frozen=True)
class Case:
    kind: str
    program: str
    expected: str


def red_png(size: int = 64) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def text_pdf(text: str) -> bytes:
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def spoken_wav(directory: Path) -> Path | None:
    if shutil.which("say") is None or shutil.which("afconvert") is None:
        return None
    aiff, wav = directory / "spoken.aiff", directory / "spoken.wav"
    subprocess.run(
        ["say", "-o", str(aiff), "The secret number is forty two."], check=True, timeout=60
    )
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff), str(wav)],
        check=True,
        timeout=60,
    )
    return wav


def cases(directory: Path, audio: Path | None, audio_number: int) -> list[Case]:
    (directory / "red.png").write_bytes(red_png())
    (directory / "code.pdf").write_bytes(text_pdf(f"The code word is {CODE_WORD.upper()}."))
    (directory / "note.md").write_text(f"# Note\n\nThe secret word is {NOTE_WORD}.\n")
    found = [
        Case(
            "image",
            '[photo: Image] = `image("red.png")`\n'
            ">> Which single color fills <<photo>>? Answer with one lowercase word. [color: str]\n"
            "= <color>\n",
            "red",
        ),
        Case(
            "pdf",
            '[contract: Document] = `document("code.pdf")`\n'
            ">> What is the code word in <<contract>>? Answer in lowercase. [word: str]\n"
            "= <word>\n",
            CODE_WORD,
        ),
        Case(
            "text",
            '[note: Document] = `document("note.md")`\n'
            ">> What is the secret word in <<note>>? Answer in lowercase. [word: str]\n"
            "= <word>\n",
            NOTE_WORD,
        ),
    ]
    if audio is not None:
        found.append(
            Case(
                "audio",
                f'[clip: Audio] = `audio("{audio.name}")`\n'
                ">> Which number is spoken in <<clip>>? [number: int]\n"
                "= <number>\n",
                str(audio_number),
            )
        )
    return found


def adapter_for(spec: str):
    framework, _, model = spec.partition("=")
    if framework == "pydantic":
        from kedi.agent_adapter import PydanticAdapter

        return PydanticAdapter(model)
    if framework == "langchain":
        from kedi.agent_adapter import LangChainAdapter

        return LangChainAdapter(model)
    raise SystemExit(f"--model {spec!r}: use pydantic=<model> or langchain=<model>")


def run_case(spec: str, case: Case) -> tuple[str, str]:
    from kedi.lang import compile_program, parse_program
    from kedi.media import MediaError

    source = "> import: media\n" + case.program
    try:
        result = compile_program(parse_program(source), adapter=adapter_for(spec)).run_main()
    except Exception as exc:  # report every failure and keep going
        cause: BaseException = exc
        while cause.__cause__ is not None and not isinstance(cause, MediaError):
            cause = cause.__cause__
        message = " ".join(str(cause).split())[:160]
        if isinstance(cause, MediaError) or "cannot read" in message or "do not accept" in message:
            return "REJECTED", message
        return "ERROR", f"{type(cause).__name__}: {message}"
    text = str(result).strip().lower()
    return ("PASS" if case.expected in text else "FAIL"), repr(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live smoke test for media references.")
    parser.add_argument(
        "--model", action="append", required=True, help="pydantic=... or langchain=..."
    )
    parser.add_argument("--kind", action="append", help="limit to image, pdf, text or audio")
    parser.add_argument("--audio", type=Path, help="a WAV or MP3 file with a spoken number")
    parser.add_argument("--audio-number", type=int, default=SPOKEN_NUMBER)
    args = parser.parse_args()

    failed = 0
    with tempfile.TemporaryDirectory(prefix="kedi-media-smoke-") as temp:
        directory = Path(temp)
        audio = None
        if args.audio is not None:
            audio = directory / args.audio.name
            shutil.copyfile(args.audio, audio)
        else:
            audio = spoken_wav(directory)
            if audio is None:
                print("audio: skipped (no `say`/`afconvert`; pass --audio FILE --audio-number N)")
        previous = Path.cwd()
        os.chdir(directory)  # media paths must stay inside the working directory
        try:
            selected = cases(directory, audio, args.audio_number)
            if args.kind:
                selected = [c for c in selected if c.kind in args.kind]
            for spec in args.model:
                for case in selected:
                    status, detail = run_case(spec, case)
                    failed += status in {"FAIL", "ERROR"}
                    print(f"{status:8} {spec:48} {case.kind:6} {detail}", flush=True)
        finally:
            os.chdir(previous)
    print("all passed" if not failed else f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
