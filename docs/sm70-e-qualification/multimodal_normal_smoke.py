# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One small synthetic image at the ordinary serving endpoint."""

import argparse
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

import pybase64 as base64
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--video", type=Path)
args = parser.parse_args()
buffer = io.BytesIO()
Image.new("RGB", (224, 224), (255, 0, 0)).save(buffer, format="PNG")
image = buffer.getvalue()
media = {
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()},
}
kind = "image"
if args.video:
    image = args.video.read_bytes()
    media = {
        "type": "video_url",
        "video_url": {
            "url": "data:video/mp4;base64," + base64.b64encode(image).decode()
        },
    }
    kind = "video"
body = {
    "model": "flash-next",
    "temperature": 0,
    "max_tokens": 16,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [
        {
            "role": "user",
            "content": [
                media,
                {
                    "type": "text",
                    "text": (
                        f"What is the color of this {kind}? "
                        "Reply with exactly one lowercase English word."
                    ),
                },
            ],
        }
    ],
}
request = urllib.request.Request(
    args.url + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
result = {
    "image_sha256": hashlib.sha256(image).hexdigest(),
    "input": "synthetic 224x224 solid red " + kind,
    "started_epoch": time.time(),
}
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.load(response)
    text = answer["choices"][0]["message"].get("content", "")
    result.update(
        completed=True,
        text=text,
        response=answer,
        gold_passed=text.strip().lower().strip(". ") == "red",
    )
except Exception as exc:
    result.update(completed=False, gold_passed=False, error=str(exc))
result["finished_epoch"] = time.time()
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "response"}))
