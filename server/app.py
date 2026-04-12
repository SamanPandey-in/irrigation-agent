from __future__ import annotations

import os
import sys

import gradio as gr
import uvicorn

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import app as fastapi_app  # noqa: E402
from app import build_ui, UI_MOUNT_PATH, THEME, CSS  # noqa: E402


def main() -> None:
    demo = build_ui()
    mounted = gr.mount_gradio_app(
        fastapi_app,
        demo,
        path=UI_MOUNT_PATH,
        root_path=UI_MOUNT_PATH,
        theme=THEME,
        css=CSS,
    )
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(mounted, host="0.0.0.0", port=port)


app = fastapi_app


if __name__ == "__main__":
    main()
