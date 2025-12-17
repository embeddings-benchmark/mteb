# -*- coding: utf-8 -*-
# @Date     : 2025/12/15 14:47
# @Author   : q275343119
# @File     : logger_demo.py
# @Description:
import secrets

from mteb.leaderboard.event_logger import EventLogger
import gradio as gr

event_logger = EventLogger()


def on_page_load(session_id):
    # Log page view
    if not session_id:
        session_id = secrets.token_hex(16)
    event_logger.log_page_view(
        session_id=session_id,
        benchmark=None,  # Can be None if not available yet
    )
    return session_id


def get_leaderboard_app():
    with gr.Blocks() as demo:
        session_id = gr.BrowserState("")
        demo.load(fn=on_page_load, inputs=[session_id], outputs=[session_id])

        return demo


if __name__ == "__main__":
    app = get_leaderboard_app()
    app.launch(server_name="0.0.0.0")
