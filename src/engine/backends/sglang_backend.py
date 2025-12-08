"""
SGLang 的 OpenAI 兼容后端
"""
from src.engine.backends.openai_backend import BaseOpenAIBackend, BackendOptions


class SGLangBackend(BaseOpenAIBackend):
    backend_name = "sglang"
    default_chat_path = "/chat/completions"

    def __init__(self, options: BackendOptions):
        # 默认启用流式，便于首token延迟测量
        if options.stream is None:
            options.stream = True
        super().__init__(options)



