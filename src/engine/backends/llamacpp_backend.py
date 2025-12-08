"""
Llama.cpp 的 OpenAI 兼容后端
"""
from src.engine.backends.openai_backend import BaseOpenAIBackend, BackendOptions


class LlamaCppBackend(BaseOpenAIBackend):
    backend_name = "llama.cpp"
    # llama.cpp OpenAI server 默认路径
    default_chat_path = "/chat/completions"

    def __init__(self, options: BackendOptions):
        # llama.cpp 多数使用本地GGUF，默认为非流式
        if options.precision is None:
            options.precision = options.extra_body_params.get("precision", "gguf")
        super().__init__(options)



