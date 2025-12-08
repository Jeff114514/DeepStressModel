"""
TGI (Text Generation Inference) 的 OpenAI 兼容后端
"""
from src.engine.backends.openai_backend import BaseOpenAIBackend, BackendOptions


class TGIBackend(BaseOpenAIBackend):
    backend_name = "tgi"
    # TGI OpenAI兼容网关默认路径
    default_chat_path = "/chat/completions"

    def __init__(self, options: BackendOptions):
        # TGI 通常支持bf16，保持precision透传
        if options.precision is None:
            options.precision = options.extra_body_params.get("precision")
        super().__init__(options)



