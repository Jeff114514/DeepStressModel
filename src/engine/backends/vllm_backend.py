"""
vLLM 的 OpenAI 兼容后端
"""
from src.engine.backends.openai_backend import BaseOpenAIBackend, BackendOptions
from src.utils.logger import setup_logger

logger = setup_logger("vllm_backend")


class VLLMBackend(BaseOpenAIBackend):
    backend_name = "vllm"
    # vLLM 默认兼容 /v1/chat/completions
    default_chat_path = "/chat/completions"

    def __init__(self, options: BackendOptions):
        # vLLM 通常建议使用流式
        if options.stream is None:
            options.stream = True
        super().__init__(options)
        logger.info("初始化 VLLMBackend: api_url=%s, model=%s, chat_path=%s", 
                   options.api_url, options.model, options.chat_path)





