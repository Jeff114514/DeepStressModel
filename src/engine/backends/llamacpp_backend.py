"""
Llama.cpp 的 OpenAI 兼容后端
直接使用 vLLM 相同的后端实现，仅保留名称差异
"""
from src.engine.backends.vllm_backend import VLLMBackend


class LlamaCppBackend(VLLMBackend):
    """Llama.cpp 后端，使用与 vLLM 相同的实现"""
    backend_name = "llama.cpp"





