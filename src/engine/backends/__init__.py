"""
后端适配器注册与导出
"""
from typing import Dict, Type
from src.engine.backends.openai_backend import (
    BackendOptions,
    BaseOpenAIBackend,
    OpenAIBackend,
)
from src.engine.backends.vllm_backend import VLLMBackend
from src.engine.backends.llamacpp_backend import LlamaCppBackend
from src.engine.backends.sglang_backend import SGLangBackend
from src.engine.backends.tgi_backend import TGIBackend

# 后端注册表
_BACKENDS: Dict[str, Type[BaseOpenAIBackend]] = {
    "openai": OpenAIBackend,
    "vllm": VLLMBackend,
    "llama.cpp": LlamaCppBackend,
    "llamacpp": LlamaCppBackend,
    "sglang": SGLangBackend,
    "tgi": TGIBackend,
}


def register_backend(name: str, backend_cls: Type[BaseOpenAIBackend]):
    """注册新的后端类"""
    _BACKENDS[name.lower()] = backend_cls


def get_backend_class(name: str) -> Type[BaseOpenAIBackend]:
    """根据名称获取后端类，默认返回OpenAIBackend"""
    return _BACKENDS.get((name or "openai").lower(), OpenAIBackend)


# 按需导出注册器与基础类型
__all__ = [
    "BackendOptions",
    "BaseOpenAIBackend",
    "OpenAIBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    "SGLangBackend",
    "TGIBackend",
    "register_backend",
    "get_backend_class",
]




