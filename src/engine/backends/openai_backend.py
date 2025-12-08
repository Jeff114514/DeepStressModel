"""
OpenAI兼容后端基类与默认实现
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from src.engine.api_client import APIClient


@dataclass
class BackendOptions:
    """后端初始化配置"""
    api_url: str
    api_key: str = ""
    model: str = ""
    chat_path: str = "/chat/completions"
    precision: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body_params: Dict[str, Any] = field(default_factory=dict)
    stream: Optional[bool] = None


class BaseOpenAIBackend:
    """
    OpenAI兼容接口的通用后端
    - 子类可覆盖chat_path/extra_headers/extra_body_params等
    """
    backend_name: str = "openai"
    default_chat_path: str = "/chat/completions"
    supports_stream: bool = True

    def __init__(self, options: BackendOptions):
        self.options = options
        # 确保chat_path存在
        if not self.options.chat_path:
            self.options.chat_path = self.default_chat_path

    @classmethod
    def from_model_config(cls, model_config: Dict[str, Any]) -> "BaseOpenAIBackend":
        """
        根据model_config构建后端实例
        预期字段：
          - api_url: 后端服务基地址
          - api_key: 认证key，可为空
          - model: 模型名称
          - chat_path: 自定义chat/completions路径（可选）
          - precision: 精度，bf16/gguf/等（可选）
          - extra_headers: 额外HTTP头（可选）
          - extra_body_params: 额外请求体参数（可选）
          - stream: 是否强制流式（可选）
        """
        return cls(
            BackendOptions(
                api_url=model_config.get("api_url", ""),
                api_key=model_config.get("api_key", ""),
                model=model_config.get("model", ""),
                chat_path=model_config.get("chat_path", cls.default_chat_path),
                precision=model_config.get("precision"),
                extra_headers=model_config.get("extra_headers", {}) or {},
                extra_body_params=model_config.get("extra_body_params", {}) or {},
                stream=model_config.get("stream"),
            )
        )

    def create_client(
        self,
        timeout: Optional[int] = None,
        retry_count: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> APIClient:
        """创建对应的APIClient"""
        return APIClient(
            api_url=self.options.api_url,
            api_key=self.options.api_key,
            model=self.options.model,
            timeout=timeout or 10,
            retry_count=retry_count or 1,
            chat_path=self.options.chat_path,
            extra_headers=self.options.extra_headers,
            extra_body_params=self.options.extra_body_params,
            precision=self.options.precision,
            stream=self.options.stream,
            max_tokens=max_tokens or 2048,
            temperature=temperature or 0.7,
            top_p=top_p or 0.9,
        )


class OpenAIBackend(BaseOpenAIBackend):
    """默认OpenAI兼容实现"""
    backend_name = "openai"