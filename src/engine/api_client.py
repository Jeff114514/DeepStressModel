"""
API客户端模块，负责与模型API通信
"""
import time
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, AsyncGenerator
from src.utils.logger import setup_logger
from src.utils.token_counter import token_counter  # 导入token计数器
from src.utils.config import config

logger = setup_logger("api_client")


def _diagnose_disconnect_error(error: Exception, data_received_count: int, time_since_last_data: float, prompt_len: int) -> str:
    """
    诊断连接断开错误的原因
    
    Args:
        error: 异常对象
        data_received_count: 已接收的数据块数量
        time_since_last_data: 距离最后一次收到数据的时间（秒）
        prompt_len: 请求prompt长度
    
    Returns:
        诊断信息字符串
    """
    diagnosis = []
    
    if isinstance(error, aiohttp.ServerDisconnectedError):
        diagnosis.append("🔴 服务器主动断开连接")
        if data_received_count == 0:
            diagnosis.append("  - 未收到任何数据，连接在响应开始前断开")
            diagnosis.append("  - 可能原因：服务器端超时、资源限制、请求格式错误")
        else:
            diagnosis.append(f"  - 已接收 {data_received_count} 个数据块后断开")
            if time_since_last_data < 5:
                diagnosis.append("  - ⚠️ 在正常接收数据时突然断开，可能是数据接收速度问题")
            diagnosis.append("  - 可能原因：服务器处理超时、资源耗尽、数据接收不及时")
        
        if prompt_len > 1000:
            diagnosis.append(f"  - 请求较长（{prompt_len}字符），可能超出服务器处理能力")
    elif isinstance(error, asyncio.TimeoutError):
        diagnosis.append("⏱️ 客户端超时")
        if data_received_count == 0:
            diagnosis.append("  - 未收到任何数据")
            diagnosis.append("  - 可能原因：服务器响应慢、网络延迟高、连接建立失败")
        else:
            diagnosis.append(f"  - 在接收 {data_received_count} 个数据块后超时")
            diagnosis.append(f"  - 距离最后一次数据已过去 {time_since_last_data:.2f} 秒")
            diagnosis.append("  - 可能原因：服务器生成速度慢、网络不稳定")
    else:
        diagnosis.append("🔌 客户端连接错误")
        diagnosis.append(f"  - 错误类型: {type(error).__name__}")
        diagnosis.append("  - 可能原因：网络问题、DNS解析失败、防火墙限制")
    
    return "\n".join(diagnosis)

class StreamStats:
    """流式输出统计"""
    def __init__(self, model_name: str = None):
        self.total_chars = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.last_update_time = time.time()
        self.current_char_speed = 0.0
        self.current_token_speed = 0.0
        self.char_speeds = []
        self.token_speeds = []
        self.model_name = model_name
    
    def update(self, new_text: str):
        """更新统计信息"""
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        
        if time_diff > 0:
            # 计算新增字符数
            new_chars = len(new_text)
            # 使用tiktoken计算token数
            new_tokens = token_counter.count_tokens(new_text, self.model_name)
            
            # 计算字符速度
            self.current_char_speed = new_chars / time_diff
            self.char_speeds.append(self.current_char_speed)
            
            # 计算token速度
            self.current_token_speed = new_tokens / time_diff
            self.token_speeds.append(self.current_token_speed)
            
            # 更新总计数
            self.total_chars += new_chars
            self.total_tokens += new_tokens
            self.total_time += time_diff
        
        self.last_update_time = current_time
    
    @property
    def avg_char_speed(self) -> float:
        """平均字符生成速度（字符/秒）"""
        if self.total_time > 0:
            return self.total_chars / self.total_time
        return 0.0
    
    @property
    def avg_token_speed(self) -> float:
        """平均token生成速度（token/秒）"""
        if self.total_time > 0:
            return self.total_tokens / self.total_time
        return 0.0

class APIResponse:
    """API响应数据类"""
    def __init__(
        self,
        success: bool,
        response_text: str = "",
        error_msg: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration: float = 0.0,
        start_time: float = 0.0,
        end_time: float = 0.0,
        model_name: str = "",
        stream_stats: Optional[StreamStats] = None,
        first_token_latency: Optional[float] = None
    ):
        self.success = success
        self.response_text = response_text
        self.error_msg = error_msg
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        # 兼容旧字段命名
        self.tokens_generated = output_tokens
        self.duration = duration
        self.start_time = start_time
        self.end_time = end_time
        self.model_name = model_name
        self.stream_stats = stream_stats
        # 首字节/首token延迟（秒）
        self.first_token_latency = first_token_latency
    
    @property
    def generation_speed(self) -> float:
        """计算生成速度（字符/秒）"""
        if self.stream_stats:
            # 使用流式统计的平均速度
            return self.stream_stats.avg_char_speed
        elif self.duration > 0 and self.response_text:
            # 如果没有流式统计，使用总字符数除以总时间
            return len(self.response_text) / self.duration
        return 0.0
    
    @property
    def total_chars(self) -> int:
        """获取总字符数"""
        return len(self.response_text) if self.response_text else 0
    
    @property
    def total_tokens(self) -> int:
        """获取总token数"""
        return self.output_tokens

class APIClient:
    """API客户端类"""
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int = 10,  # 添加超时参数
        retry_count: int = 1,  # 添加重试次数参数
        chat_path: str = "/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body_params: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = None
    ):
        # 确保 API URL 格式正确
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/v1"):
            self.api_url += "/v1"
        self.api_key = api_key
        self.model = model
        self.chat_path = chat_path
        
        # 使用传入的超时和重试设置
        self.connect_timeout = timeout
        self.max_retries = retry_count
        
        # 其他参数
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # 模型参数
        self.model_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        if extra_body_params:
            self.model_params.update(extra_body_params)
        # 允许覆盖流模式
        self._force_stream = stream
        # 附加请求头
        self.extra_headers = extra_headers or {}
        
        # 创建异步HTTP会话，禁用连接并发限制
        default_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        default_headers.update(self.extra_headers)
        # 改进连接器配置，增加连接池大小和keepalive时间
        # 对于压力测试，增加连接池大小和keepalive时间，提高连接稳定性
        # 对于长请求和流式响应，需要更长的keepalive超时时间（至少覆盖max_tokens生成时间）
        # 假设平均生成速度20 tokens/s，2048 tokens需要约20秒，加上网络延迟，设置300秒更安全
        keepalive_timeout = max(300, (self.max_tokens // 20) * 2) if self.max_tokens else 300
        connector = aiohttp.TCPConnector(
            limit=0,  # 无限制总连接数
            limit_per_host=0,  # 无限制每个主机连接数
            keepalive_timeout=keepalive_timeout,  # 根据max_tokens动态调整keepalive超时时间
            enable_cleanup_closed=True,  # 启用清理已关闭的连接
            force_close=False,  # 不强制关闭连接，允许重用
            ttl_dns_cache=300,  # DNS缓存TTL
            use_dns_cache=True,  # 启用DNS缓存
        )
        self.session = aiohttp.ClientSession(headers=default_headers, connector=connector)
        logger.info(f"初始化 API 客户端: URL={api_url}, model={model}, connect_timeout={self.connect_timeout}, max_retries={self.max_retries}")
    
    async def _recreate_session(self):
        """重新创建HTTP会话"""
        if self.session and not self.session.closed:
            await self.session.close()
        default_headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        default_headers.update(self.extra_headers)
        # 使用与初始化时相同的keepalive超时逻辑
        keepalive_timeout = max(300, (self.max_tokens // 20) * 2) if self.max_tokens else 300
        connector = aiohttp.TCPConnector(
            limit=0,
            limit_per_host=0,
            keepalive_timeout=keepalive_timeout,
            enable_cleanup_closed=True,
            force_close=False,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        self.session = aiohttp.ClientSession(headers=default_headers, connector=connector)
        logger.info("已重新创建API客户端会话")
    
    async def close(self):
        """关闭客户端会话"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("API客户端会话已关闭")
    
    def _prepare_request(self, prompt: str) -> dict:
        """准备请求数据"""
        # 根据配置决定是否使用流式输出（允许外部强制覆盖）
        use_stream = self._force_stream
        if use_stream is None:
            use_stream = config.get('openai_api.stream_mode', True)
        
        return {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": use_stream,  # 根据配置决定是否启用流式输出
            **self.model_params  # 只包含支持的参数
        }
    
    async def _process_stream(
        self,
        response: aiohttp.ClientResponse
    ) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        last_data_time = time.time()  # 记录最后一次收到数据的时间
        data_received_count = 0  # 记录已接收的数据块数量
        try:
            async for line in response.content:
                # 更新最后接收数据的时间
                last_data_time = time.time()
                data_received_count += 1
                
                # 检查连接状态 - 在读取数据前检查，避免在已关闭的连接上操作
                may_continue = True
                if response.closed:
                    # logger.warning("检测到响应连接已关闭（在读取数据时）")
                    # 如果已经收到一些数据，可能是正常结束；如果没有数据，可能是服务器主动断开
                    if data_received_count == 0:
                        raise aiohttp.ServerDisconnectedError("服务器在响应开始前断开连接")
                    else:
                        # logger.info(f"连接已关闭，但已收到 {data_received_count} 个数据块，可能是正常结束")
                        may_continue = False
                if may_continue:
                    line = line.decode('utf-8').strip()
                    if not line:  # 空行，跳过
                        continue
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            # 检查是否是结束标记
                            if data.get('choices') and data['choices'][0].get('finish_reason'):
                                logger.debug("收到流式响应结束标记")
                                break
                            if data.get('choices'):
                                # 支持两种格式：delta 和 text
                                content = (
                                    data['choices'][0].get('delta', {}).get('content', '') or
                                    data['choices'][0].get('text', '')
                                )
                                if content:
                                    yield content
                        except json.JSONDecodeError as e:
                            logger.debug(f"JSON解析失败，跳过该行: {line[:100]}")
                            continue
                    elif line.startswith(':'):  # SSE注释行，跳过
                        continue
                    else:
                        logger.debug(f"未识别的SSE格式行: {line[:100]}")
                except UnicodeDecodeError as e:
                    logger.warning(f"流式响应解码错误: {e}，跳过该行")
                    continue
        except asyncio.TimeoutError as e:
            # 客户端超时：长时间没有收到数据
            time_since_last_data = time.time() - last_data_time
            logger.error(f"流式响应读取超时: 距离最后一次收到数据已过去 {time_since_last_data:.2f} 秒")
            logger.error(f"已接收数据块数量: {data_received_count}")
            if data_received_count == 0:
                logger.error("客户端超时：未收到任何数据，可能是服务器响应慢或网络问题")
            else:
                logger.error("客户端超时：在接收数据过程中超时，可能是服务器生成速度慢或网络不稳定")
            raise
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError, ConnectionError) as e:
            # 连接断开错误，区分超时和主动断开
            error_type = type(e).__name__
            error_msg = str(e)
            time_since_last_data = time.time() - last_data_time
            
            # 使用诊断函数生成详细的错误信息
            # 注意：这里无法获取prompt_len，所以传入0
            diagnosis = _diagnose_disconnect_error(e, data_received_count, time_since_last_data, 0)
            logger.error(f"流式响应连接断开: {error_type} - {error_msg}")
            logger.error("错误诊断:\n%s", diagnosis)
            
            raise  # 向上传递异常，让generate方法处理重试
        except Exception as e:
            logger.error(f"流式输出处理异常: {type(e).__name__} - {e}")
            logger.error(f"已接收数据块数量: {data_received_count}")
            raise  # 向上传递异常，让generate方法处理
    
    async def generate(self, prompt: str) -> APIResponse:
        """生成响应"""
        # 将本地分词耗时也计入首 token 延迟
        start_time = time.time()
        prompt_tokens = token_counter.count_tokens(prompt, self.model)
        stream_stats = StreamStats(self.model)  # 传入模型名称
        full_response = []
        first_token_latency = None
        
        # 根据配置决定是否使用流式输出
        use_stream = self._force_stream
        if use_stream is None:
            use_stream = config.get('openai_api.stream_mode', True)
        
        for attempt in range(self.max_retries):
            try:
                request_url = f"{self.api_url}{self.chat_path}"
                request_data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": use_stream,  # 根据配置决定是否启用流式输出
                    **self.model_params  # 使用model_params代替直接指定参数
                }
                # 打印完整请求数据以供调试
                logger.info(f"发送请求: URL={request_url}, model={self.model}, stream={request_data.get('stream')}")
                # 安全地打印请求体，避免打印过长的内容
                request_data_log = request_data.copy()
                if 'messages' in request_data_log and request_data_log['messages']:
                    # 限制content长度，避免日志过长
                    content = request_data_log['messages'][0].get('content', '')
                    if len(content) > 200:
                        request_data_log['messages'][0]['content'] = content[:200] + f"... (已截断，总长度: {len(content)})"
                logger.info(f"请求体: {json.dumps(request_data_log, ensure_ascii=False)}")
                
                # 如果 connect_timeout 为 None，则关闭所有超时限制
                if self.connect_timeout is None:
                    timeout_config = None  # 完全关闭超时
                else:
                    # 对于流式响应，需要更长的读取超时时间
                    # 根据max_tokens估算：假设100 tokens/s，2048 tokens需要约20秒，加上缓冲设置为60秒
                    read_timeout = max(60, (self.max_tokens // 100) + 30) if self.max_tokens and use_stream else None
                    timeout_config = aiohttp.ClientTimeout(
                        connect=self.connect_timeout,
                        sock_connect=self.connect_timeout,
                        sock_read=read_timeout,  # 流式响应时设置读取超时
                        total=None  # 不限制总体超时，由sock_read控制
                    )
                async with self.session.post(
                    request_url,  # 允许自定义路径
                    json=request_data,
                    timeout=timeout_config
                ) as response:
                    if response.status == 200:
                        try:
                            # 根据配置决定处理方式
                            if use_stream:
                                # 流式输出处理
                                try:
                                    # 流式输出处理：及时消费数据，避免数据积压
                                    # 在压力测试场景下，及时处理数据非常重要，避免服务器因客户端接收慢而断开连接
                                    async for chunk in self._process_stream(response):
                                        full_response.append(chunk)
                                        stream_stats.update(chunk)
                                        if first_token_latency is None:
                                            first_token_latency = time.time() - start_time
                                        # 注意：异步迭代器会自动让出控制权，不需要额外的sleep
                                        # 但如果压力测试时仍有问题，可能是事件循环过载或网络问题
                                except asyncio.TimeoutError as timeout_error:
                                    # 客户端超时：长时间没有收到数据
                                    error_msg_str = str(timeout_error)
                                    logger.error(f"流式响应读取超时: {error_msg_str}")
                                    if not full_response:
                                        # 没有收到任何响应，抛出异常让重试机制处理
                                        raise
                                    # 有部分响应，返回部分结果但标记为失败
                                    end_time = time.time()
                                    return APIResponse(
                                        success=False,
                                        response_text="".join(full_response),
                                        error_msg=f"客户端读取超时: {error_msg_str}",
                                        input_tokens=prompt_tokens,
                                        output_tokens=stream_stats.total_tokens,
                                        duration=end_time - start_time,
                                        start_time=start_time,
                                        end_time=end_time,
                                        model_name=self.model,
                                        stream_stats=stream_stats,
                                        first_token_latency=first_token_latency
                                    )
                                except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError, ConnectionError) as stream_error:
                                    # 流式响应过程中连接断开，区分服务器主动断开和客户端错误
                                    error_type = type(stream_error).__name__
                                    error_msg_str = str(stream_error)
                                    
                                    # 判断是服务器主动断开还是客户端错误
                                    is_server_disconnect = isinstance(stream_error, aiohttp.ServerDisconnectedError)
                                    
                                    # 计算数据接收统计
                                    data_chunks = len(full_response)
                                    total_chars = len("".join(full_response))
                                    time_elapsed = time.time() - start_time
                                    
                                    # 使用诊断函数
                                    diagnosis = _diagnose_disconnect_error(stream_error, data_chunks, 0, len(prompt))
                                    logger.error(f"流式响应过程中连接断开: {error_type} - {error_msg_str}")
                                    logger.error("错误诊断:\n%s", diagnosis)
                                    logger.error(f"数据接收统计: {data_chunks} 个数据块, {total_chars} 字符, 耗时 {time_elapsed:.2f} 秒")
                                    
                                    # 如果有部分响应，返回部分结果，否则抛出异常让重试机制处理
                                    if not full_response:
                                        raise  # 没有收到任何响应，抛出异常让重试机制处理
                                    # 有部分响应，返回部分结果但标记为失败
                                    end_time = time.time()
                                    disconnect_type = "服务器主动断开" if is_server_disconnect else "客户端连接错误"
                                    return APIResponse(
                                        success=False,
                                        response_text="".join(full_response),
                                        error_msg=f"流式响应中断 ({disconnect_type}): {error_type} - {error_msg_str}",
                                        input_tokens=prompt_tokens,
                                        output_tokens=stream_stats.total_tokens,
                                        duration=end_time - start_time,
                                        start_time=start_time,
                                        end_time=end_time,
                                        model_name=self.model,
                                        stream_stats=stream_stats,
                                        first_token_latency=first_token_latency
                                    )
                                
                                end_time = time.time()
                                return APIResponse(
                                    success=True,
                                    response_text="".join(full_response),
                                    input_tokens=prompt_tokens,
                                    output_tokens=stream_stats.total_tokens,
                                    duration=end_time - start_time,
                                    start_time=start_time,
                                    end_time=end_time,
                                    model_name=self.model,
                                    stream_stats=stream_stats,
                                    first_token_latency=first_token_latency
                                )
                            else:
                                # 非流式输出处理
                                data = await response.json()
                                response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                                
                                # 估算token数量
                                tokens_generated = token_counter.count_tokens(response_text, self.model)
                                
                                # 更新流统计（虽然不是流式但仍需要计算速度）
                                stream_stats.update(response_text)
                                end_time = time.time()
                                first_token_latency = end_time - start_time
                                return APIResponse(
                                    success=True,
                                    response_text=response_text,
                                    input_tokens=prompt_tokens,
                                    output_tokens=tokens_generated,
                                    duration=end_time - start_time,
                                    start_time=start_time,
                                    end_time=end_time,
                                    model_name=self.model,
                                    stream_stats=stream_stats,
                                    first_token_latency=first_token_latency
                                )
                        except Exception as e:
                            logger.error(f"流式输出中断: {e}")
                            # 返回已生成的部分内容，但标记为失败
                            end_time = time.time()
                            response_text = ''.join(full_response)
                            return APIResponse(
                                success=False,
                                response_text=response_text,
                                error_msg=f"流式输出中断: {str(e)}",
                                input_tokens=prompt_tokens,
                                output_tokens=stream_stats.total_tokens,
                                duration=end_time - start_time,
                                start_time=start_time,
                                end_time=end_time,
                                model_name=self.model,
                                stream_stats=stream_stats
                            )
                    else:
                        error_text = await response.text()
                        logger.error(f"API请求失败 (尝试 {attempt + 1}/{self.max_retries}): {response.status} - {error_text}")
                        logger.error(f"请求URL: {self.api_url}{self.chat_path}, 模型: {self.model}")
                        if attempt == self.max_retries - 1:
                            return APIResponse(
                                success=False,
                                error_msg=f"HTTP {response.status}: {error_text}",
                                input_tokens=prompt_tokens,
                                duration=time.time() - start_time,
                                start_time=start_time,
                                end_time=time.time()
                            )
            
            except asyncio.TimeoutError as e:
                error_msg = "连接超时" if "connect" in str(e) else "请求超时"
                logger.error(f"API{error_msg} (尝试 {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    return APIResponse(
                        success=False,
                        error_msg=error_msg,
                        input_tokens=prompt_tokens,
                        duration=time.time() - start_time,
                        start_time=start_time,
                        end_time=time.time()
                    )
            
            except asyncio.TimeoutError as e:
                # 客户端超时：连接或读取超时
                error_msg_str = str(e)
                logger.error("API连接超时 (尝试 %d/%d): %s", attempt + 1, self.max_retries, error_msg_str)
                logger.error("请求URL: %s%s, 模型: %s", self.api_url, self.chat_path, self.model)
                logger.error("客户端超时：可能是连接建立慢或服务器响应慢")
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.info("等待 %d 秒后重试...", wait_time)
                    await asyncio.sleep(wait_time)
                    continue
                return APIResponse(
                    success=False,
                    error_msg=f"客户端超时: {error_msg_str}",
                    input_tokens=prompt_tokens,
                    duration=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time()
                )
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError, ConnectionError, OSError) as e:
                error_type = type(e).__name__
                error_msg_str = str(e)
                logger.error("API连接错误 (尝试 %d/%d): %s - %s", attempt + 1, self.max_retries, error_type, error_msg_str)
                logger.error("请求URL: %s%s, 模型: %s", self.api_url, self.chat_path, self.model)
                
                # 区分服务器主动断开和客户端连接错误
                is_server_disconnect = isinstance(e, aiohttp.ServerDisconnectedError)
                
                if is_server_disconnect:
                    logger.error("⚠️ 服务器主动断开连接")
                    prompt_len = len(prompt) if prompt else 0
                    logger.error("可能原因：")
                    logger.error("  1) 服务器端超时（请求处理时间过长）")
                    logger.error("  2) 服务器资源限制（内存/GPU不足）")
                    logger.error("  3) 服务器连接数限制（并发过高）")
                    if prompt_len > 1000:
                        logger.error("  4) 请求过长（%d字符），服务器可能无法处理", prompt_len)
                    logger.error("  5) 数据接收速度问题（压力测试时数据接收不及时）")
                else:
                    logger.error("客户端连接错误")
                    logger.error("可能原因：网络问题、DNS解析失败、防火墙限制等")
                
                # 对于连接错误，增加重试等待时间，并尝试重新创建会话
                if attempt < self.max_retries - 1:
                    # 根据错误类型和请求长度调整等待时间
                    if is_server_disconnect:
                        prompt_len = len(prompt) if prompt else 0
                        # 服务器断开时，等待时间更长，特别是长请求
                        base_wait = 3 if prompt_len > 1000 else 2
                    else:
                        base_wait = 1
                    wait_time = base_wait * (attempt + 1)  # 递增等待时间
                    logger.info("等待 %d 秒后重试...", wait_time)
                    await asyncio.sleep(wait_time)
                    # 如果会话已关闭，尝试重新创建
                    if self.session.closed:
                        logger.warning("检测到会话已关闭，尝试重新创建会话")
                        await self._recreate_session()
                    continue  # 继续重试循环
                # 最后一次尝试失败
                disconnect_type = "服务器主动断开" if is_server_disconnect else "客户端连接错误"
                return APIResponse(
                    success=False,
                    error_msg=f"{disconnect_type} ({error_type}): {error_msg_str}",
                    input_tokens=prompt_tokens,
                    duration=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time()
                )
            
            except Exception as e:
                error_type = type(e).__name__
                error_msg_str = str(e)
                logger.error("API请求异常 (尝试 %d/%d): %s - %s", attempt + 1, self.max_retries, error_type, error_msg_str)
                logger.error("请求URL: %s%s, 模型: %s", self.api_url, self.chat_path, self.model)
                if attempt < self.max_retries - 1:
                    wait_time = 1 * (attempt + 1)  # 通用异常等待时间：1s, 2s, 3s...
                    logger.info("等待 %d 秒后重试...", wait_time)
                    await asyncio.sleep(wait_time)
                    continue  # 继续重试循环
                # 最后一次尝试失败
                return APIResponse(
                    success=False,
                    error_msg=f"{error_type}: {error_msg_str}",
                    input_tokens=prompt_tokens,
                    duration=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time()
                )
        
        return APIResponse(
            success=False,
            error_msg="未知错误",
            input_tokens=prompt_tokens,
            duration=time.time() - start_time,
            start_time=start_time,
            end_time=time.time()
        )
