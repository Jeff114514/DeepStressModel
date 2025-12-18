"""
OpenAI 兼容多后端压测工具
"""
import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import psutil

from src.data.dataset_manager import dataset_manager
from src.engine.backends import BaseOpenAIBackend, get_backend_class
from src.engine.api_client import APIClient, APIResponse
from src.monitor.gpu_monitor import gpu_monitor
from src.utils.config import config
from src.utils.logger import setup_logger
from src.utils.token_counter import token_counter

matplotlib.use("Agg")
logger = setup_logger("load_tester")


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)


def _normalize_backend_key(name: str) -> str:
    name = (name or "").lower()
    if name in ("llama.cpp", "llamacpp"):
        return "llamacpp"
    return name


def _default_result_dir() -> str:
    """获取压测结果输出目录，优先使用环境变量，可落在工作目录便于挂载。"""
    base = os.environ.get("DEEPSTRESS_RESULT_DIR")
    if not base:
        base = os.path.join(os.getcwd(), "benchmark_results")
    return os.path.abspath(base)


class GPUSampler:
    """简单的GPU采样器，异步拉取 GPU 利用率与显存使用"""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._error_logged = False  # 标记是否已经记录过错误

    async def _sample_loop(self):
        while self._running:
            try:
                stats = await asyncio.to_thread(gpu_monitor.get_stats)
                if stats and stats.gpus:
                    self.samples.append(
                        {
                            "timestamp": time.time(),
                            "gpu_util": stats.gpu_util,
                            "gpu_memory_used": stats.memory_used,
                            "gpu_memory_total": stats.memory_total,
                        }
                    )
                    # 如果之前有错误但现在成功了，重置错误标记
                    if self._error_logged:
                        self._error_logged = False
            except Exception as exc:  # noqa: BLE001
                # 只在第一次错误时记录日志
                if not self._error_logged:
                    logger.warning("GPU 采样失败: %s (后续错误将静默处理)", exc)
                    self._error_logged = True
            await asyncio.sleep(self.interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._task:
            await self._task

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {}
        util_list = [s["gpu_util"] for s in self.samples]
        mem_used = [s["gpu_memory_used"] for s in self.samples]
        mem_total = [s["gpu_memory_total"] for s in self.samples if s["gpu_memory_total"]]
        return {
            "gpu_util_avg": sum(util_list) / len(util_list),
            "gpu_util_max": max(util_list),
            "memory_used_avg": sum(mem_used) / len(mem_used),
            "memory_used_max": max(mem_used),
            "memory_total": max(mem_total) if mem_total else 0,
        }


class CPUSampler:
    """CPU/内存采样器"""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _sample_loop(self):
        while self._running:
            try:
                cpu = await asyncio.to_thread(psutil.cpu_percent, None)
                mem = await asyncio.to_thread(psutil.virtual_memory)
                self.samples.append(
                    {
                        "timestamp": time.time(),
                        "cpu_percent": cpu,
                        "mem_percent": mem.percent,
                        "mem_used": mem.used,
                        "mem_total": mem.total,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"CPU 采样失败: {exc}")
            await asyncio.sleep(self.interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._task:
            await self._task

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {}
        cpu_list = [s["cpu_percent"] for s in self.samples]
        mem_list = [s["mem_percent"] for s in self.samples]
        mem_used = [s["mem_used"] for s in self.samples]
        mem_total = [s["mem_total"] for s in self.samples if s["mem_total"]]
        return {
            "cpu_avg": sum(cpu_list) / len(cpu_list),
            "cpu_max": max(cpu_list),
            "mem_percent_avg": sum(mem_list) / len(mem_list),
            "mem_percent_max": max(mem_list),
            "mem_used_avg": sum(mem_used) / len(mem_used),
            "mem_used_max": max(mem_used),
            "mem_total": max(mem_total) if mem_total else 0,
        }


class LoadTester:
    """多后端 OpenAI 兼容压测器"""

    def __init__(
        self,
        backend_name: str,
        concurrency: int = 4,
        qps: Optional[float] = None,
        duration_seconds: Optional[int] = None,
        total_requests: Optional[int] = None,
        dataset_name: Optional[str] = None,
        max_new_tokens: int = 2048,
        stream: Optional[bool] = None,
        timeout: Optional[int] = None,
        retry_count: Optional[int] = None,
        prompts_override: Optional[List[str]] = None,
    ):
        self.backend_name = backend_name
        self.backend_key = _normalize_backend_key(backend_name)
        self.concurrency = max(1, concurrency)
        self.qps = qps
        self.interval = 1 / qps if qps else None
        self.duration_seconds = duration_seconds
        self.total_requests = total_requests
        self.dataset_name = dataset_name
        self.max_new_tokens = max_new_tokens
        self.stream = stream
        self.timeout = timeout
        self.retry_count = retry_count
        self.prompts_override = prompts_override
        self.use_local_dataset = bool(prompts_override is None and (dataset_name or dataset_manager.get_offline_dataset_data()))

        self.backend_config = self._build_backend_config()
        self.backend = self._create_backend()
        self.client: Optional[APIClient] = None
        self.gpu_samples: List[Dict[str, Any]] = []
        self.cpu_samples: List[Dict[str, Any]] = []

        self._rate_lock = asyncio.Lock()
        self._next_slot = time.perf_counter()
        self.results: List[Dict[str, Any]] = []
        self.result_dir = _default_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)

    def _build_backend_config(self) -> Dict[str, Any]:
        defaults = config.get("openai_benchmarks.defaults", {}) or {}
        backend_cfg = config.get(f"openai_benchmarks.{self.backend_key}", {}) or {}

        merged = {**defaults, **backend_cfg}
        merged["extra_body_params"] = {
            **defaults.get("extra_body_params", {}),
            **backend_cfg.get("extra_body_params", {}),
        }
        merged["extra_headers"] = {
            **defaults.get("extra_headers", {}),
            **backend_cfg.get("extra_headers", {}),
        }
        if self.stream is not None:
            merged["stream"] = self.stream
        return merged

    def _create_backend(self):
        backend_cls = get_backend_class(self.backend_name)
        logger.info("创建后端: backend_name=%s, backend_key=%s, backend_class=%s", 
                   self.backend_name, self.backend_key, backend_cls.__name__)
        logger.info("后端配置: api_url=%s, model=%s, chat_path=%s", 
                   self.backend_config.get("api_url"), 
                   self.backend_config.get("model"),
                   self.backend_config.get("chat_path"))
        backend_or_options = backend_cls.from_model_config(self.backend_config)
        # from_model_config 目前返回的就是实例，需兼容旧逻辑
        if isinstance(backend_or_options, BaseOpenAIBackend):
            logger.info("后端实例创建成功: %s", type(backend_or_options).__name__)
            return backend_or_options
        logger.info("使用BackendOptions创建后端实例")
        return backend_cls(backend_or_options)

    def _prepare_prompts(self) -> List[str]:
        prompts: List[str] = []

        if self.prompts_override:
            return list(self.prompts_override)

        raw_dataset = dataset_manager.get_offline_dataset_data()
        if raw_dataset:
            for item in raw_dataset:
                text = self._extract_text(item)
                if text:
                    prompts.append(text)

        if not prompts:
            if self.dataset_name:
                prompts.extend(dataset_manager.get_dataset(self.dataset_name))
            else:
                for _, items in dataset_manager.get_all_datasets().items():
                    prompts.extend(items)

        if not prompts:
            raise ValueError("未找到可用的测试提示，请先加载或配置数据集")

        # 数据集模式：按 total_requests 采样，避免超大数据集全量加载
        if self.use_local_dataset and self.total_requests:
            if len(prompts) >= self.total_requests:
                prompts = random.sample(prompts, self.total_requests)
            else:
                prompts = random.choices(prompts, k=self.total_requests)
        else:
            random.shuffle(prompts)

        return prompts

    @staticmethod
    def _extract_text(item: Any) -> Optional[str]:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("input", "prompt", "question", "instruction"):
                if item.get(key):
                    return str(item[key])
            # 兼容 messages 结构
            if "messages" in item and isinstance(item["messages"], list):
                contents = [msg.get("content", "") for msg in item["messages"] if isinstance(msg, dict)]
                return "\n".join([c for c in contents if c])
        return None

    async def _rate_limit(self):
        if not self.interval:
            return
        async with self._rate_lock:
            now = time.perf_counter()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_slot = max(self._next_slot, now) + self.interval

    async def _single_request(self, idx: int, prompt: str):
        await self._rate_limit()
        start = time.perf_counter()
        response: APIResponse = await self.client.generate(prompt)
        end = time.perf_counter()

        total_tokens = (response.input_tokens or 0) + (response.output_tokens or 0)

        record = {
            "id": idx,
            "prompt": prompt[:200],
            "success": response.success,
            "duration": response.duration or (end - start),
            "first_token_latency": response.first_token_latency,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "tokens": total_tokens,
            "generation_speed": response.generation_speed,
            "error": response.error_msg,
            "start_ts": start,
            "end_ts": end,
        }
        self.results.append(record)

    async def _producer(self, queue: asyncio.Queue, prompts: List[str], start_time: float):
        idx = 0
        while True:
            if self.duration_seconds and (time.perf_counter() - start_time) >= self.duration_seconds:
                break
            if self.total_requests and idx >= self.total_requests:
                break

            prompt = prompts[idx % len(prompts)]
            await queue.put((idx, prompt))
            idx += 1
        for _ in range(self.concurrency):
            await queue.put((None, None))

    async def _worker(self, queue: asyncio.Queue):
        while True:
            idx, prompt = await queue.get()
            if idx is None:
                queue.task_done()
                break
            try:
                await self._single_request(idx, prompt)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"请求 {idx} 失败: {exc}")
            finally:
                queue.task_done()

    def _aggregate(
        self,
        start_ts: float,
        end_ts: float,
        gpu_summary: Dict[str, Any],
        cpu_summary: Dict[str, Any],
        start_wall: Optional[float] = None,
        end_wall: Optional[float] = None,
    ) -> Dict[str, Any]:
        durations = [r["duration"] for r in self.results if r.get("duration") is not None]
        first_tokens = [r["first_token_latency"] for r in self.results if r.get("first_token_latency")]
        input_tokens = [r.get("input_tokens", 0) for r in self.results]
        output_tokens = [r.get("output_tokens", 0) for r in self.results]
        wall = end_ts - start_ts
        wall = wall if wall > 0 else 1e-6

        total_input_tokens = sum(input_tokens)
        total_output_tokens = sum(output_tokens)
        total_all_tokens = total_input_tokens + total_output_tokens
        throughput_tokens_per_s = (total_all_tokens / wall) if total_all_tokens else 0.0
        throughput_tokens_per_s_per_conc = (
            throughput_tokens_per_s / self.concurrency if self.concurrency else 0.0
        )

        summary = {
            "backend": self.backend_name,
            "model": self.backend_config.get("model"),
            "concurrency": self.concurrency,
            "qps_target": self.qps,
            "requests": len(self.results),
            "success_rate": sum(1 for r in self.results if r["success"]) / len(self.results) if self.results else 0,
            "wall_time_sec": wall,
            "start_perf": start_ts,
            "end_perf": end_ts,
            "start_time": datetime.fromtimestamp(start_wall).isoformat() if start_wall else None,
            "end_time": datetime.fromtimestamp(end_wall).isoformat() if end_wall else None,
            "latency_avg": statistics.mean(durations) if durations else 0.0,
            "latency_p50": _percentile(durations, 50),
            "latency_p95": _percentile(durations, 95),
            "latency_p99": _percentile(durations, 99),
            "first_token_avg": statistics.mean(first_tokens) if first_tokens else 0.0,
            "first_token_p95": _percentile(first_tokens, 95) if first_tokens else 0.0,
            "input_tokens_total": total_input_tokens,
            "output_tokens_total": total_output_tokens,
            "throughput_input_tokens_per_s": (total_input_tokens / wall) if total_input_tokens else 0.0,
            "throughput_output_tokens_per_s": (total_output_tokens / wall) if total_output_tokens else 0.0,
            "throughput_tokens_per_s": throughput_tokens_per_s,
            "throughput_tokens_per_s_per_concurrency": throughput_tokens_per_s_per_conc,
            "qps_observed": (len(self.results) / wall) if self.results else 0.0,
            "first_token_p50": _percentile(first_tokens, 50) if first_tokens else 0.0,
            "gpu": gpu_summary,
            "cpu": cpu_summary,
        }
        return summary

    def _save_result(self, summary: Dict[str, Any], start_ts: float, end_ts: float, start_wall: float, end_wall: float):
        result = {
            "backend": self.backend_name,
            "backend_config": self.backend_config,
            "start_time": datetime.fromtimestamp(start_wall).isoformat(),
            "end_time": datetime.fromtimestamp(end_wall).isoformat(),
            "summary": summary,
            "results": self.results,
            "gpu_samples": self.gpu_samples,
            "cpu_samples": self.cpu_samples,
        }
        filename = f"loadtest_{self.backend_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        filepath = os.path.join(self.result_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"压测结果已保存: {filepath}")
        self._export_csv(filepath, summary)
        self._export_gpu_csv(filepath)
        self._export_charts(filepath)
        return filepath

    def _export_csv(self, json_path: str, summary: Dict[str, Any]) -> Optional[str]:
        csv_path = json_path.replace(".json", ".csv")
        fieldnames = [
            "id",
            "success",
            "duration",
            "first_token_latency",
            "input_tokens",
            "output_tokens",
            "tokens",
            "generation_speed",
            "throughput_tokens_per_s_per_concurrency",
            "error",
            "start_ts",
            "end_ts",
        ]
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.results:
                    writer.writerow({k: r.get(k) for k in fieldnames})
                writer.writerow(
                    {
                        "id": "summary",
                        "success": summary.get("success_rate"),
                        "duration": summary.get("wall_time_sec"),
                        "first_token_latency": summary.get("first_token_avg"),
                        "input_tokens": summary.get("input_tokens_total"),
                        "output_tokens": summary.get("output_tokens_total"),
                        "tokens": summary.get("input_tokens_total", 0) + summary.get("output_tokens_total", 0),
                        "generation_speed": summary.get("throughput_tokens_per_s"),
                        "error": "",
                        "start_ts": summary.get("start_time"),
                        "end_ts": summary.get("end_time"),
                    }
                )
            logger.info(f"CSV 已生成: {csv_path}")
            return csv_path
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"生成 CSV 失败: {exc}")
            return None

    def _export_gpu_csv(self, json_path: str) -> Optional[str]:
        if not self.gpu_samples and not self.cpu_samples:
            return None
        gpu_csv = json_path.replace(".json", "_gpu.csv")
        fieldnames = [
            "timestamp",
            "gpu_util",
            "gpu_memory_used",
            "gpu_memory_total",
            "cpu_percent",
            "mem_percent",
        ]
        try:
            with open(gpu_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for s in self.gpu_samples:
                    writer.writerow(
                        {
                            "timestamp": s.get("timestamp"),
                            "gpu_util": s.get("gpu_util"),
                            "gpu_memory_used": s.get("gpu_memory_used"),
                            "gpu_memory_total": s.get("gpu_memory_total"),
                            "cpu_percent": "",
                            "mem_percent": "",
                        }
                    )
                for s in self.cpu_samples:
                    writer.writerow(
                        {
                            "timestamp": s.get("timestamp"),
                            "gpu_util": "",
                            "gpu_memory_used": "",
                            "gpu_memory_total": "",
                            "cpu_percent": s.get("cpu_percent"),
                            "mem_percent": s.get("mem_percent"),
                        }
                    )
            logger.info(f"GPU/CPU 采样 CSV 已生成: {gpu_csv}")
            return gpu_csv
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"生成 GPU/CPU CSV 失败: {exc}")
            return None

    def _export_charts(self, json_path: str) -> Optional[List[str]]:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"导入 matplotlib 失败，跳过图表导出: {exc}")
            return None

        chart_paths: List[str] = []
        base_path = json_path.rsplit(".", 1)[0]

        try:
            durations = [r["duration"] for r in self.results if r.get("duration") is not None]
            if durations:
                plt.figure()
                plt.hist(durations, bins=20, color="steelblue")
                plt.xlabel("Latency (s)")
                plt.ylabel("Count")
                plt.title("Latency Distribution")
                latency_path = f"{base_path}_latency_hist.png"
                plt.savefig(latency_path)
                plt.close()
                chart_paths.append(latency_path)

            throughput = []
            for r in self.results:
                dur = r.get("duration")
                tokens = r.get("output_tokens", 0)
                if dur and dur > 0:
                    throughput.append(tokens / dur)
                else:
                    throughput.append(0)
            if throughput:
                plt.figure()
                plt.plot(throughput, label="output tokens/s")
                plt.xlabel("Request Index")
                plt.ylabel("Tokens/s")
                plt.title("Per-request Throughput")
                plt.legend()
                throughput_path = f"{base_path}_throughput.png"
                plt.savefig(throughput_path)
                plt.close()
                chart_paths.append(throughput_path)

            # QPS 随时间
            if self.results:
                start_ref = min(r.get("start_ts", 0) for r in self.results if r.get("start_ts") is not None)
                buckets: Dict[int, int] = {}
                for r in self.results:
                    if r.get("start_ts") is None:
                        continue
                    sec = int(r["start_ts"] - start_ref)
                    buckets[sec] = buckets.get(sec, 0) + 1
                qps_series = sorted(buckets.items())
                if qps_series:
                    plt.figure()
                    plt.plot([t for t, _ in qps_series], [v for _, v in qps_series], marker="o")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Requests per second")
                    plt.title("QPS Over Time")
                    qps_path = f"{base_path}_qps.png"
                    plt.savefig(qps_path)
                    plt.close()
                    chart_paths.append(qps_path)

            # 首字节响应分布
            fts = [r["first_token_latency"] for r in self.results if r.get("first_token_latency")]
            if fts:
                plt.figure()
                plt.hist(fts, bins=20, color="darkorange")
                plt.xlabel("First Token Latency (s)")
                plt.ylabel("Count")
                plt.title("First Token Latency Distribution")
                ft_path = f"{base_path}_first_token.png"
                plt.savefig(ft_path)
                plt.close()
                chart_paths.append(ft_path)

            # GPU 利用率/显存占用随时间
            if self.gpu_samples:
                times = [s["timestamp"] - self.gpu_samples[0]["timestamp"] for s in self.gpu_samples]
                utils = [s["gpu_util"] for s in self.gpu_samples]
                mem_used = [s["gpu_memory_used"] for s in self.gpu_samples]
                plt.figure()
                plt.plot(times, utils, label="GPU Util (%)")
                plt.xlabel("Time (s)")
                plt.ylabel("Utilization %")
                plt.title("GPU Utilization Over Time")
                plt.legend()
                gpu_util_path = f"{base_path}_gpu_util.png"
                plt.savefig(gpu_util_path)
                plt.close()
                chart_paths.append(gpu_util_path)

                plt.figure()
                plt.plot(times, mem_used, label="GPU Memory Used (MB)", color="green")
                plt.xlabel("Time (s)")
                plt.ylabel("Memory (MB)")
                plt.title("GPU Memory Over Time")
                plt.legend()
                gpu_mem_path = f"{base_path}_gpu_mem.png"
                plt.savefig(gpu_mem_path)
                plt.close()
                chart_paths.append(gpu_mem_path)

            for p in chart_paths:
                logger.info(f"图表已生成: {p}")
            return chart_paths
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"生成图表失败: {exc}")
            return None

    async def run(self, save_result: bool = True) -> Tuple[Dict[str, Any], Optional[str]]:
        # 在后台线程中准备 prompts，避免阻塞事件循环
        prompts = await asyncio.to_thread(self._prepare_prompts)
        queue: asyncio.Queue = asyncio.Queue()

        # 对于 llamacpp 后端，使用 369 秒超时
        timeout = self.timeout or config.get("test.timeout", 60)
        if self.backend_name.lower() in ("llama.cpp", "llamacpp"):
            timeout = 369

        self.client = self.backend.create_client(
            timeout=timeout,
            retry_count=self.retry_count or config.get("test.retry_count", 1),
            max_tokens=self.max_new_tokens,
        )

        gpu_interval = config.get("load_test.gpu_sample_interval", 2.0)
        gpu_sampler = GPUSampler(interval=gpu_interval)
        cpu_interval = config.get("load_test.cpu_sample_interval", 2.0)
        cpu_sampler = CPUSampler(interval=cpu_interval)

        start_wall = time.time()
        start_ts = time.perf_counter()
        gpu_sampler.start()
        cpu_sampler.start()

        producer = asyncio.create_task(self._producer(queue, prompts, start_ts))
        workers = [asyncio.create_task(self._worker(queue)) for _ in range(self.concurrency)]

        await producer
        await queue.join()
        for w in workers:
            await w

        await gpu_sampler.stop()
        await cpu_sampler.stop()
        if self.client:
            await self.client.close()

        end_ts = time.perf_counter()
        end_wall = time.time()
        self.gpu_samples = gpu_sampler.samples
        self.cpu_samples = cpu_sampler.samples

        summary = self._aggregate(
            start_ts,
            end_ts,
            gpu_sampler.summary(),
            cpu_sampler.summary(),
            start_wall,
            end_wall,
        )
        path = None
        if save_result:
            path = self._save_result(summary, start_ts, end_ts, start_wall, end_wall)
        return summary, path


def parse_args() -> argparse.Namespace:
    defaults = config.get("load_test", {}) or {}
    parser = argparse.ArgumentParser(description="多后端 OpenAI 兼容压测")
    parser.add_argument("--backend", default="vllm", help="后端名称: vllm/llamacpp/sglang/tgi/openai")
    parser.add_argument("--concurrency", type=int, default=defaults.get("concurrency", 4))
    parser.add_argument("--qps", type=float, default=defaults.get("qps"))
    parser.add_argument("--duration", type=int, default=defaults.get("duration_seconds"))
    parser.add_argument("--requests", type=int, default=defaults.get("total_requests"))
    parser.add_argument("--dataset", default=defaults.get("default_dataset"), help="数据集名称，留空则使用全部")
    parser.add_argument("--max_new_tokens", type=int, default=defaults.get("max_new_tokens", 2048))
    parser.add_argument("--stream", action="store_true", help="强制开启流式")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="强制关闭流式")
    parser.add_argument("--timeout", type=int, default=config.get("test.timeout", 60))
    parser.add_argument("--retry", type=int, default=config.get("test.retry_count", 1))
    parser.set_defaults(stream=defaults.get("stream", True))
    return parser.parse_args()


async def main_async(args: argparse.Namespace):
    tester = LoadTester(
        backend_name=args.backend,
        concurrency=args.concurrency,
        qps=args.qps,
        duration_seconds=args.duration,
        total_requests=args.requests,
        dataset_name=args.dataset,
        max_new_tokens=args.max_new_tokens,
        stream=args.stream,
        timeout=args.timeout,
        retry_count=args.retry,
    )
    summary, path = await tester.run()
    logger.info(f"压测完成，结果文件: {path}")
    logger.info(f"摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()



