"""
实验批量运行脚本
对比不同并发、初始 prompt 长度、多轮对话配置下的吞吐量、QPS、首字节响应与 GPU 指标。
"""

import argparse
import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt

from src.data.dataset_manager import dataset_manager
from src.engine.load_tester import LoadTester
from src.utils.config import config
from src.utils.logger import setup_logger


matplotlib.use("Agg")
logger = setup_logger("experiment_runner")


def _parse_int_list(raw: str, default: List[int]) -> List[int]:
    if not raw:
        return default
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        logger.warning(f"无法解析列表 {raw}，使用默认值 {default}")
        return default


def _gen_prompt(length: int) -> str:
    token = "深度压力测试"
    words = [token for _ in range(max(1, length // len(token)))]
    return " ".join(words)[: max(1, length)]


def _build_prompts(length: int, rounds: int, count: int) -> List[str]:
    prompts: List[str] = []
    for i in range(count):
        base = _gen_prompt(length)
        if rounds <= 1:
            prompts.append(base)
            continue
        convo = []
        for r in range(rounds):
            convo.append(f"Round {r+1}: {base} ({i}-{r})")
        prompts.append("\n".join(convo))
    return prompts


def _default_result_dir() -> str:
    """获取实验输出目录，优先环境变量，默认为当前工作目录下的 benchmark_results。"""
    base = os.environ.get("DEEPSTRESS_RESULT_DIR")
    if not base:
        base = os.path.join(os.getcwd(), "benchmark_results")
    return os.path.abspath(base)


async def _run_case(
    backend: str,
    precision: str,
    concurrency: int,
    qps: float,
    max_new_tokens: int,
    prompt_len: int,
    rounds: int,
    total_requests: int,
    use_local_dataset: bool,
    dataset_name: str | None,
) -> Dict[str, Any]:
    prompts = None
    if not use_local_dataset:
        prompts = _build_prompts(prompt_len, rounds, max(total_requests, concurrency))

    tester = LoadTester(
        backend_name=backend,
        precision=precision,
        concurrency=concurrency,
        qps=qps,
        total_requests=total_requests,
        max_new_tokens=max_new_tokens,
        prompts_override=prompts,
        dataset_name=dataset_name,
        stream=config.get("openai_benchmarks.defaults.stream", True),
    )
    summary, path = await tester.run()
    summary["result_path"] = path
    summary["prompt_len"] = prompt_len
    summary["rounds"] = rounds
    return summary


def _save_summary_csv(rows: List[Dict[str, Any]], out_dir: str) -> str:
    if not rows:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(
        out_dir, f"experiment_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    import csv

    fieldnames = [
        "backend",
        "precision",
        "concurrency",
        "qps_target",
        "qps_observed",
        "prompt_len",
        "rounds",
        "throughput_tokens_per_s",
        "throughput_input_tokens_per_s",
        "throughput_output_tokens_per_s",
        "first_token_avg",
        "first_token_p95",
        "latency_p50",
        "latency_p95",
        "latency_p99",
        "gpu_util_avg",
        "gpu_util_max",
        "memory_used_avg",
        "memory_used_max",
        "result_path",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            gpu = r.get("gpu") or {}
            writer.writerow(
                {
                    "backend": r.get("backend"),
                    "precision": r.get("precision"),
                    "concurrency": r.get("concurrency"),
                    "qps_target": r.get("qps_target"),
                    "qps_observed": r.get("qps_observed"),
                    "prompt_len": r.get("prompt_len"),
                    "rounds": r.get("rounds"),
                    "throughput_tokens_per_s": r.get("throughput_tokens_per_s"),
                    "throughput_input_tokens_per_s": r.get("throughput_input_tokens_per_s"),
                    "throughput_output_tokens_per_s": r.get("throughput_output_tokens_per_s"),
                    "first_token_avg": r.get("first_token_avg"),
                    "first_token_p95": r.get("first_token_p95"),
                    "latency_p50": r.get("latency_p50"),
                    "latency_p95": r.get("latency_p95"),
                    "latency_p99": r.get("latency_p99"),
                    "gpu_util_avg": gpu.get("gpu_util_avg"),
                    "gpu_util_max": gpu.get("gpu_util_max"),
                    "memory_used_avg": gpu.get("memory_used_avg"),
                    "memory_used_max": gpu.get("memory_used_max"),
                    "result_path": r.get("result_path"),
                }
            )
    logger.info(f"实验汇总 CSV 已生成: {filename}")
    return filename


def _plot_compare(rows: List[Dict[str, Any]], out_dir: str):
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    # 并发 vs 吞吐量
    plt.figure()
    plt.scatter([r["concurrency"] for r in rows], [r["throughput_tokens_per_s"] for r in rows])
    plt.xlabel("Concurrency")
    plt.ylabel("Tokens/s (total)")
    plt.title("Throughput vs Concurrency")
    path1 = os.path.join(out_dir, "compare_concurrency.png")
    plt.savefig(path1)
    plt.close()

    # prompt 长度 vs 首字响应
    plt.figure()
    plt.scatter([r["prompt_len"] for r in rows], [r["first_token_avg"] for r in rows], color="orange")
    plt.xlabel("Prompt Length")
    plt.ylabel("First Token Latency (avg)")
    plt.title("First Token vs Prompt Length")
    path2 = os.path.join(out_dir, "compare_prompt_len.png")
    plt.savefig(path2)
    plt.close()

    # 多轮 vs QPS
    plt.figure()
    plt.scatter([r["rounds"] for r in rows], [r["qps_observed"] for r in rows], color="green")
    plt.xlabel("Rounds")
    plt.ylabel("QPS Observed")
    plt.title("QPS vs Rounds")
    path3 = os.path.join(out_dir, "compare_rounds_qps.png")
    plt.savefig(path3)
    plt.close()

    logger.info(f"对比图已生成: {path1}, {path2}, {path3}")


async def main_async(args: argparse.Namespace):
    backend = args.backend
    precision = args.precision
    concurrency_list = _parse_int_list(args.concurrency, [1, 2, 4, 8])
    prompt_lens = _parse_int_list(args.prompt_len, [32, 128, 512])
    rounds_list = _parse_int_list(args.rounds, [1, 3])
    total_requests = args.requests
    qps = args.qps
    max_new_tokens = args.max_new_tokens
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name

    use_local_dataset = False
    if dataset_path:
        if dataset_manager.load_benchmark_dataset(dataset_path):
            use_local_dataset = True
            logger.info(f"已加载本地数据集文件: {dataset_path}")
        else:
            logger.error(f"本地数据集加载失败: {dataset_path}")
            return
    if dataset_name:
        use_local_dataset = True
        logger.info(f"将使用本地数据集: {dataset_name}")

    if use_local_dataset:
        # 使用本地数据集时，prompt_len / rounds 不再影响提示生成
        logger.info("已启用本地数据集，prompt_len 与 rounds 参数将被忽略")
        prompt_lens = prompt_lens[:1] or [32]
        rounds_list = rounds_list[:1] or [1]

    rows: List[Dict[str, Any]] = []
    for conc in concurrency_list:
        for plen in prompt_lens:
            for rnd in rounds_list:
                logger.info(f"运行实验: 并发={conc}, prompt_len={plen}, rounds={rnd}")
                summary = await _run_case(
                    backend=backend,
                    precision=precision,
                    concurrency=conc,
                    qps=qps,
                    max_new_tokens=max_new_tokens,
                    prompt_len=plen,
                    rounds=rnd,
                    total_requests=total_requests,
                    use_local_dataset=use_local_dataset,
                    dataset_name=dataset_name,
                )
                rows.append(summary)

    result_dir = _default_result_dir()
    csv_path = _save_summary_csv(rows, result_dir)
    _plot_compare(rows, result_dir)
    logger.info(f"实验完成，汇总文件: {csv_path}")


def main():
    defaults = config.get("load_test", {}) or {}
    parser = argparse.ArgumentParser(description="批量压测对比")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--concurrency", default="1,2,4,8", help="逗号分隔并发列表")
    parser.add_argument("--prompt_len", default="32,128,512", help="逗号分隔初始prompt长度列表")
    parser.add_argument("--rounds", default="1,3", help="逗号分隔对话轮次列表")
    parser.add_argument("--qps", type=float, default=defaults.get("qps"))
    parser.add_argument("--requests", type=int, default=defaults.get("total_requests", 20))
    parser.add_argument("--max_new_tokens", type=int, default=defaults.get("max_new_tokens", 256))
    parser.add_argument("--dataset_path", default=None, help="本地数据集文件路径（JSON，包含data）")
    parser.add_argument("--dataset_name", default=None, help="已加载的数据集名称，使用本地数据集时 prompt_len/rounds 将失效")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

