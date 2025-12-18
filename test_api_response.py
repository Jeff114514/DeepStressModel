#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Optional

import requests


DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen3-80B-A3B"
DEFAULT_PROMPT = "你好！请简单介绍一下北京。"


def _build_payload(model: str, prompt: str, system_prompt: Optional[str], max_tokens: int, stream: bool) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def _print_stream_response(response: requests.Response):
    print("开始接收流式输出（SSE）...")
    full_text = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            print(f"\n[WARN] 无法解析的片段: {data}")
            continue
        choices = payload.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        chunk = delta.get("content") or choices[0].get("text", "")
        if chunk:
            full_text.append(chunk)
            print(chunk, end="", flush=True)
    print("\n[流结束]")
    if full_text:
        print("\n完整回答：\n", "".join(full_text))


def send_chat_request(
    api_url: str,
    model: str,
    prompt: str,
    api_key: str = "",
    stream: bool = True,
    max_tokens: int = 512,
    system_prompt: Optional[str] = None,
):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = _build_payload(model, prompt, system_prompt, max_tokens, stream)
    print(f"发送请求到 {api_url} (stream={stream}) ...")

    try:
        with requests.post(
            api_url,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=600,
        ) as response:
            response.raise_for_status()
            if stream:
                _print_stream_response(response)
            else:
                data = response.json()
                print(json.dumps(data, ensure_ascii=False, indent=2))
                usage = data.get("usage") or {}
                if usage:
                    print(
                        f"\nusage: prompt_tokens={usage.get('prompt_tokens')}, "
                        f"completion_tokens={usage.get('completion_tokens')}, "
                        f"total_tokens={usage.get('total_tokens')}"
                    )
    except requests.RequestException as exc:
        print(f"请求失败: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="简单的 OpenAI 接口流式测试脚本")
    parser.add_argument("--api-url", default=os.environ.get("DEEPSTRESS_TEST_API", DEFAULT_API_URL))
    parser.add_argument("--model", default=os.environ.get("DEEPSTRESS_TEST_MODEL", DEFAULT_MODEL))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default="你是一个有用的助手。")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSTRESS_TEST_KEY", ""))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="关闭流式输出")
    parser.set_defaults(stream=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    send_chat_request(
        api_url=args.api_url,
        model=args.model,
        prompt=args.prompt,
        api_key=args.api_key,
        stream=args.stream,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
    )