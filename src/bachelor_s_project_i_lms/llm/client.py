"""Unified LLM call layer with retry, error logging, and JSON schema parsing via OpenRouter."""

import json
import logging
import os
import re
from typing import Any, Callable, Optional, Type, TypeVar
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from bachelor_s_project_i_lms.db.events import log_event

# بارگذاری متغیرهای محیطی
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Explicit cap is required: without it OpenRouter reserves the model's full output
# ceiling (131k tokens) against the account balance and rejects with 402 on low credit.
# GLM reasoning models spend completion tokens on chain-of-thought before the JSON,
# so this must comfortably cover reasoning + a full edge list (~200-800 tokens).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
DEFAULT_PROMPT_VERSION = "v1"

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# امضای تابع جنریتور برای تست‌پذیری و ماک کردن
GenerateFn = Callable[[list[dict], Optional[str]], tuple[str, dict]]


def _openrouter_generate(messages: list[dict], system: Optional[str]) -> tuple[str, dict]:
    """ارسال درخواست به OpenRouter با استفاده از کلاینت استاندارد OpenAI."""
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment or .env file.")

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    formatted_messages = []
    if system:
        formatted_messages.append({"role": "system", "content": system})
    formatted_messages.extend(messages)

    model = os.getenv("LLM_MODEL", LLM_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=formatted_messages,
        temperature=0.0,
        max_tokens=LLM_MAX_TOKENS,
    )

    choice = response.choices[0]
    text = choice.message.content or ""

    usage = {
        "input_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
        "output_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
    }
    return text, usage


def _extract_json(raw: str) -> str:
    """استخراج امن شیء JSON بیرونی و حذف تگ‌های مارک‌داون."""
    raw = raw.strip()
    raw = re.sub(r"^(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*$", "", raw)
    raw = raw.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in output")
    return raw[start : end + 1]


def call_llm(
    messages: list[dict],
    response_model: Type[T],
    system: Optional[str] = None,
    learner_id: Optional[Any] = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    generate: Optional[GenerateFn] = None,
    max_retries: int = 1,
) -> T:
    """
    فراخوانی مدل، اعتبارسنجی خروجی بر اساس Pydantic و لاگ کردن رخدادها در events.
    """
    gen = generate or _openrouter_generate
    model_name = os.getenv("LLM_MODEL", LLM_MODEL)
    history_messages = list(messages)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            raw_text, usage = gen(history_messages, system)
        except Exception as api_err:
            logger.error("LLM API Call failed: %s", api_err)
            try:
                log_event(
                    type="llm_error",
                    payload={
                        "error_type": "api_error",
                        "error_message": str(api_err),
                        "attempt": attempt,
                    },
                    learner_id=learner_id,
                    model_version=model_name,
                    prompt_version=prompt_version,
                )
            except Exception:
                pass
            raise api_err

        try:
            clean_json = _extract_json(raw_text)
            parsed_data = response_model.model_validate_json(clean_json)

            # ثبت موفقیت در دیتابیس با پارامتر type مطابق events.py
            try:
                log_event(
                    type="llm_call",
                    payload={
                        "status": "success",
                        "attempt": attempt,
                        "usage": usage,
                        "output": parsed_data.model_dump(),
                    },
                    learner_id=learner_id,
                    model_version=model_name,
                    prompt_version=prompt_version,
                )
            except Exception as db_err:
                logger.warning("Could not log success event to DB: %s", db_err)

            return parsed_data

        except (ValueError, ValidationError) as err:
            last_error = err
            logger.warning("Attempt %d validation failed: %s", attempt, err)

            try:
                log_event(
                    type="llm_validation_error",
                    payload={
                        "status": "validation_failed",
                        "attempt": attempt,
                        "error": str(err),
                        "raw_output": raw_text,
                        "usage": usage,
                    },
                    learner_id=learner_id,
                    model_version=model_name,
                    prompt_version=prompt_version,
                )
            except Exception as db_err:
                logger.warning("Could not log validation error event to DB: %s", db_err)

            if attempt < max_retries:
                history_messages.append({"role": "assistant", "content": raw_text})
                history_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response caused a validation error: {err}. "
                            f"Please provide ONLY valid JSON matching the required schema."
                        ),
                    }
                )

    raise RuntimeError(f"call_llm failed after {max_retries + 1} attempts. Last error: {last_error}")
