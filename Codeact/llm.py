import time

from openai import OpenAI

from config import (
    ARK_API_BASE,
    ARK_API_KEY,
    ARK_MODEL,
    CHATGLM_API_BASE,
    CHATGLM_API_KEY,
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
)


TRANSIENT_ERROR_KEYWORDS = (
    "timeout",
    "timed out",
    "handshake",
    "connection",
    "network",
    "temporarily",
    "rate limit",
    "429",
    "too many requests",
    "server error",
    "502",
    "503",
    "504",
)


def _is_transient_error(error: Exception) -> bool:
    error_msg = _format_error(error).lower()
    return any(keyword in error_msg for keyword in TRANSIENT_ERROR_KEYWORDS)


def _format_error(error: Exception) -> str:
    parts = []
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current) or repr(current)
        parts.append(f"{type(current).__name__}: {message}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


BIGMODEL_MODEL = "glm-5.1"
DEFAULT_MODEL = BIGMODEL_MODEL
DEFAULT_API_KEY = CHATGLM_API_KEY
DEFAULT_BASE_URL = CHATGLM_API_BASE
DEEPSEEK_FALLBACK_MODEL = "deepseek-chat"


def _candidate_configs(model, api_key, base_url):
    candidates = [(model, api_key, base_url)]

    # For the default GLM path, fall back to other compatible providers on
    # transient connection failures so a long batch job can keep moving.
    if (
        model == DEFAULT_MODEL
        and api_key == DEFAULT_API_KEY
        and base_url == DEFAULT_BASE_URL
    ):
        candidates.extend(
            [
                (DEEPSEEK_FALLBACK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE),
                (ARK_MODEL, ARK_API_KEY, ARK_API_BASE),
            ]
        )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _request_with_retries(
    prompt,
    query,
    model,
    api_key,
    base_url,
    timeout=60.0,
    max_retries=3,
):
    # def llm_client(prompt, query, model="gemini-3-pro-preview-11-2025", api_key="", base_url="https://yunwu.ai/v1"):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            conclusion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                temperature=0.3,
                top_p=0.7,
            )

            content = conclusion.choices[0].message.content
            if not content:
                raise ValueError("LLM returned an empty response")
            return content

        except Exception as e:
            last_error = e
            error_detail = _format_error(e)
            if attempt < max_retries - 1 and _is_transient_error(e):
                wait_time = 2 ** attempt
                print(
                    f"LLM request failed for model={model}, base_url={base_url}: "
                    f"{error_detail}. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                continue
            raise RuntimeError(
                f"LLM request failed for model={model}, base_url={base_url}: "
                f"{error_detail}"
            ) from e

    raise RuntimeError(
        f"LLM request failed for model={model}, base_url={base_url}: "
        f"{_format_error(last_error)}"
    )


# def llm_client(prompt, query, model=ARK_MODEL, api_key=ARK_API_KEY, base_url=ARK_API_BASE):
def llm_client(
    prompt,
    query,
    model=DEFAULT_MODEL,
    api_key=DEFAULT_API_KEY,
    base_url=DEFAULT_BASE_URL,
    timeout=60.0,
    max_retries=3,
):
    errors = []
    candidates = _candidate_configs(model, api_key, base_url)
    for index, (candidate_model, candidate_key, candidate_base_url) in enumerate(candidates):
        try:
            return _request_with_retries(
                prompt=prompt,
                query=query,
                model=candidate_model,
                api_key=candidate_key,
                base_url=candidate_base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as e:
            error_detail = _format_error(e)
            errors.append(error_detail)
            has_fallback = index < len(candidates) - 1
            if has_fallback and _is_transient_error(e):
                next_model, _, next_base_url = candidates[index + 1]
                print(
                    f"Switching LLM provider after transient failure. "
                    f"Next model={next_model}, base_url={next_base_url}. "
                    f"Previous error: {error_detail}"
                )
                continue
            raise RuntimeError("All LLM providers failed: " + " || ".join(errors)) from e

    raise RuntimeError("All LLM providers failed: " + " || ".join(errors))


if __name__ == "__main__":
    print(
        llm_client(
            prompt="Repeat the user's question three times.",
            query="hello",
            model="deepseek-reasoner",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
        )
    )
