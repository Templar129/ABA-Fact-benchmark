# task_eval/gpt_utils.py
from __future__ import annotations
import re
import time
from typing import Dict, Any, List, Tuple, Optional

# OpenAI >= 1.0 SDK
from openai import OpenAI
import openai

# Optional tokenizer
try:
    import tiktoken
except Exception:
    tiktoken = None

SESSION_KEY_RE = re.compile(r"^session_(\d+)$")

def get_openai_client() -> OpenAI:
    return OpenAI()

def get_encoder(model_hint: Optional[str]):
    if not tiktoken:
        return None
    try:
        if model_hint and "gpt-4o" in model_hint:
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_tokens(text: str, enc) -> int:
    if not text:
        return 0
    if not enc:
        return max(1, len(text)//4)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text)//4)

def truncate_by_tokens_right(text: str, max_tokens: int, enc):
    if not text:
        return "", False, 0, 0
    if not enc or max_tokens <= 0:
        n = count_tokens(text, enc)
        return text, False, n, n
    toks = enc.encode(text)
    before = len(toks)
    if before <= max_tokens:
        return text, False, before, before
    toks = toks[-max_tokens:]
    after = len(toks)
    return enc.decode(toks), True, before, after

def extract_sessions(sample: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    convo = sample.get("conversation", {})
    if not isinstance(convo, dict):
        return []
    numbered = []
    for k, v in convo.items():
        m = SESSION_KEY_RE.match(k)
        if m and isinstance(v, list):
            numbered.append((int(m.group(1)), v))
    numbered.sort(key=lambda x: x[0])
    return [sess for _, sess in numbered]

def build_full_context_and_index(sample: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    sessions = extract_sessions(sample)
    lines, index = [], {}
    for turns in sessions:
        for t in turns:
            spk = t.get("speaker", "Speaker")
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            line = f"{spk}: {txt}"
            lines.append(line)
            dia = t.get("dia_id")
            if dia:
                index[str(dia)] = line
    return "\n".join(lines), index

def chat_once(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 6,
    initial_backoff: float = 2.0,
) -> str:
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.APITimeoutError):
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 1.7
        except Exception:
            raise