#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sudresh_reversible_formatter.py (v6)

Скрипт делает "сплошную стену" текста читаемой (абзацы/списки/структура) и переводит
все нерусскоязычные слова/фразы (латиница, иероглифы и т.д.) на русский через LLM,
при этом гарантируя строгую обратимость преобразования.

Исправления v5:
1. Исправлен TypeError: bool() undefined в tqdm (замена `if pbar:` на `if pbar is not None:`).
2. Удаление артефактов `<think>`, `Assistant:`, `User:` из ответов модели.
3. Поддержка перевода не только латиницы, но и иероглифов (CJK).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import html
import io
import json
import os
import random
import re
import sys
import time
import zipfile
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from rapidfuzz.distance import Levenshtein

# -----------------------------
# Optional deps (progress)
# -----------------------------
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

# -----------------------------
# Optional deps (LLM + HF)
# -----------------------------
try:
    from g4f.client import AsyncClient  # type: ignore
    import g4f  # type: ignore
except Exception:
    AsyncClient = None  # type: ignore
    g4f = None  # type: ignore

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore

# -----------------------------
# Optional deps (rate limiting / retries)
# -----------------------------
try:
    from aiolimiter import AsyncLimiter  # type: ignore
except Exception:
    AsyncLimiter = None  # type: ignore

try:
    # tenacity gives battle-tested exponential backoff with jitter
    from tenacity import (  # type: ignore
        AsyncRetrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential_jitter,
    )
except Exception:
    AsyncRetrying = None  # type: ignore
    retry_if_exception_type = None  # type: ignore
    stop_after_attempt = None  # type: ignore
    wait_exponential_jitter = None  # type: ignore


# -----------------------------
# Progress bar
# -----------------------------
class _SimpleProgress:
    def __init__(self, total: Optional[int] = None, desc: str = "", file=None):
        self.total = int(total) if total is not None and total > 0 else None
        self.desc = (desc or "").strip()
        self.file = file if file is not None else sys.stderr
        self.count = 0
        self._spinner = "|/-\\"
        self._spin_i = 0
        self._last_render = 0.0
        self._start = time.perf_counter()

    def update(self, n: int = 1):
        self.count += int(n)
        now = time.perf_counter()
        if now - self._last_render < 0.05 and self.total is None:
            return
        self._last_render = now

        prefix = f"{self.desc}: " if self.desc else ""
        if self.total:
            frac = min(1.0, self.count / max(1, self.total))
            bar_len = 24
            filled = int(bar_len * frac)
            bar = "█" * filled + "·" * (bar_len - filled)
            pct = int(frac * 100)
            msg = f"\r{prefix}[{bar}] {pct:3d}% ({self.count}/{self.total})"
        else:
            spin = self._spinner[self._spin_i % len(self._spinner)]
            self._spin_i += 1
            msg = f"\r{prefix}{spin} {self.count}"

        try:
            self.file.write(msg)
            self.file.flush()
        except Exception:
            pass

    def close(self):
        try:
            elapsed = time.perf_counter() - self._start
            if self.total:
                msg = f"\r{self.desc + ': ' if self.desc else ''}{self.count}/{self.total} done in {elapsed:.1f}s\n"
            else:
                msg = f"\r{self.desc + ': ' if self.desc else ''}{self.count} done in {elapsed:.1f}s\n"
            self.file.write(msg)
            self.file.flush()
        except Exception:
            pass


def _make_progress_bar(total: Optional[int], desc: str):
    # FIXED: Removed the isatty check that was suppressing the progress bar in non-TTY terminals (e.g. IDEs)
    # if not getattr(sys.stderr, "isatty", lambda: False)():
    #     return None
    
    if tqdm is not None:
        try:
            return tqdm(total=total, desc=desc, unit="rec", dynamic_ncols=True)
        except Exception:
            return _SimpleProgress(total=total, desc=desc)
    return _SimpleProgress(total=total, desc=desc)


# -----------------------------
# Streaming JSON array parser
# -----------------------------
def iter_json_objects_from_array_textstream(
    ts: io.TextIOBase,
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    in_string = False
    escape = False
    depth = 0
    collecting = False
    buf_chars: List[str] = []
    count = 0

    while True:
        chunk = ts.read(65536)
        if not chunk:
            break
        for ch in chunk:
            if not collecting:
                if ch == "{":
                    collecting = True
                    depth = 1
                    in_string = False
                    escape = False
                    buf_chars = ["{"]
                else:
                    continue
            else:
                buf_chars.append(ch)
                if in_string:
                    if escape:
                        escape = False
                    else:
                        if ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            obj_str = "".join(buf_chars)
                            obj = json.loads(obj_str)
                            if not isinstance(obj, dict):
                                raise ValueError("Expected object in top-level array")
                            yield obj
                            count += 1
                            if limit is not None and count >= limit:
                                return
                            collecting = False
                            buf_chars = []
    return


# -----------------------------
# Helpers: normalization + join key
# -----------------------------
_WS_RE = re.compile(r"\s+")


def normalize_text_for_key(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_RE.sub(" ", s)
    return s.strip()


def extract_instruction_from_question(question: str) -> str:
    q = question.replace("\r\n", "\n").replace("\r", "\n")
    for line in q.split("\n"):
        if line.strip():
            return line.strip()
    return ""


# -----------------------------
# Dataset readers
# -----------------------------
def _pick_first_json_in_zip(z: zipfile.ZipFile) -> str:
    candidates = [n for n in z.namelist() if n.lower().endswith(".json") or n.lower().endswith(".jsonl")]
    if not candidates:
        raise FileNotFoundError("Zip не содержит .json/.jsonl")
    candidates.sort(key=lambda x: (len(x), x))
    return candidates[0]


def iter_records_local(path: str, inner: Optional[str] = None, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(str(p), "r") as z:
            name = inner or _pick_first_json_in_zip(z)
            with z.open(name) as f:
                if name.lower().endswith(".jsonl"):
                    for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
                        line = line.strip()
                        if not line:
                            continue
                        yield json.loads(line)
                        if limit is not None and (i + 1) >= limit:
                            return
                else:
                    ts = io.TextIOWrapper(f, encoding="utf-8")
                    yield from iter_json_objects_from_array_textstream(ts, limit=limit)
        return

    if p.suffix.lower() == ".jsonl":
        with open(str(p), "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                if limit is not None and (i + 1) >= limit:
                    return
        return

    if p.suffix.lower() == ".json":
        with open(str(p), "rb") as fb:
            head = fb.read(1024).lstrip()
        if head.startswith(b"["):
            with open(str(p), "r", encoding="utf-8") as f:
                yield from iter_json_objects_from_array_textstream(f, limit=limit)
        else:
            with open(str(p), "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                for i, rec in enumerate(obj):
                    if isinstance(rec, dict):
                        yield rec
                    if limit is not None and (i + 1) >= limit:
                        return
            elif isinstance(obj, dict):
                yield obj
            else:
                raise ValueError("Unexpected JSON type")
        return

    raise ValueError(f"Неизвестный формат файла: {path}")


def iter_records_hf(repo_id: str, split: str = "train", streaming: bool = True, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    if load_dataset is None:
        raise ImportError("Для загрузки с Hugging Face установите пакет datasets: pip install datasets")
    ds = load_dataset(repo_id, split=split, streaming=streaming)
    count = 0
    for rec in ds:
        yield dict(rec)
        count += 1
        if limit is not None and count >= limit:
            return


# -----------------------------
# Case mapping from benchmark
# -----------------------------
@dataclass
class BenchmarkMappings:
    source_to_case_id: Dict[str, str]
    case_id_order: List[str]
    case_id_to_source: Dict[str, str]
    case_id_to_source_sha1: Dict[str, str]
    case_id_to_benchmark_count: Dict[str, int]
    key_exact_to_case_id: Dict[Tuple[str, str], str]
    key_norm_to_case_id: Dict[Tuple[str, str], str]
    key_exact_to_benchmark_id: Dict[Tuple[str, str], str]
    key_exact_to_source_sha1: Dict[Tuple[str, str], str]


def build_benchmark_mappings(benchmark_records: Iterable[Dict[str, Any]]) -> BenchmarkMappings:
    source_to_case: Dict[str, str] = {}
    case_id_order: List[str] = []
    case_id_to_source: Dict[str, str] = {}
    case_id_to_source_sha1: Dict[str, str] = {}
    case_id_to_benchmark_count: Dict[str, int] = {}
    key_exact_to_case: Dict[Tuple[str, str], str] = {}
    key_norm_to_case: Dict[Tuple[str, str], str] = {}
    key_exact_to_bench_id: Dict[Tuple[str, str], str] = {}
    key_exact_to_source_sha1: Dict[Tuple[str, str], str] = {}

    for rec in benchmark_records:
        source = rec.get("source") or ""
        if source not in source_to_case:
            idx = len(source_to_case) + 1
            source_sha1 = hashlib.sha1(source.encode("utf-8")).hexdigest()
            case_id = f"case_{idx:06d}_{source_sha1[:12]}"
            source_to_case[source] = case_id
            case_id_order.append(case_id)

            case_id_to_source[case_id] = source
            case_id_to_source_sha1[case_id] = source_sha1
            case_id_to_benchmark_count[case_id] = 0

        case_id = source_to_case[source]
        case_id_to_benchmark_count[case_id] += 1

        instr = rec.get("instruction") or ""
        ca = rec.get("correct_answer") or ""

        key_exact = (instr, ca)
        key_norm = (normalize_text_for_key(instr), normalize_text_for_key(ca))

        if key_exact not in key_exact_to_case:
            key_exact_to_case[key_exact] = case_id
            if "id" in rec and isinstance(rec["id"], str):
                key_exact_to_bench_id[key_exact] = rec["id"]
            key_exact_to_source_sha1[key_exact] = case_id_to_source_sha1[case_id]

        if key_norm not in key_norm_to_case:
            key_norm_to_case[key_norm] = case_id

    return BenchmarkMappings(
        source_to_case_id=source_to_case,
        case_id_order=case_id_order,
        case_id_to_source=case_id_to_source,
        case_id_to_source_sha1=case_id_to_source_sha1,
        case_id_to_benchmark_count=case_id_to_benchmark_count,
        key_exact_to_case_id=key_exact_to_case,
        key_norm_to_case_id=key_norm_to_case,
        key_exact_to_benchmark_id=key_exact_to_bench_id,
        key_exact_to_source_sha1=key_exact_to_source_sha1,
    )


# -----------------------------
# Reversible HTML format + metrics + REGEX FIXES
# -----------------------------

# ИСПРАВЛЕНИЕ: Regex, который ловит ВСЁ, что не является:
# 1) Кирилицей (\u0400-\u04FF)
# 2) Цифрами (\d)
# 3) Пробелами (\s)
# 4) Стандартной пунктуацией (.,!?;:()[]«»"'-/ и т.д.)
# Это захватит латиницу, китайские иероглифы, эмодзи и т.д.
FOREIGN_TOKEN_RE = re.compile(
    r"[^\u0400-\u04FF\d\s\.,!?;:()\[\]{}^$«»\"\'\-\\/\\|%&*_+=<>`~@#№]+",
    flags=re.IGNORECASE
)


def build_tagged_html(field_name: str, record_id: str, original_text: str, readable_html: str) -> str:
    orig_b64 = base64.b64encode(original_text.encode("utf-8")).decode("ascii")
    return (
        f'<rmk v="1" field="{html.escape(field_name, quote=True)}" record="{html.escape(record_id, quote=True)}">'
        f'<orig encoding="base64">{orig_b64}</orig>'
        f"<view>{readable_html}</view>"
        f"</rmk>"
    )


def restore_original_from_tagged(tagged_html: str) -> str:
    m = re.search(r'<orig\s+encoding="base64">([^<]+)</orig>', tagged_html)
    if not m:
        raise ValueError("orig tag not found")
    b64 = m.group(1).strip()
    return base64.b64decode(b64.encode("ascii")).decode("utf-8")

# --- "tagged only" (реверсивная разметка без изменения текста) ---

_TRN_PAIR_RE = re.compile(
    r"<trn>\s*<ru>(.*?)</ru>\s*<src>(.*?)</src>\s*</trn>",
    flags=re.IGNORECASE | re.DOTALL,
)

def _extract_trn_map_from_readable_html(readable_html: str) -> Dict[str, str]:
    """
    Извлекает пары src -> ru из readable_html (если они там есть).
    Используется для построения 'tagged_only_html', где текст остается исходным,
    а перевод сохраняется в атрибуте data-ru.
    """
    mp: Dict[str, str] = {}
    if not readable_html:
        return mp
    for m in _TRN_PAIR_RE.finditer(readable_html):
        ru = html.unescape((m.group(1) or "").strip())
        src = html.unescape((m.group(2) or "").strip())
        if not src:
            continue
        if src not in mp:
            mp[src] = ru
    return mp


def build_tagged_only_html(original_text: str, readable_html: str) -> str:
    """
    Возвращает HTML, в котором *видимый текст* полностью совпадает с original_text,
    но иностранные фрагменты размечены тегом <trn data-ru="...">...</trn>.

    Это гарантирует обратимость: восстановление исходника возможно простым
    удалением тегов <trn ...> и HTML-unescape.
    """
    mp = _extract_trn_map_from_readable_html(readable_html)
    s = original_text or ""
    parts: List[str] = []
    last = 0
    for m in FOREIGN_TOKEN_RE.finditer(s):
        parts.append(html.escape(s[last:m.start()]))
        tok = m.group(0)
        ru = mp.get(tok) or "требуется перевод"
        parts.append(
            f'<trn data-ru="{html.escape(ru, quote=True)}">{html.escape(tok)}</trn>'
        )
        last = m.end()
    parts.append(html.escape(s[last:]))
    # <pre> сохраняет пробелы/переносы строк (важно для обратимости)
    return "<pre class=\"tagged-only\">" + "".join(parts) + "</pre>"


def restore_original_from_tagged_only(tagged_only_html: str) -> str:
    """
    Восстанавливает исходный текст из build_tagged_only_html():
    - снимает обертку <pre ...>
    - удаляет теги <trn ...>...</trn>, оставляя их текстовое содержимое
    - делает HTML-unescape
    """
    s = tagged_only_html or ""
    # снимаем <pre ...>...</pre> (если есть)
    s = re.sub(r"(?is)^\s*<pre[^>]*>", "", s)
    s = re.sub(r"(?is)</pre>\s*$", "", s)
    # убираем <trn ...> / </trn>
    s = re.sub(r"(?is)<\s*trn\b[^>]*>", "", s)
    s = re.sub(r"(?is)<\s*/\s*trn\s*>", "", s)
    return html.unescape(s)



def token_f1(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    from collections import Counter

    ca = Counter(ta)
    cb = Counter(tb)
    common = sum((ca & cb).values())
    precision = common / len(tb)
    recall = common / len(ta)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def find_untranslated_foreign(readable_html: str) -> List[str]:
    """
    Проверка: есть ли "иностранный" текст вне <src>...</src> и вне <orig>...</orig>.
    Теперь "иностранный" определяется как все, что не попадает в список безопасных символов.
    """
    # Удаляем содержимое <orig> (там base64, его проверять не надо)
    tmp = re.sub(r"<orig\b[^>]*>.*?</orig>", " ", readable_html, flags=re.DOTALL | re.IGNORECASE)
    # Удаляем содержимое <src> (там оригинал, его переводить не надо)
    tmp = re.sub(r"<src\b[^>]*>.*?</src>", " ", tmp, flags=re.DOTALL | re.IGNORECASE)
    # Удаляем сами теги
    tmp = re.sub(r"<[^>]+>", " ", tmp)

    tokens = FOREIGN_TOKEN_RE.findall(tmp)
    out: List[str] = []
    seen = set()
    for t in tokens:
        # Иногда regex может захватить одиночные символы мусора, но для строгости оставим как есть
        if len(t.strip()) > 0 and t not in seen:
            out.append(t)
            seen.add(t)
    return out


# -----------------------------
# Heuristic (offline) formatter
# -----------------------------
LABEL_RE = re.compile(r"(?P<label>[А-ЯЁA-Z][А-ЯЁA-Za-zа-яё0-9«»\"'()\\/\-\s]{1,80}?)\s*:")


def _escape_and_wrap_translations(text: str, translate_fn) -> str:
    """
    Оборачивает найденные FOREIGN_TOKEN_RE фрагменты в <trn>...</trn>.
    """
    out: List[str] = []
    last = 0
    # Используем новый regex для поиска фрагментов
    for m in FOREIGN_TOKEN_RE.finditer(text):
        # Текст до найденного токена
        out.append(html.escape(text[last:m.start()]))
        tok = m.group(0)
        # Получаем перевод
        ru = translate_fn(tok)
        out.append("<trn>")
        out.append(f"<ru>{html.escape(ru)}</ru>")
        out.append(f"<src>{html.escape(tok)}</src>")
        out.append("</trn>")
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def _strip_bullet_prefix(line: str) -> str:
    return re.sub(r"^\s*[-•*]\s+", "", line)


def _strip_number_prefix(line: str) -> str:
    return re.sub(r"^\s*\d+\.\s+", "", line)


def heuristic_readable_html(text: str, translate_fn, do_structure: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    if not do_structure:
        return f"<p>{_escape_and_wrap_translations(text, translate_fn)}</p>"

    matches = list(LABEL_RE.finditer(text))
    if len(matches) >= 2:
        items: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            label = m.group("label").strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            items.append((label, content))
        parts = ['<dl class="kv">']
        for label, content in items:
            parts.append(f"<dt>{_escape_and_wrap_translations(label, translate_fn)}</dt>")
            parts.append(f"<dd>{_escape_and_wrap_translations(content, translate_fn)}</dd>")
        parts.append("</dl>")
        return "\n".join(parts)

    blocks = re.split(r"\n\s*\n", text)
    parts: List[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        bullet_ok = all(re.match(r"^\s*[-•*]\s+", ln) for ln in lines) and len(lines) >= 2
        num_ok = all(re.match(r"^\s*\d+\.\s+", ln) for ln in lines) and len(lines) >= 2
        if bullet_ok:
            lis = "".join(
                f"<li>{_escape_and_wrap_translations(_strip_bullet_prefix(ln), translate_fn)}</li>" for ln in lines
            )
            parts.append(f"<ul>{lis}</ul>")
        elif num_ok:
            lis = "".join(
                f"<li>{_escape_and_wrap_translations(_strip_number_prefix(ln), translate_fn)}</li>" for ln in lines
            )
            parts.append(f"<ol>{lis}</ol>")
        else:
            parts.append(f"<p>{_escape_and_wrap_translations(block, translate_fn)}</p>")
    return "\n".join(parts)


def mock_translate(token: str) -> str:
    return f"перевод({token})"


# -----------------------------
# LLM transform via g4f (AsyncClient) & Cleaning
# -----------------------------

_CODE_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.IGNORECASE | re.DOTALL)

def _clean_llm_artifacts(text: str) -> str:
    """
    Удаляет мусорные артефакты из ответов LLM:
      - chain-of-thought теги/блоки (<think>...</think>), а также одиночные <think>
      - служебные маркеры ролей: Assistant/User/System/Bot (в т.ч. в виде <Assistant>)
    Важно: функция старается убирать только "обвязку" и заголовки, не трогая содержимое внутри JSON.
    """
    s = text or ""

    # 1) Удаляем блоки <think>...</think> (варианты регистра/пробелов)
    s = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", s, flags=re.DOTALL | re.IGNORECASE)

    # 2) Удаляем одиночные теги ролей вида <Assistant>, </User>, <System/> и т.п. (только сами теги)
    s = re.sub(r"</?\s*(?:assistant|user|system|bot|think)\s*/?\s*>", "", s, flags=re.IGNORECASE)

    # 3) Удаляем строки-заголовки ролей (Assistant / User / System / think), с двоеточием и без,
    #    а также markdown-заголовки типа "### Assistant"
    s = re.sub(
        r"(?im)^\s*(?:#+\s*)?(?:assistant|user|system|bot|think)\s*:?\s*$",
        "",
        s,
    )

    # 4) Удаляем префиксы ролей в начале строки "Assistant: ..." -> "..."
    s = re.sub(r"(?im)^\s*(?:assistant|user|system|bot)\s*:\s*", "", s)

    # 5) Удаляем stray 'think' в начале строки, если модель вывела его как маркер
    s = re.sub(r"(?im)^\s*think\s*$", "", s)

    # 6) Нормализуем множественные пустые строки
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()

def _strip_code_fences(text: str) -> str:
    """Удаляет обертку ```json ... ```."""
    s = (text or "").strip()
    if "```" not in s:
        return s
    m = _CODE_FENCE_BLOCK_RE.search(s)
    if m:
        return (m.group(1) or "").strip()
    # fallback
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_first_json_object_substring(text: str) -> str:
    """Возвращает подстроку первого JSON-объекта {...}."""
    s = _strip_code_fences(text)
    start = s.find("{")
    if start == -1:
        raise ValueError("LLM response does not contain '{'")
    in_string = False
    escape = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    raise ValueError("Unterminated JSON object in LLM response")

def _looks_like_html_fragment(s: str) -> bool:
    s2 = (s or "").strip().lower()
    if "<" not in s2 or ">" not in s2:
        return False
    return any(tag in s2 for tag in ("<p", "<div", "<br", "<ul", "<ol", "<li", "<table", "<section", "<article", "<span"))

def _fallback_translate_placeholder(_token: str) -> str:
    return "требуется перевод"

def _coerce_llm_output_to_readable_html(raw: str, target_text: str, llm_mode: str) -> str:
    # Сначала чистим артефакты
    cleaned = _clean_llm_artifacts(raw)
    s = _strip_code_fences(cleaned)
    
    if _looks_like_html_fragment(s):
        return s.strip()
    
    do_structure = ("structure" in (llm_mode or "")) or ("and_structure" in (llm_mode or ""))
    return heuristic_readable_html(target_text, _fallback_translate_placeholder, do_structure=do_structure).strip()

def _json_from_response_text(text: str) -> Dict[str, Any]:
    # 1. Сначала жесткая чистка от think/chat-артефактов
    clean_text = _clean_llm_artifacts(text)
    s = _strip_code_fences(clean_text)

    # 2. Попробуем как есть
    if s.lstrip().startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Достанем подстроку
    try:
        j = _extract_first_json_object_substring(s)
        try:
            obj = json.loads(j)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
             # Попробуем raw_decode
            decoder = json.JSONDecoder()
            obj_any, _ = decoder.raw_decode(j)
            if isinstance(obj_any, dict):
                return obj_any
    except Exception:
        pass
    
    raise ValueError("Failed to extract JSON from LLM response")


def _validate_readable_html_json(obj: Any) -> str:
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    keys = set(obj.keys())
    if keys != {"readable_html"}:
        raise ValueError(f"JSON must contain ONLY key 'readable_html', got keys: {sorted(keys)}")
    readable_html = obj.get("readable_html")
    if not isinstance(readable_html, str) or not readable_html.strip():
        raise ValueError("JSON must contain non-empty readable_html string")
    return readable_html


def build_llm_messages(
    target_text: str,
    llm_mode: str,
    target_name: str,
    context_text: str,
) -> List[Dict[str, str]]:
    system = (
        "Ты — редактор юридических текстов на русском языке.\n"
        "Цель: сделать целевой текст максимально читаемым (абзацы/списки), "
        "перевести все нерусскоязычные слова/фразы на русский С УЧЕТОМ КОНТЕКСТА.\n\n"
        "ВАЖНО: Нерусскоязычный текст — это не только латиница (English), но и ИЕРОГЛИФЫ (Китайский и др.), "
        "а также любые другие иностранные символы.\n\n"
        "Формат ответа: только JSON (без Markdown), один единственный объект.\n"
        "Схема ответа строго такая: {\"readable_html\": \"...\"}.\n"
        "В JSON НЕ ДОЛЖНО БЫТЬ никаких других ключей, кроме readable_html.\n"
        "Первый значащий символ ответа — '{', последний — '}'. Никакого текста до/после JSON.\n"
        "readable_html — HTML-фрагмент (НЕ полный HTML документ).\n"
        "Пример (шаблон): {\"readable_html\": \"<p>...</p>\"}\n\n"
        "Критические требования:\n"
        "1) Переведи ВСЕ нерусскоязычные слова/фразы (латиница, иероглифы, CJK) на русский.\n"
        "2) Каждый переведенный фрагмент ОБЯЗАТЕЛЬНО оформи как:\n"
        "   <trn><ru>РУССКИЙ ПЕРЕВОД</ru><src>ORIGINAL</src></trn>\n"
        "   где ORIGINAL — исходный фрагмент ровно как в целевом тексте (сохранив иероглифы/латиницу).\n"
        "3) Смысл и юридические формулировки сохраняй.\n"
        "4) Не добавляй никаких тройных кавычек, никаких ```.\n"
        "5) Не пиши никаких мыслей (<think>) или вступлений.\n"
        "6) Нерусские фрагменты могут быть ВНУТРИ слов с кириллицей: например, 'на实质' -> 'на<trn><ru>... </ru><src>实质</src></trn>'.\n"
    )

    if llm_mode == "translate":
        instructions = (
            "Сделай перевод нерусских фрагментов и минимальную разметку (абзацы).\n"
            "Сильную переработку структуры НЕ делай.\n"
        )
    else:
        instructions = (
            "Сделай максимально удобную для чтения структуру: абзацы, списки, "
            "можно преобразовать «ключ: значение» в definition list (<dl>/<dt>/<dd>).\n"
        )

    user = (
        f"КОНТЕКСТ (для понимания термина/сюжета, не переписывай его целиком):\n{context_text}\n\n"
        f"ЦЕЛЕВОЕ ПОЛЕ: {target_name}\n"
        "Сгенерируй readable_html ТОЛЬКО для целевого поля.\n\n"
        f"ЦЕЛЕВОЙ ТЕКСТ:\n{target_text}"
    )

    return [
        {"role": "system", "content": system + "\n" + instructions},
        {"role": "user", "content": user},
    ]


async def _chat_create_with_timeout(client: Any, timeout_s: Optional[int], **kwargs: Any) -> Any:
    if "provider" in kwargs and kwargs["provider"] is None:
        kwargs.pop("provider", None)

    try:
        fn = client.chat.completions.create
        sig = inspect.signature(fn)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_varkw:
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        pass

    call = client.chat.completions.create(**kwargs)
    if timeout_s and timeout_s > 0:
        return await asyncio.wait_for(call, timeout=float(timeout_s))
    return await call


async def llm_make_readable_html(
    client: Any,
    target_text: str,
    llm_mode: str,
    target_name: str,
    context_text: str,
    model: str,
    provider: Optional[Any],
    max_retries: int = 3,
    repair_retries: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    request_timeout_s: Optional[int] = None,
    strict_translate: bool = True,
) -> Tuple[str, str]:
    messages = build_llm_messages(
        target_text=target_text,
        llm_mode=llm_mode,
        target_name=target_name,
        context_text=context_text,
    )

    def _is_probably_provider_error(s: str) -> bool:
        s2 = (s or "").lower()
        return any(tok in s2 for tok in (
            "error", "rate limit", "too many requests", "cloudflare", "captcha",
            "unauthorized", "forbidden", "timeout", "invalid", "blocked",
            "server", "cookie", "1psid", "__secure-1psid",
        ))

    last_err: Optional[Exception] = None
    base_messages = messages

    for attempt in range(1, max_retries + 1):
        try:
            resp = await _chat_create_with_timeout(
                client,
                request_timeout_s,
                model=model,
                messages=messages,
                provider=provider,
                web_search=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content  # type: ignore
            if raw is None or not str(raw).strip():
                raise ValueError("Empty LLM response")
            raw = str(raw)

            if _is_probably_provider_error(raw) and "{\"readable_html\"" not in raw:
                raise RuntimeError(f"Provider returned error-like text: {raw[:200]}")

            obj: Optional[Dict[str, Any]] = None
            parse_err: Optional[Exception] = None
            method = "llm"
            try:
                obj = _json_from_response_text(raw)
                _validate_readable_html_json(obj)
            except Exception as e:
                obj = None
                parse_err = e

            if obj is None:
                repair_messages = base_messages[:]
                repair_messages.append({
                    "role": "user",
                    "content": (
                        "Ты вернул НЕвалидный JSON или добавил лишний текст (мысли, преамбулы). "
                        "Верни ТОЛЬКО один JSON-объект строго вида {\"readable_html\": \"...\"}. "
                        "БЕЗ ```, БЕЗ <think>, БЕЗ лишних слов."
                    ),
                })

                for _ in range(max(0, repair_retries)):
                    _kwargs2: Dict[str, Any] = dict(
                        model=model,
                        messages=repair_messages,
                        provider=provider,
                        web_search=False,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    resp2 = await _chat_create_with_timeout(client, request_timeout_s, **_kwargs2)
                    raw2 = resp2.choices[0].message.content  # type: ignore
                    if raw2 is None or not str(raw2).strip():
                        continue
                    raw2 = str(raw2)
                    if _is_probably_provider_error(raw2) and "{\"readable_html\"" not in raw2:
                        continue
                    try:
                        obj = _json_from_response_text(raw2)
                        _validate_readable_html_json(obj)
                        raw = raw2
                        parse_err = None
                        break
                    except Exception as e:
                        parse_err = e
                        continue

            if obj is None:
                method = "heuristic"
                print(
                    f"[WARN] Non-JSON LLM output for {target_name}: {parse_err}. Using heuristic fallback.",
                    file=sys.stderr,
                )
                readable_html = "\n" + _coerce_llm_output_to_readable_html(raw, target_text, llm_mode)
            else:
                readable_html = _validate_readable_html_json(obj)

            # 4) Чистим служебные артефакты уже внутри readable_html (на случай, если модель положила их в JSON)
            readable_html = _clean_llm_artifacts(readable_html)

            # 5) Строгая проверка перевода (ищем всё, что не кирилица/цифры/знаки)
            if strict_translate:
                leftovers = find_untranslated_foreign(readable_html)
                if leftovers:
                    messages = [
                        base_messages[0],
                        {
                            "role": "user",
                            "content": (
                                "В readable_html остался непереведенный иностранный текст (латиница или иероглифы) вне <src>. Также проверь смешанные слова (например, \"на实质\"): иностранные части внутри русских слов тоже нужно разметить. "
                                f"Вот примеры: {leftovers[:50]}. "
                                "Исправь readable_html: переведи их на русский и заверни в "
                                "<trn><ru>...</ru><src>ORIGINAL</src></trn>. "
                                "Верни ТОЛЬКО JSON."
                            ),
                        },
                    ]
                    raise ValueError(f"Untranslated foreign leftovers: {leftovers[:10]}")
            return readable_html, method

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if any(tok in msg for tok in (
                "__secure-1psid", "1psid", "cookie", "cf_clearance",
                "captcha", "cloudflare", "forbidden", "unauthorized",
            )):
                break
            await asyncio.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
            # IMPORTANT: не перетираем messages, потому что они могут быть
            # усилены (например, если нашли непереведенные фрагменты и попросили исправить).
            continue

    raise RuntimeError(f"LLM failed after {max_retries} retries: {last_err}")


# -----------------------------
# LLM orchestration (providers/models rotation, circuit breaker, hedged requests)
# -----------------------------

@dataclass(frozen=True)
class _BackendSpec:
    provider: Optional[Any]
    model: str

    @property
    def provider_name(self) -> str:
        return _provider_class_name(self.provider) or "default"

    @property
    def key(self) -> str:
        return f"{self.provider_name}::{self.model}"


@dataclass
class _BackendState:
    fails: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""


def _now_s() -> float:
    return time.monotonic()


def _exp_backoff_with_jitter(attempt: int, base: float = 0.4, cap: float = 8.0) -> float:
    # Exponential backoff with jitter, similar in spirit to tenacity's wait_exponential_jitter.
    a = max(0, int(attempt))
    raw = min(cap, base * (2 ** a))
    return random.uniform(0.0, raw)


class LLMOrchestrator:
    """
    Делает вызовы LLM устойчивыми к:
      - зависаниям/таймаутам (жесткий timeout + отмена задач)
      - пустым ответам
      - флейки-провайдерам (rotation + circuit breaker + cooldown)
      - пиковым задержкам (hedged requests — 'дублирующий' запрос на запасной backend)
    """

    def __init__(
        self,
        client: Any,
        provider_candidates: Sequence[Optional[Any]],
        model_candidates: Sequence[str],
        *,
        global_concurrency: int = 5,
        per_backend_concurrency: int = 2,
        rpm_limit: Optional[int] = None,
        fail_threshold: int = 2,
        cooldown_s: float = 30.0,
        hedge_k: int = 1,
        hedge_delay_s: float = 0.25,
    ):
        self.client = client
        self.provider_candidates = list(provider_candidates) if provider_candidates else [None]
        self.model_candidates = [m for m in model_candidates if (m or "").strip()] or ["gpt-4o-mini"]

        self.global_sem = asyncio.BoundedSemaphore(max(1, int(global_concurrency)))
        self.per_backend_concurrency = max(1, int(per_backend_concurrency))
        self.fail_threshold = max(1, int(fail_threshold))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.hedge_k = max(1, int(hedge_k))
        self.hedge_delay_s = max(0.0, float(hedge_delay_s))

        # Optional: token-bucket limiter (requests per minute)
        self.limiter = None
        if rpm_limit and rpm_limit > 0 and AsyncLimiter is not None:
            # AsyncLimiter(max_rate, time_period_seconds)
            self.limiter = AsyncLimiter(int(rpm_limit), 60)

        # Per-backend semaphores to avoid hammering a single provider/model
        self._backend_sems: Dict[str, asyncio.BoundedSemaphore] = {}
        # Circuit breaker state
        self._state: Dict[str, _BackendState] = {}

        for spec in self._all_backend_specs():
            self._backend_sems[spec.key] = asyncio.BoundedSemaphore(self.per_backend_concurrency)
            self._state[spec.key] = _BackendState()

    def _all_backend_specs(self) -> List[_BackendSpec]:
        return [_BackendSpec(provider=p, model=m) for p in self.provider_candidates for m in self.model_candidates]

    def _iter_available_specs(self) -> Iterator[_BackendSpec]:
        """Yield backend specs in priority order, skipping those in cooldown."""
        now = _now_s()
        for spec in self._all_backend_specs():
            st = self._state.get(spec.key)
            if st and st.cooldown_until and st.cooldown_until > now:
                continue
            yield spec

    def _mark_success(self, spec: _BackendSpec):
        st = self._state.setdefault(spec.key, _BackendState())
        st.fails = 0
        st.cooldown_until = 0.0
        st.last_error = ""

    def _mark_failure(self, spec: _BackendSpec, err: Exception):
        st = self._state.setdefault(spec.key, _BackendState())
        st.fails += 1
        st.last_error = str(err)[:500]
        if self.cooldown_s > 0 and st.fails >= self.fail_threshold:
            # exponential-ish cooldown, capped
            extra = min(300.0, self.cooldown_s * (2 ** (st.fails - self.fail_threshold)))
            st.cooldown_until = _now_s() + extra

    async def _call_one_backend(
        self,
        spec: _BackendSpec,
        *,
        target_text: str,
        llm_mode: str,
        target_name: str,
        context_text: str,
        max_retries: int,
        repair_retries: int,
        temperature: float,
        max_tokens: int,
        request_timeout_s: Optional[int],
        strict_translate: bool,
    ) -> Tuple[str, str]:
        # Global concurrency
        async with self.global_sem:
            # Per-backend concurrency
            sem = self._backend_sems.setdefault(spec.key, asyncio.BoundedSemaphore(self.per_backend_concurrency))
            async with sem:
                # Optional RPM limit
                if self.limiter is not None:
                    async with self.limiter:
                        pass

                html_out, method = await llm_make_readable_html(
                    client=self.client,
                    target_text=target_text,
                    llm_mode=llm_mode,
                    target_name=target_name,
                    context_text=context_text,
                    model=spec.model,
                    provider=spec.provider,
                    max_retries=max_retries,
                    repair_retries=repair_retries,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout_s=request_timeout_s,
                    strict_translate=strict_translate,
                )
                return html_out, method

    async def make_readable_html(
        self,
        *,
        target_text: str,
        llm_mode: str,
        target_name: str,
        context_text: str,
        llm_max_retries: int,
        llm_repair_retries: int,
        llm_temperature: float,
        llm_max_tokens: int,
        llm_timeout_s: Optional[int],
        strict_translate: bool,
    ) -> Tuple[str, str]:
        """
        Пытается получить ответ LLM. Если backend флейки/молчит — пробует следующие.
        Использует hedging, если hedge_k > 1.
        """
        candidates = list(self._iter_available_specs())
        if not candidates:
            # everything in cooldown: ignore cooldown for this call
            candidates = self._all_backend_specs()

        last_err: Optional[Exception] = None
        overall_attempts = max(1, int(llm_max_retries))

        cand_i = 0
        for attempt in range(overall_attempts):
            round_specs = candidates[cand_i : cand_i + self.hedge_k]
            cand_i += len(round_specs)
            if not round_specs:
                cand_i = 0
                round_specs = candidates[cand_i : cand_i + self.hedge_k]
                cand_i += len(round_specs)

            tasks: List[asyncio.Task] = []

            async def _runner(spec: _BackendSpec, delay_s: float) -> Tuple[_BackendSpec, Tuple[str, str]]:
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                res = await self._call_one_backend(
                    spec,
                    target_text=target_text,
                    llm_mode=llm_mode,
                    target_name=target_name,
                    context_text=context_text,
                    max_retries=1,  # orchestrator handles overall retries/rotation
                    repair_retries=llm_repair_retries,
                    temperature=llm_temperature,
                    max_tokens=llm_max_tokens,
                    request_timeout_s=llm_timeout_s,
                    strict_translate=strict_translate,
                )
                return spec, res

            for j, spec in enumerate(round_specs):
                delay = 0.0 if j == 0 else self.hedge_delay_s * float(j)
                tasks.append(asyncio.create_task(_runner(spec, delay)))

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # If first completed succeeded -> cancel rest
            succeeded = None
            for d in done:
                try:
                    succeeded = d.result()
                    break
                except Exception as e:
                    last_err = e

            if succeeded is not None:
                spec, (html_out, method) = succeeded
                self._mark_success(spec)
                for p in pending:
                    p.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                return html_out, method

            # Round failed: mark failure for the "primary" spec (best-effort)
            if round_specs and last_err is not None:
                self._mark_failure(round_specs[0], last_err)

            # Cancel pending hedged calls
            for p in pending:
                p.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            # Backoff before next round
            await asyncio.sleep(_exp_backoff_with_jitter(attempt))

        raise RuntimeError(f"LLM orchestrator failed after {overall_attempts} rounds: {last_err}")


# -----------------------------
# HTML sandbox generator
# -----------------------------
def build_sandbox_html(records_view: List[Dict[str, Any]]) -> str:
    data_json = json.dumps(records_view, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<title>Sud-resh: Reversible readability sandbox</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 0; }}
  header {{ padding: 12px 16px; background: #111; color: #fff; }}
  header small {{ opacity: .8; }}
  .controls {{ padding: 10px 16px; border-bottom: 1px solid #ddd; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; padding: 8px; height: calc(100vh - 110px); box-sizing: border-box; }}
  .pane {{ border: 1px solid #ddd; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }}
  .pane h2 {{ margin: 0; padding: 8px 10px; font-size: 14px; background: #f6f6f6; border-bottom: 1px solid #ddd; }}
  textarea {{ width: 100%; height: 100%; border: 0; padding: 10px; box-sizing: border-box; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; font-size: 12px; line-height: 1.4; resize: none; }}
  #rendered {{ padding: 10px; overflow: auto; height: 100%; box-sizing: border-box; }}
  .meta {{ font-size: 12px; color: #444; margin-bottom: 8px; }}
  section {{ margin: 12px 0; }}
  section h3 {{ margin: 0 0 6px 0; font-size: 13px; }}
  p {{ margin: 6px 0; }}
  dl.kv {{ margin: 0; }}
  dl.kv dt {{ font-weight: 700; margin-top: 8px; }}
  dl.kv dd {{ margin-left: 14px; margin-bottom: 6px; }}
  trn ru {{ font-weight: 600; }}
  trn src {{ display: none; opacity: .7; margin-left: 6px; }}
  .show-src trn src {{ display: inline; }}
  rmk orig {{ display: none; }}
  rmk view {{ display: contents; }}
</style>
</head>
<body>
<header>
  <div><b>Reversible formatting sandbox</b> <small>(примеры: {len(records_view)})</small></div>
</header>
<div class="controls">
  <label>Пример:
    <select id="sel"></select>
  </label>
  <label>Сортировка:
    <select id="sort">
      <option value="default">как в файле</option>
      <option value="llm_first">llm → heuristic</option>
      <option value="heuristic_first">heuristic → llm</option>
    </select>
  </label>
  <label><input type="checkbox" id="toggleSrc"/> Показать оригинальные нерусские фрагменты рядом с переводом</label>
</div>
<div class="grid">
  <div class="pane">
    <h2>1) Оригинальный текст</h2>
    <textarea id="orig" spellcheck="false"></textarea>
  </div>
  <div class="pane">
    <h2>2) Текст с тегами (обратимый формат)</h2>
    <textarea id="tagged" spellcheck="false"></textarea>
  </div>
  <div class="pane">
    <h2>3) Полностью преобразованный (рендер)</h2>
    <div id="rendered"></div>
  </div>
</div>
<script>
const DATA = {data_json};
const sel = document.getElementById('sel');
const sortSel = document.getElementById('sort');
let IDX = [];
const orig = document.getElementById('orig');
const tagged = document.getElementById('tagged');
const rendered = document.getElementById('rendered');
const toggleSrc = document.getElementById('toggleSrc');
function setExample(pos) {{
  const i = (IDX[pos] !== undefined) ? IDX[pos] : pos;
  const ex = DATA[i];
  orig.value = ex.original_combined;
  tagged.value = ex.tagged_combined;
  rendered.innerHTML = ex.readable_combined_html;
  rendered.classList.toggle('show-src', toggleSrc.checked);
}}
function buildIndex() {{
  IDX = DATA.map((_, i) => i);
  const mode = (sortSel && sortSel.value) ? sortSel.value : 'default';
  if (mode === 'llm_first') {{
    IDX.sort((i, j) => {{
      const ai = (DATA[i].processing_source === 'llm') ? 0 : 1;
      const aj = (DATA[j].processing_source === 'llm') ? 0 : 1;
      return (ai - aj) || (i - j);
    }});
  }} else if (mode === 'heuristic_first') {{
    IDX.sort((i, j) => {{
      const ai = (DATA[i].processing_source === 'heuristic') ? 0 : 1;
      const aj = (DATA[j].processing_source === 'heuristic') ? 0 : 1;
      return (ai - aj) || (i - j);
    }});
  }}
}}

function rebuildOptions() {{
  buildIndex();
  sel.innerHTML = '';
  IDX.forEach((i, pos) => {{
    const ex = DATA[i];
    const opt = document.createElement('option');
    opt.value = pos;
    const ps = ex.processing_source ? ex.processing_source : '';
    opt.textContent = `${{pos+1}} — ${{ps}} — case_id=${{ex.case_id}} (eval_id=${{ex.eval_id.slice(0,8)}}…)`;
    sel.appendChild(opt);
  }});
  sel.value = 0;
  setExample(0);
}}

sel.addEventListener('change', () => setExample(parseInt(sel.value, 10)));
if (sortSel) sortSel.addEventListener('change', rebuildOptions);
toggleSrc.addEventListener('change', () => {{
  rendered.classList.toggle('show-src', toggleSrc.checked);
}});
rebuildOptions();
</script>
</body>
</html>
"""


# -----------------------------
# Main pipeline
# -----------------------------
@dataclass
class Options:
    benchmark: str
    evaluated: str
    benchmark_inner: Optional[str]
    evaluated_inner: Optional[str]
    from_hf: bool
    benchmark_repo: str
    evaluated_repo: str
    hf_split: str
    hf_streaming: bool
    num_cases: Optional[int]
    max_records: Optional[int]
    case_sampling: str
    seed: int
    case_ids: Optional[List[str]]
    llm_mode: str
    model: str
    provider_name: Optional[str]
    provider_fallbacks: List[str]
    concurrency: int
    strict_translate: bool
    llm_max_retries: int
    llm_repair_retries: int
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_s: Optional[int]
    llm_per_backend_concurrency: int
    llm_rpm_limit: Optional[int]
    llm_backend_fail_threshold: int
    llm_backend_cooldown_s: float
    llm_hedge_k: int
    llm_hedge_delay_s: float
    model_fallbacks: List[str]
    model_auto: bool
    model_auto_limit: int
    out: str
    out_html: Optional[str]
    sandbox_limit: int
    case_index: Optional[str]


def _resolve_provider(provider_name: Optional[str]):
    if provider_name is None:
        return None
    if g4f is None:
        raise ImportError("g4f не установлен, но указан provider")
    if not hasattr(g4f.Provider, provider_name):
        raise ValueError(f"Неизвестный provider '{provider_name}'. Смотрите g4f.Provider.*")
    return getattr(g4f.Provider, provider_name)


def _resolve_provider_list(provider_names: Sequence[str]) -> List[Any]:
    resolved: List[Any] = []
    for name in provider_names:
        if not name:
            continue
        resolved.append(_resolve_provider(name))
    return resolved


def _provider_class_name(provider: Optional[Any]) -> Optional[str]:
    if provider is None:
        return None
    return getattr(provider, "__name__", None) or provider.__class__.__name__


def _extract_all_g4f_model_names(provider_names: Sequence[str]) -> List[str]:
    if g4f is None:
        return []
    try:
        from g4f.models import ModelRegistry  # type: ignore
        names: set[str] = set()
        if provider_names:
            for pn in provider_names:
                try:
                    for m in ModelRegistry.list_models_by_provider(str(pn)):
                        if m:
                            names.add(str(m))
                except Exception:
                    continue
        if not names:
            try:
                names.update([str(k) for k in ModelRegistry.all_models().keys()])
            except Exception:
                pass
        return sorted(names)
    except Exception:
        return []


def _model_power_score(name: str) -> float:
    n = (name or "").lower().replace("_", "-")
    score = 0.0
    if "gpt-4.5" in n or "gpt-45" in n: score += 1200
    if "gpt-4.1" in n: score += 1150
    if "gpt-4o" in n: score += 1100
    if re.search(r"\bgpt-4\b", n): score += 1000
    if re.search(r"\bo[13]\b", n) or "o1-" in n or "o3-" in n: score += 1120
    if "claude-3.5" in n: score += 1080
    if "claude-3" in n: score += 980
    if "opus" in n: score += 120
    if "sonnet" in n: score += 80
    if "deepseek-r1" in n or n == "deepseek-r1": score += 1030
    if "deepseek-v3" in n: score += 980
    if "mini" in n: score -= 200
    return score


def rank_models_by_power(model_names: Sequence[str], limit: Optional[int] = None) -> List[str]:
    uniq: List[str] = []
    seen = set()
    for m in model_names:
        m2 = (m or "").strip()
        if m2 and m2 not in seen:
            seen.add(m2)
            uniq.append(m2)
    uniq.sort(key=_model_power_score, reverse=True)
    if limit is not None:
        return uniq[: max(1, int(limit))]
    return uniq


def _build_views_from_processed(processed_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    for r in processed_records:
        orig_q = r["fields"]["question"]["original"]
        orig_a = r["fields"]["answer"]["original"]
        orig_ca = r["fields"]["correct_answer"]["original"]
        original_combined = (
            "[question]\n" + orig_q + "\n\n"
            "[answer]\n" + orig_a + "\n\n"
            "[correct_answer]\n" + orig_ca
        )
        tagged_combined = (
            f'<record eval_id="{r["eval_id"]}" case_id="{r["case_id"]}">'
            + r["fields"]["question"]["tagged_html"]
            + r["fields"]["answer"]["tagged_html"]
            + r["fields"]["correct_answer"]["tagged_html"]
            + "</record>"
        )
        readable_combined_html = (
            "<div class='record'>"
            f"<div class='meta'><b>eval_id:</b> {html.escape(r['eval_id'])} "
            f"<b>case_id:</b> {html.escape(r['case_id'])} "
            f"<b>category:</b> {html.escape(str(r.get('category','')))} "
            f"<b>test_model:</b> {html.escape(str(r.get('test_model','')))} "            f"<b>processing_source:</b> {html.escape(str(r.get('processing_source','')))}"
            "</div>"
            f"<section><h3>question</h3>{r['fields']['question']['readable_html']}</section>"
            f"<section><h3>answer</h3>{r['fields']['answer']['readable_html']}</section>"
            f"<section><h3>correct_answer</h3>{r['fields']['correct_answer']['readable_html']}</section>"
            "</div>"
        )
        views.append(
            {
                "eval_id": r["eval_id"],
                "case_id": r["case_id"],
                "processing_source": r.get("processing_source", ""),
                "original_combined": original_combined,
                "tagged_combined": tagged_combined,
                "readable_combined_html": readable_combined_html,
            }
        )
    return views


async def run_pipeline(opts: Options) -> int:
    # 1) Load benchmark -> mappings
    if opts.from_hf:
        bench_iter = iter_records_hf(opts.benchmark_repo, split=opts.hf_split, streaming=opts.hf_streaming)
    else:
        bench_iter = iter_records_local(opts.benchmark, inner=opts.benchmark_inner)

    bench_mappings = build_benchmark_mappings(bench_iter)

    all_case_ids = list(bench_mappings.case_id_order)
    if opts.case_ids:
        allowed_case_ids = {cid for cid in opts.case_ids if cid in bench_mappings.case_id_to_source}
    elif opts.num_cases is not None:
        k = min(opts.num_cases, len(all_case_ids))
        if opts.case_sampling == "random":
            rng = random.Random(opts.seed)
            allowed_case_ids = set(rng.sample(all_case_ids, k=k))
        else:
            allowed_case_ids = set(all_case_ids[:k])
    else:
        allowed_case_ids = set(all_case_ids)

    print(f"[INFO] cases selected: {len(allowed_case_ids)} / {len(all_case_ids)}")

    case_index: Dict[str, Dict[str, Any]] = {}
    if opts.case_index:
        for cid in bench_mappings.case_id_order:
            if cid not in allowed_case_ids:
                continue
            source = bench_mappings.case_id_to_source.get(cid, "")
            case_index[cid] = {
                "source": source,
                "source_sha1": bench_mappings.case_id_to_source_sha1.get(cid, hashlib.sha1(source.encode("utf-8")).hexdigest()),
                "benchmark_records": bench_mappings.case_id_to_benchmark_count.get(cid, 0),
                "evaluated_records": 0,
            }

    # 2) LLM setup
    provider_primary = _resolve_provider(opts.provider_name)
    provider_candidates: List[Any] = []
    if provider_primary is not None:
        provider_candidates.append(provider_primary)
    provider_candidates.extend(_resolve_provider_list(opts.provider_fallbacks or []))
    if not provider_candidates:
        provider_candidates = [None]
    provider_names_for_models: List[str] = [n for n in (_provider_class_name(p) for p in provider_candidates) if n]

    client = None
    if opts.llm_mode != "off":
        if AsyncClient is None:
            raise ImportError("g4f не установлен (нужен для llm-mode != off)")
        client = AsyncClient()

    model_candidates: List[str] = []
    for m in [opts.model] + (opts.model_fallbacks or []):
        m2 = (m or "").strip()
        if m2 and m2 not in model_candidates:
            model_candidates.append(m2)

    if opts.model_auto and opts.llm_mode != "off":
        auto_pool = _extract_all_g4f_model_names(provider_names_for_models)
        auto_ranked = rank_models_by_power(auto_pool, limit=opts.model_auto_limit)
        for m in auto_ranked:
            if m not in model_candidates:
                model_candidates.append(m)

        async def _probe_pair(prov: Optional[Any], mdl: str) -> bool:
            if client is None: return False
            try:
                kw = dict(model=mdl, messages=[{"role":"user","content":"ok"}], provider=prov, max_tokens=10)
                resp = await _chat_create_with_timeout(client, min(int(opts.llm_timeout_s or 30), 30), **kw)
                return bool(resp.choices[0].message.content)
            except:
                return False

        async def _select_working_pair():
            for prov in provider_candidates:
                for mdl in model_candidates:
                    if await _probe_pair(prov, mdl): return prov, mdl
            return None

        picked = await _select_working_pair()
        if picked:
            prov, mdl = picked
            print(f"[INFO] model-auto: selected provider={_provider_class_name(prov)}, model={mdl}")
            provider_candidates[:] = [prov] + [p for p in provider_candidates if p is not prov]
            model_candidates[:] = [mdl] + [m for m in model_candidates if m != mdl]

    cache: Dict[str, Tuple[str, str]] = {}
    out_path = Path(opts.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {k: {"lev_max": 0, "f1_min": 1.0, "count": 0} for k in ("question", "answer", "correct_answer")}
    sandbox_records: List[Dict[str, Any]] = []

    orchestrator: Optional[LLMOrchestrator] = None
    if opts.llm_mode != "off":
        assert client is not None
        orchestrator = LLMOrchestrator(
            client=client,
            provider_candidates=provider_candidates,
            model_candidates=model_candidates,
            global_concurrency=max(1, opts.concurrency),
            per_backend_concurrency=max(1, opts.llm_per_backend_concurrency),
            rpm_limit=opts.llm_rpm_limit,
            fail_threshold=max(1, opts.llm_backend_fail_threshold),
            cooldown_s=float(opts.llm_backend_cooldown_s),
            hedge_k=max(1, opts.llm_hedge_k),
            hedge_delay_s=float(opts.llm_hedge_delay_s),
        )

    client_reset_lock = asyncio.Lock()
    client_reset_used = False

    async def transform_field(field_name: str, target_text: str, llm_mode: str, record_context: str) -> Tuple[str, str]:
        ctx_hash = hashlib.sha1(record_context.encode("utf-8")).hexdigest()[:12]
        key = hashlib.sha1((llm_mode + "\0" + field_name + "\0" + target_text + "\0" + ctx_hash).encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key]

        if llm_mode == "off" or orchestrator is None:
            html_out = heuristic_readable_html(target_text, mock_translate, do_structure=True)
            method = "heuristic"
            cache[key] = (html_out, method)
            return cache[key]

        try:
            html_out, method = await orchestrator.make_readable_html(
                target_text=target_text,
                llm_mode=llm_mode,
                target_name=field_name,
                context_text=record_context,
                llm_max_retries=opts.llm_max_retries,
                llm_repair_retries=opts.llm_repair_retries,
                llm_temperature=opts.llm_temperature,
                llm_max_tokens=opts.llm_max_tokens,
                llm_timeout_s=opts.llm_timeout_s,
                strict_translate=opts.strict_translate,
            )
        except Exception as e:
            print(f"[WARN] LLM error for {field_name}: {e}. Trying client reset once, then fallback.", file=sys.stderr)

            # g4f иногда может залипнуть на внутреннем состоянии/сессии.
            # Делаем ОДИН безопасный reset клиента на весь рантайм, чтобы не плодить сессии.
            nonlocal client_reset_used
            if (not client_reset_used) and (AsyncClient is not None):
                async with client_reset_lock:
                    if not client_reset_used:
                        client_reset_used = True
                        try:
                            client2 = AsyncClient()
                            orch2 = LLMOrchestrator(
                                client=client2,
                                provider_candidates=provider_candidates,
                                model_candidates=model_candidates,
                                global_concurrency=max(1, opts.concurrency),
                                per_backend_concurrency=max(1, opts.llm_per_backend_concurrency),
                                rpm_limit=opts.llm_rpm_limit,
                                fail_threshold=max(1, opts.llm_backend_fail_threshold),
                                cooldown_s=float(opts.llm_backend_cooldown_s),
                                hedge_k=max(1, opts.llm_hedge_k),
                                hedge_delay_s=float(opts.llm_hedge_delay_s),
                            )
                            html_out, method = await orch2.make_readable_html(
                                target_text=target_text,
                                llm_mode=llm_mode,
                                target_name=field_name,
                                context_text=record_context,
                                llm_max_retries=opts.llm_max_retries,
                                llm_repair_retries=opts.llm_repair_retries,
                                llm_temperature=opts.llm_temperature,
                                llm_max_tokens=opts.llm_max_tokens,
                                llm_timeout_s=opts.llm_timeout_s,
                                strict_translate=opts.strict_translate,
                            )
                            cache[key] = (html_out, method)
                            return cache[key]
                        except Exception as e2:
                            print(f"[WARN] Client reset attempt also failed: {e2}. Fallback.", file=sys.stderr)

            html_out = "\n" + heuristic_readable_html(target_text, _fallback_translate_placeholder, do_structure=True)
            method = "heuristic"

        cache[key] = (html_out, method)
        return cache[key]

    if opts.from_hf:
        eval_iter = iter_records_hf(opts.evaluated_repo, split=opts.hf_split, streaming=opts.hf_streaming, limit=opts.max_records)
    else:
        eval_iter = iter_records_local(opts.evaluated, inner=opts.evaluated_inner, limit=opts.max_records)

    record_sem = asyncio.Semaphore(max(1, opts.concurrency))

    async def process_one_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        q = rec.get("question") or ""
        a = rec.get("answer") or ""
        ca = rec.get("correct_answer") or ""
        instr_line = extract_instruction_from_question(q)

        key_exact = (instr_line, ca)
        case_id = bench_mappings.key_exact_to_case_id.get(key_exact) or bench_mappings.key_norm_to_case_id.get((normalize_text_for_key(instr_line), normalize_text_for_key(ca)))

        if case_id is None or case_id not in allowed_case_ids:
            return None

        rid = str(rec.get("id") or "").strip()
        if not rid:
            rid = hashlib.sha1((instr_line + ca).encode()).hexdigest()

        out: Dict[str, Any] = {
            "eval_id": rid,
            "category": rec.get("category"),
            "test_model": rec.get("test_model"),
            "case_id": case_id,
            "join_instruction_line": instr_line,
            "instruction_code": rec.get("instruction"),
            "question_length": rec.get("question_length"),
            "answer_length": rec.get("answer_length"),
            "correct_answer_length": rec.get("correct_answer_length"),
            "generation_time_seconds": rec.get("generation_time_seconds"),
            "llm_evaluation": {
                "accuracy": rec.get("evaluation.accuracy"),
                "completeness": rec.get("evaluation.completeness"),
                "clarity": rec.get("evaluation.clarity"),
                "comment": rec.get("evaluation.comment"),
            },
            "fields": {},
            "benchmark_source_sha1": bench_mappings.case_id_to_source_sha1.get(case_id),
        }

        if key_exact in bench_mappings.key_exact_to_benchmark_id:
            out["benchmark_record_id"] = bench_mappings.key_exact_to_benchmark_id[key_exact]

        for field_name, text, ctx in (("question", q, ""), ("answer", a, q), ("correct_answer", ca, q)):
            readable, method = await transform_field(field_name, text, opts.llm_mode, ctx)
            tagged_only = build_tagged_only_html(text, readable)
            restored_only = restore_original_from_tagged_only(tagged_only)
            if restored_only != text:
                # На всякий случай: если что-то пошло не так, сохраняем разметку без data-ru
                print(f"[WARN] tagged_only_html is not perfectly reversible for {field_name}/{rid}", file=sys.stderr)
                tagged_only = build_tagged_only_html(text, "")
            tagged = build_tagged_html(field_name, rid, text, readable)
            restored = restore_original_from_tagged(tagged)
            lev = Levenshtein.distance(text, restored)
            f1 = token_f1(text, restored)
            
            out["fields"][field_name] = {
                "original": text,
                "tagged_html": tagged,
                "readable_html": readable,
                "tagged_only_html": tagged_only,
                "processing_source": method,
                "restored": restored,
                "metrics": {"levenshtein": lev, "token_f1": f1},
            }
            metrics[field_name]["count"] += 1
            metrics[field_name]["lev_max"] = max(metrics[field_name]["lev_max"], lev)
            metrics[field_name]["f1_min"] = min(metrics[field_name]["f1_min"], f1)

        out["processing_source"] = (
            "llm" if all((out.get("fields", {}).get(fn, {}).get("processing_source") == "llm") for fn in out.get("fields", {}))
            else "heuristic"
        )

        return out

    async def schedule(rec: Dict[str, Any]):
        async with record_sem:
            return await process_one_record(rec)

    tasks: List[asyncio.Task] = []
    processed_count = 0
    first = True
    pbar = _make_progress_bar(total=opts.max_records, desc="Processing")

    with open(str(out_path), "w", encoding="utf-8") as fout:
        fout.write("[\n")
        for rec in eval_iter:
            tasks.append(asyncio.create_task(schedule(rec)))
            if len(tasks) >= opts.concurrency * 2:
                done = await asyncio.gather(*tasks)
                tasks = []
                for item in done:
                    if item:
                        if not first: fout.write(",\n")
                        fout.write(json.dumps(item, ensure_ascii=False))
                        first = False
                        processed_count += 1
                        if opts.case_index: case_index[item["case_id"]]["evaluated_records"] += 1
                        if len(sandbox_records) < opts.sandbox_limit: sandbox_records.append(item)
                        if pbar is not None: pbar.update(1) # FIX HERE
        
        if tasks:
            done = await asyncio.gather(*tasks)
            for item in done:
                if item:
                    if not first: fout.write(",\n")
                    fout.write(json.dumps(item, ensure_ascii=False))
                    processed_count += 1
                    if opts.case_index: case_index[item["case_id"]]["evaluated_records"] += 1
                    if len(sandbox_records) < opts.sandbox_limit: sandbox_records.append(item)
                    if pbar is not None: pbar.update(1) # FIX HERE
    
    if pbar is not None: pbar.close() # FIX HERE

    if opts.out_html:
        with open(opts.out_html, "w", encoding="utf-8") as f:
            f.write(build_sandbox_html(_build_views_from_processed(sandbox_records)))

    if opts.case_index:
        ordered = {cid: case_index[cid] for cid in bench_mappings.case_id_order if cid in case_index}
        with open(opts.case_index, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)

    return 0


def parse_case_ids_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value: return None
    if os.path.exists(value):
        p = Path(value)
        if p.suffix.lower() == ".json":
            with open(str(p), "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list): return [str(x).strip() for x in obj if str(x).strip()]
            if isinstance(obj, dict): return [str(k).strip() for k in obj.keys() if str(k).strip()]
        else:
            ids = []
            with open(str(p), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): ids.append(line.strip())
            return ids
    return [p.strip() for p in value.split(",") if p.strip()] or None


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="")
    p.add_argument("--evaluated", default="")
    p.add_argument("--benchmark-inner", default=None)
    p.add_argument("--evaluated-inner", default=None)
    p.add_argument("--from-hf", action="store_true")
    p.add_argument("--benchmark-repo", default="lawful-good-project/sud-resh-benchmark")
    p.add_argument("--evaluated-repo", default="lawful-good-project/sud_resh_evaluated_llms_answers")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-streaming", action="store_true")
    p.add_argument("--num-cases", type=int, default=None)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--case-sampling", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--case-ids", default=None)
    p.add_argument("--case-index", default=None)
    p.add_argument("--llm-mode", choices=["off", "translate", "translate_and_structure"], default="off")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--provider", default=None)
    p.add_argument("--provider-fallbacks", default="")
    p.add_argument("--model-fallbacks", default="")
    p.add_argument("--model-auto", action="store_true")
    p.add_argument("--model-auto-limit", type=int, default=30)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--strict-translate", action="store_true")
    p.add_argument("--llm-max-retries", type=int, default=3)
    p.add_argument("--llm-repair-retries", type=int, default=2)
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-max-tokens", type=int, default=2048)
    p.add_argument("--llm-timeout-s", type=int, default=None)
    p.add_argument("--llm-per-backend-concurrency", type=int, default=2)
    p.add_argument("--llm-rpm-limit", type=int, default=None,
                   help="Optional rate limit (requests per minute) for LLM calls; requires aiolimiter")
    p.add_argument("--llm-backend-fail-threshold", type=int, default=2,
                   help="Circuit breaker: how many failures before backend is put on cooldown")
    p.add_argument("--llm-backend-cooldown-s", type=float, default=30.0,
                   help="Circuit breaker: base cooldown (seconds) for failing backend")
    p.add_argument("--llm-hedge-k", type=int, default=1,
                   help="Hedged requests: how many backends to race per attempt (>=1)")
    p.add_argument("--llm-hedge-delay-s", type=float, default=0.25,
                   help="Hedged requests: delay between launching backup attempts (seconds)")

    p.add_argument("--out", required=True)
    p.add_argument("--out-html", default=None)
    p.add_argument("--sandbox-limit", type=int, default=200)

    a = p.parse_args(argv)
    def _csv_list(s): return [x.strip() for x in (s or "").split(",") if x.strip()]
    
    if not a.from_hf and (not a.benchmark or not a.evaluated):
        p.error("Specify --benchmark and --evaluated or use --from-hf")

    return Options(
        benchmark=a.benchmark, evaluated=a.evaluated, benchmark_inner=a.benchmark_inner, evaluated_inner=a.evaluated_inner,
        from_hf=a.from_hf, benchmark_repo=a.benchmark_repo, evaluated_repo=a.evaluated_repo, hf_split=a.hf_split, hf_streaming=a.hf_streaming,
        num_cases=a.num_cases, max_records=a.max_records, case_sampling=a.case_sampling, seed=int(a.seed),
        case_ids=parse_case_ids_arg(a.case_ids), llm_mode=a.llm_mode, model=a.model, provider_name=a.provider,
        provider_fallbacks=_csv_list(a.provider_fallbacks), concurrency=max(1, a.concurrency), strict_translate=a.strict_translate,
        llm_max_retries=max(1, int(a.llm_max_retries)), llm_repair_retries=max(0, int(a.llm_repair_retries)),
        llm_temperature=float(a.llm_temperature), llm_max_tokens=max(256, int(a.llm_max_tokens)),
        llm_timeout_s=(int(a.llm_timeout_s) if a.llm_timeout_s else None), llm_per_backend_concurrency=max(1, int(a.llm_per_backend_concurrency)), llm_rpm_limit=(int(a.llm_rpm_limit) if a.llm_rpm_limit else None), llm_backend_fail_threshold=max(1, int(a.llm_backend_fail_threshold)), llm_backend_cooldown_s=float(a.llm_backend_cooldown_s), llm_hedge_k=max(1, int(a.llm_hedge_k)), llm_hedge_delay_s=float(a.llm_hedge_delay_s), model_fallbacks=_csv_list(a.model_fallbacks),
        model_auto=bool(a.model_auto), model_auto_limit=max(1, int(a.model_auto_limit)), out=a.out, out_html=a.out_html, sandbox_limit=max(1, a.sandbox_limit),
        case_index=a.case_index
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    opts = parse_args(argv)
    return asyncio.run(run_pipeline(opts))


if __name__ == "__main__":
    raise SystemExit(main())