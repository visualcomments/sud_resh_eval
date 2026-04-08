#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sudresh_expert_formatter.py

Что делает скрипт:
1. Загружает benchmark и evaluated-датасеты локально или с Hugging Face.
2. Матчит evaluated записи к case_id из benchmark так же, как исходный скрипт.
3. Форматирует поля question / answer / correct_answer в удобочитаемый HTML.
   Основной путь: g4f + модель r1-1776.
   Фолбэк: аккуратное эвристическое форматирование без LLM.
4. Сохраняет обработанный JSON.
5. Генерирует 2 автономные HTML-формы для экспертов:
   - по 400 образцов на форму,
   - 100 образцов общие для обеих форм,
   - модельные оценки сохраняются в экспортируемый JSON, но не показываются в интерфейсе.
6. В формах есть autosave, экспорт/импорт черновика, экспорт финального JSON,
   быстрые клавиши и навигация по неоцененным образцам.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import inspect
import io
import json
import os
import random
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# -----------------------------
# Optional deps
# -----------------------------
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

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

try:
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore
except Exception:
    HfApi = None  # type: ignore
    hf_hub_download = None  # type: ignore


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
    if tqdm is not None:
        try:
            return tqdm(total=total, desc=desc, unit="rec", dynamic_ncols=True)
        except Exception:
            return _SimpleProgress(total=total, desc=desc)
    return _SimpleProgress(total=total, desc=desc)


# -----------------------------
# Streaming JSON array parser
# -----------------------------
def iter_json_objects_from_array_textstream(ts: io.TextIOBase, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
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
# Helpers
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


def _normalize_hf_token(token: Optional[str]) -> Optional[str]:
    tok = (token or '').strip()
    return tok or None


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    tok = _normalize_hf_token(cli_token)
    if tok:
        return tok
    for env_name in ('HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        tok = _normalize_hf_token(os.environ.get(env_name))
        if tok:
            return tok
    return None


def install_hf_token_to_env(token: Optional[str]) -> Optional[str]:
    tok = resolve_hf_token(token)
    if not tok:
        return None
    os.environ['HF_TOKEN'] = tok
    os.environ['HUGGINGFACE_HUB_TOKEN'] = tok
    return tok


def _hf_is_data_file(path: str) -> bool:
    low = path.lower()
    return low.endswith('.json') or low.endswith('.jsonl') or low.endswith('.zip')


def _hf_file_priority(path: str, split: str) -> Tuple[int, int, str]:
    low = path.lower()
    base = low.rsplit('/', 1)[-1]
    split_low = (split or '').lower()
    ext_rank = 0 if low.endswith('.jsonl') else 1 if low.endswith('.json') else 2 if low.endswith('.zip') else 9
    split_rank = 0 if (f'/{split_low}' in low or base.startswith(split_low) or f'-{split_low}' in base or f'_{split_low}' in base) else 1
    return (split_rank, ext_rank, low)


def _hf_list_candidate_files(repo_id: str, split: str, token: Optional[str] = None) -> List[str]:
    if HfApi is None:
        raise ImportError('Для прямой загрузки с Hugging Face установите huggingface_hub: pip install -U huggingface_hub')
    api = HfApi(token=resolve_hf_token(token))
    files = api.list_repo_files(repo_id=repo_id, repo_type='dataset', token=resolve_hf_token(token))
    candidates = [f for f in files if _hf_is_data_file(f)]
    if not candidates:
        raise FileNotFoundError(f'В датасете {repo_id} не найдено файлов .json/.jsonl/.zip')

    split_matches = [f for f in candidates if _hf_file_priority(f, split)[0] == 0]
    chosen = split_matches or candidates
    chosen = sorted(chosen, key=lambda x: _hf_file_priority(x, split))
    return chosen


def _iter_records_hf_via_hub_download(
    repo_id: str,
    split: str = 'train',
    limit: Optional[int] = None,
    token: Optional[str] = None,
    files: Optional[Sequence[str]] = None,
) -> Iterator[Dict[str, Any]]:
    if hf_hub_download is None:
        raise ImportError('Для прямой загрузки с Hugging Face установите huggingface_hub: pip install -U huggingface_hub')

    token = install_hf_token_to_env(token)
    selected_files = list(files) if files else _hf_list_candidate_files(repo_id=repo_id, split=split, token=token)
    emitted = 0
    if len(selected_files) > 1:
        print(f'[INFO] HF direct mode: найдено {len(selected_files)} файлов-кандидатов в {repo_id}', file=sys.stderr)

    for filename in selected_files:
        local_path = hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=filename, token=token)
        for rec in iter_records_local(local_path, limit=None):
            yield rec
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_records_hf(
    repo_id: str,
    split: str = 'train',
    streaming: bool = True,
    limit: Optional[int] = None,
    token: Optional[str] = None,
    direct_only: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    По умолчанию сначала пытается определить raw-файлы датасета (.json/.jsonl/.zip) и,
    если они есть, читает их напрямую через huggingface_hub + локальный потоковый парсер.
    Это намеренно обходит путь datasets.load_dataset(...), который на больших JSON может
    падать в pyarrow с OverflowError / DatasetGenerationError.

    token можно передать явно через --hf-token или через переменные окружения HF_TOKEN / HUGGINGFACE_HUB_TOKEN.
    """
    token = install_hf_token_to_env(token)

    direct_candidates: Optional[List[str]] = None
    direct_detect_error: Optional[Exception] = None

    if hf_hub_download is not None:
        try:
            direct_candidates = _hf_list_candidate_files(repo_id=repo_id, split=split, token=token)
        except Exception as e:
            direct_detect_error = e
            direct_candidates = None

    if direct_candidates:
        print(f'[INFO] HF auto mode: raw data files detected for {repo_id}; using direct Hub download.', file=sys.stderr)
        yield from _iter_records_hf_via_hub_download(
            repo_id=repo_id,
            split=split,
            limit=limit,
            token=token,
            files=direct_candidates,
        )
        return

    if direct_only:
        if direct_detect_error is not None:
            raise RuntimeError(
                f'Не удалось включить прямое чтение Hugging Face для {repo_id}: {direct_detect_error}'
            ) from direct_detect_error
        raise RuntimeError(
            f'Для {repo_id} не найдены raw-файлы .json/.jsonl/.zip, а --hf-direct-only включен.'
        )

    if load_dataset is None:
        if direct_detect_error is not None:
            raise RuntimeError(
                f'load_dataset недоступен, а прямое чтение Hugging Face не удалось для {repo_id}: {direct_detect_error}'
            ) from direct_detect_error
        raise ImportError('datasets не установлен, а прямое чтение Hugging Face не удалось определить.')

    if direct_detect_error is not None:
        print(
            f'[WARN] HF direct auto-detect failed for {repo_id}: {direct_detect_error}. Falling back to load_dataset().',
            file=sys.stderr,
        )

    try:
        try:
            ds = load_dataset(repo_id, split=split, streaming=streaming, token=token)
        except TypeError:
            ds = load_dataset(repo_id, split=split, streaming=streaming, use_auth_token=token)
        count = 0
        for rec in ds:
            yield dict(rec)
            count += 1
            if limit is not None and count >= limit:
                return
        return
    except OverflowError as e:
        print(f'[WARN] load_dataset() failed for {repo_id} with OverflowError: {e}. Falling back to direct Hub download.', file=sys.stderr)
        yield from _iter_records_hf_via_hub_download(repo_id=repo_id, split=split, limit=limit, token=token)
        return
    except Exception as e:
        msg = str(e).lower()
        if any(tok in msg for tok in ('value too large to convert to int32_t', 'readoptions.block_size', 'pyarrow._json', 'datasetgenerationerror')):
            print(f'[WARN] load_dataset() failed for {repo_id}: {e}. Falling back to direct Hub download.', file=sys.stderr)
            yield from _iter_records_hf_via_hub_download(repo_id=repo_id, split=split, limit=limit, token=token)
            return
        raise


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


def build_benchmark_mappings(benchmark_records: Iterable[Dict[str, Any]]) -> BenchmarkMappings:
    source_to_case: Dict[str, str] = {}
    case_id_order: List[str] = []
    case_id_to_source: Dict[str, str] = {}
    case_id_to_source_sha1: Dict[str, str] = {}
    case_id_to_benchmark_count: Dict[str, int] = {}
    key_exact_to_case: Dict[Tuple[str, str], str] = {}
    key_norm_to_case: Dict[Tuple[str, str], str] = {}
    key_exact_to_bench_id: Dict[Tuple[str, str], str] = {}

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
    )


# -----------------------------
# HTML formatting helpers
# -----------------------------
SAFE_TAGS = {
    "p", "br", "ul", "ol", "li", "strong", "b", "em", "i", "u",
    "blockquote", "code", "pre", "dl", "dt", "dd",
    "table", "thead", "tbody", "tr", "th", "td",
    "h3", "h4", "h5", "div", "span"
}

SCRIPT_RE = re.compile(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", flags=re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9]+)([^>]*)>")
ON_ATTR_RE = re.compile(r"\s+on[a-zA-Z]+\s*=\s*(?:\".*?\"|'.*?'|[^\s>]+)", flags=re.IGNORECASE | re.DOTALL)
JS_URL_RE = re.compile(r"javascript:\s*", flags=re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```(?:html|json|markdown)?\s*(.*?)\s*```", flags=re.IGNORECASE | re.DOTALL)
ROLE_ARTIFACT_RE = re.compile(r"(?im)^\s*(assistant|user|system|bot|think)\s*:\s*")
BULLET_RE = re.compile(r"^\s*[-•*]\s+")
NUMBERED_RE = re.compile(r"^\s*\d+[\.)]\s+")
LABEL_RE = re.compile(r"(?P<label>[А-ЯЁA-Z][А-ЯЁA-Za-zа-яё0-9«»\"'()\/\-\s]{1,80}?)\s*:")


def clean_llm_artifacts(text: str) -> str:
    s = text or ""
    s = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"</?\s*(assistant|user|system|bot|think)\s*/?>", "", s, flags=re.IGNORECASE)
    s = ROLE_ARTIFACT_RE.sub("", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if "```" not in s:
        return s
    m = CODE_FENCE_RE.search(s)
    if m:
        return (m.group(1) or "").strip()
    s = re.sub(r"^```(?:html|json|markdown)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def sanitize_html_fragment(fragment: str) -> str:
    s = clean_llm_artifacts(strip_code_fences(fragment))
    s = SCRIPT_RE.sub("", s)

    def _replace_tag(m: re.Match) -> str:
        slash, tag, attrs = m.group(1), m.group(2).lower(), m.group(3) or ""
        if tag not in SAFE_TAGS:
            return ""
        if slash:
            return f"</{tag}>"
        attrs = ON_ATTR_RE.sub("", attrs)
        attrs = JS_URL_RE.sub("", attrs)
        attrs = re.sub(r"\s+style\s*=\s*(?:\".*?\"|'.*?'|[^\s>]+)", "", attrs, flags=re.IGNORECASE | re.DOTALL)
        return f"<{tag}{attrs}>"

    s = TAG_RE.sub(_replace_tag, s)
    return s.strip()


def heuristic_format_html(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "<p></p>"

    matches = list(LABEL_RE.finditer(text))
    if len(matches) >= 2:
        items: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            label = m.group("label").strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            items.append((label, content))
        parts = ["<dl>"]
        for label, content in items:
            parts.append(f"<dt><strong>{html.escape(label)}</strong></dt>")
            parts.append(f"<dd>{html.escape(content)}</dd>")
        parts.append("</dl>")
        return "\n".join(parts)

    blocks = re.split(r"\n\s*\n", text)
    out: List[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [ln.rstrip() for ln in block.split("\n") if ln.strip()]
        if len(lines) >= 2 and all(BULLET_RE.match(ln) for ln in lines):
            out.append("<ul>")
            for ln in lines:
                out.append(f"<li>{html.escape(BULLET_RE.sub('', ln))}</li>")
            out.append("</ul>")
        elif len(lines) >= 2 and all(NUMBERED_RE.match(ln) for ln in lines):
            out.append("<ol>")
            for ln in lines:
                out.append(f"<li>{html.escape(NUMBERED_RE.sub('', ln))}</li>")
            out.append("</ol>")
        else:
            text_block = " ".join(ln.strip() for ln in lines)
            out.append(f"<p>{html.escape(text_block)}</p>")
    return "\n".join(out) if out else f"<p>{html.escape(text)}</p>"


def validate_or_fallback_html(raw_text: str, llm_html: str) -> str:
    cleaned = sanitize_html_fragment(llm_html)
    if cleaned and any(tag in cleaned.lower() for tag in ("<p", "<ul", "<ol", "<dl", "<div", "<blockquote", "<table", "<h3", "<h4")):
        return cleaned
    return heuristic_format_html(raw_text)


# -----------------------------
# g4f formatting
# -----------------------------
def build_llm_messages(target_text: str, target_name: str, context_text: str) -> List[Dict[str, str]]:
    system = (
        "Ты — редактор юридических и экспертных текстов на русском языке. "
        "Тебе дают сырой текст, который нужно только ПЕРЕФОРМАТИРОВАТЬ для удобства чтения экспертом. "
        "Нельзя восстанавливать исходный текст, нельзя добавлять служебные теги, нельзя писать пояснения. "
        "Нужно: аккуратно разбить текст на смысловые абзацы, оформить списки и перечисления, "
        "сохранить юридический смысл, перевести иностранные слова и фразы на русский язык по контексту, "
        "оставив на выходе только чистый удобочитаемый HTML-фрагмент. "
        "Разрешенные теги: <p>, <ul>, <ol>, <li>, <strong>, <em>, <blockquote>, <dl>, <dt>, <dd>, <br>, <h3>, <h4>, <table>, <thead>, <tbody>, <tr>, <th>, <td>. "
        "Никакого Markdown, никаких ``` блоков, никаких <script>, никаких комментариев. "
        "Если в тексте есть иностранные слова, переведи их на русский. "
        "Первый символ ответа должен быть '<' и ответ должен быть только HTML-фрагментом."
    )
    user = (
        f"Контекст вопроса/кейса:\n{context_text or '—'}\n\n"
        f"Поле: {target_name}\n\n"
        f"Сырой текст для форматирования:\n{target_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _is_outer_task_cancelled() -> bool:
    task = asyncio.current_task()
    if task is None:
        return False
    try:
        return task.cancelling() > 0
    except Exception:
        return task.cancelled()


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
    try:
        if timeout_s and timeout_s > 0:
            return await asyncio.wait_for(call, timeout=float(timeout_s))
        return await call
    except asyncio.CancelledError as e:
        if _is_outer_task_cancelled():
            raise
        raise RuntimeError("g4f request was cancelled unexpectedly by provider/session") from e


G4F_MODEL_PROVIDER_PREFERENCES: Dict[str, List[str]] = {
    "r1-1776": ["PerplexityLabs", "Perplexity"],
    "sonar": ["PerplexityLabs", "Perplexity"],
    "sonar-pro": ["PerplexityLabs", "Perplexity"],
    "sonar-reasoning": ["PerplexityLabs", "Perplexity"],
    "sonar-reasoning-pro": ["PerplexityLabs", "Perplexity"],
}

G4F_AUTH_REQUIRED_PROVIDER_NAMES = {
    "PuterJS", "Puter", "Gemini", "GeminiPro", "OpenRouter", "DeepInfra", "Together",
    "HuggingFaceMedia", "HuggingFaceAPI", "Yupp", "Anthropic", "Groq",
}


def _provider_display_name(provider: Optional[Any]) -> Optional[str]:
    if provider is None:
        return None
    return getattr(provider, "__name__", None) or provider.__class__.__name__


def _resolve_provider(provider_name: Optional[str]):
    if provider_name is None:
        return None
    if g4f is None:
        raise ImportError("g4f не установлен, но указан provider")
    if not hasattr(g4f.Provider, provider_name):
        raise ValueError(f"Неизвестный provider '{provider_name}'. Смотрите g4f.Provider.*")
    return getattr(g4f.Provider, provider_name)


def _resolve_model_provider(model: str, requested_provider_name: Optional[str]) -> Optional[Any]:
    if g4f is None:
        return None
    if requested_provider_name:
        return _resolve_provider(requested_provider_name)
    for candidate_name in G4F_MODEL_PROVIDER_PREFERENCES.get((model or "").strip(), []):
        if hasattr(g4f.Provider, candidate_name):
            return getattr(g4f.Provider, candidate_name)
    return None


def _looks_like_api_key_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return (
        "api key" in msg
        or "api_key" in msg
        or 'add a "api_key"' in msg
        or "puter.js" in msg
        or "puterjs" in msg
        or "requires authentication" in msg
    )


def _client_supports_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_varkw:
            return kwargs
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def create_g4f_async_client(provider: Optional[Any], api_key: Optional[str]) -> Any:
    if AsyncClient is None:
        return None
    kwargs: Dict[str, Any] = {}
    if provider is not None:
        kwargs["provider"] = provider
    if api_key:
        kwargs["api_key"] = api_key
    kwargs = _client_supports_kwargs(AsyncClient, kwargs)
    try:
        return AsyncClient(**kwargs)
    except TypeError:
        return AsyncClient()


def resolve_g4f_api_key(cli_value: Optional[str]) -> Optional[str]:
    for value in (
        cli_value,
        os.environ.get("G4F_API_KEY"),
        os.environ.get("PUTER_API_KEY"),
        os.environ.get("OPENROUTER_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
    ):
        if value and str(value).strip():
            return str(value).strip()
    return None


async def llm_format_html(
    client: Any,
    target_text: str,
    target_name: str,
    context_text: str,
    model: str,
    provider: Optional[Any],
    max_retries: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 2200,
    request_timeout_s: Optional[int] = None,
) -> str:
    messages = build_llm_messages(target_text, target_name, context_text)
    last_err: Optional[Exception] = None

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
            return validate_or_fallback_html(target_text, str(raw))
        except asyncio.CancelledError as e:
            if _is_outer_task_cancelled():
                raise
            last_err = RuntimeError(f"Unexpected CancelledError from provider for model {model}")
        except Exception as e:
            last_err = e
            if provider is not None:
                provider_name = _provider_display_name(provider) or "unknown"
                if _looks_like_api_key_error(e):
                    raise RuntimeError(f"Provider {provider_name} requires api_key or interactive auth for model {model}: {e}")

            # Экспоненциальный backoff с небольшим jitter, чтобы не бомбить provider burst-запросами.
            # Это согласуется с best practices из документации g4f для async usage.
            delay = min(45.0, (1.25 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.75))
            if attempt < max_retries:
                await asyncio.sleep(delay)
                continue
        delay = min(45.0, (1.25 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.75))
        if attempt < max_retries:
            await asyncio.sleep(delay)

    raise RuntimeError(f"LLM failed after {max_retries} retries: {last_err}")


# -----------------------------
# Expert forms
# -----------------------------
def choose_assignments(
    processed_records: List[Dict[str, Any]],
    samples_per_expert: int,
    shared_samples: int,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if shared_samples >= samples_per_expert:
        raise ValueError("shared_samples must be smaller than samples_per_expert")

    required_unique = samples_per_expert * 2 - shared_samples
    if len(processed_records) < required_unique:
        raise ValueError(
            f"Недостаточно обработанных записей для двух форм: нужно минимум {required_unique}, доступно {len(processed_records)}"
        )

    rng = random.Random(seed)
    pool = processed_records[:]
    rng.shuffle(pool)

    common = pool[:shared_samples]
    rest = pool[shared_samples:]
    per_expert_unique = samples_per_expert - shared_samples
    expert1_unique = rest[:per_expert_unique]
    expert2_unique = rest[per_expert_unique:per_expert_unique * 2]

    expert1 = [dict(item, is_shared_between_experts=True) for item in common] + [dict(item, is_shared_between_experts=False) for item in expert1_unique]
    expert2 = [dict(item, is_shared_between_experts=True) for item in common] + [dict(item, is_shared_between_experts=False) for item in expert2_unique]
    rng.shuffle(expert1)
    rng.shuffle(expert2)

    for idx, item in enumerate(expert1, start=1):
        item["form_sample_index"] = idx
    for idx, item in enumerate(expert2, start=1):
        item["form_sample_index"] = idx

    return {
        "expert_1": expert1,
        "expert_2": expert2,
        "shared": [x["eval_id"] for x in common],
    }


def build_form_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "eval_id": record["eval_id"],
        "case_id": record["case_id"],
        "category": record.get("category"),
        "test_model": record.get("test_model"),
        "benchmark_record_id": record.get("benchmark_record_id"),
        "question_html": record["fields"]["question"]["formatted_html"],
        "answer_html": record["fields"]["answer"]["formatted_html"],
        "correct_answer_html": record["fields"]["correct_answer"]["formatted_html"],
        "question_raw": record["fields"]["question"]["original"],
        "answer_raw": record["fields"]["answer"]["original"],
        "correct_answer_raw": record["fields"]["correct_answer"]["original"],
        "generation_time_seconds": record.get("generation_time_seconds"),
        "model_evaluation": record.get("llm_evaluation") or {},
        "processing": record.get("processing") or {},
        "is_shared_between_experts": bool(record.get("is_shared_between_experts", False)),
        "form_sample_index": record.get("form_sample_index"),
    }


def build_expert_form_html(
    *,
    form_id: str,
    expert_name: str,
    samples: List[Dict[str, Any]],
    shared_count: int,
    generated_at: str,
) -> str:
    payload = [build_form_payload(x) for x in samples]
    data_json = json.dumps(payload, ensure_ascii=False)
    meta_json = json.dumps(
        {
            "schema_version": "1.0",
            "form_id": form_id,
            "expert_name": expert_name,
            "total_samples": len(samples),
            "shared_samples": shared_count,
            "generated_at": generated_at,
        },
        ensure_ascii=False,
    )

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Экспертная оценка — {html.escape(expert_name)}</title>
<style>
  :root {{
    --bg: #0b1020;
    --panel: #11182d;
    --panel-2: #17213a;
    --border: #2b3a63;
    --text: #edf2ff;
    --muted: #9fb0d9;
    --accent: #6ea8ff;
    --ok: #3ddc97;
    --warn: #ffcc66;
    --danger: #ff7b7b;
    --shadow: 0 12px 32px rgba(0, 0, 0, .28);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
  body {{ height: 100vh; overflow: hidden; }}
  button, input, textarea, select {{ font: inherit; }}
  .app {{ display: grid; grid-template-columns: 320px 1fr; height: 100vh; }}
  .sidebar {{ border-right: 1px solid var(--border); background: rgba(10, 15, 29, 0.95); display: flex; flex-direction: column; min-width: 0; }}
  .sidebar-head {{ padding: 16px; border-bottom: 1px solid var(--border); }}
  .title {{ font-size: 18px; font-weight: 700; margin-bottom: 6px; }}
  .meta {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
  .progress-wrap {{ margin-top: 12px; }}
  .progress-bar {{ width: 100%; height: 10px; border-radius: 999px; background: #0e1530; overflow: hidden; border: 1px solid var(--border); }}
  .progress-fill {{ height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent), var(--ok)); }}
  .toolbar {{ display: grid; gap: 8px; padding: 12px 16px; border-bottom: 1px solid var(--border); }}
  .toolbar-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .toolbar button, .toolbar select, .toolbar input[type="search"] {{ background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px; }}
  .toolbar button {{ cursor: pointer; }}
  .toolbar button:hover {{ border-color: var(--accent); }}
  .status-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; background: rgba(110,168,255,.12); color: var(--muted); font-size: 12px; border: 1px solid var(--border); }}
  .sample-list {{ overflow: auto; padding: 8px; display: grid; gap: 6px; }}
  .sample-item {{ border: 1px solid var(--border); background: var(--panel); border-radius: 12px; padding: 10px 12px; cursor: pointer; }}
  .sample-item:hover {{ border-color: var(--accent); }}
  .sample-item.active {{ border-color: var(--accent); box-shadow: 0 0 0 1px rgba(110,168,255,.35) inset; }}
  .sample-item.rated {{ border-color: rgba(61,220,151,.45); }}
  .sample-item.flagged {{ border-color: rgba(255,204,102,.6); }}
  .sample-top {{ display: flex; justify-content: space-between; gap: 8px; font-size: 12px; color: var(--muted); }}
  .sample-main {{ margin-top: 6px; font-size: 14px; font-weight: 600; }}
  .sample-tags {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }}
  .tag {{ padding: 2px 8px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); font-size: 11px; }}
  .content {{ overflow: auto; min-width: 0; }}
  .content-inner {{ max-width: 1100px; margin: 0 auto; padding: 22px; display: grid; gap: 16px; }}
  .sticky-nav {{ position: sticky; top: 0; z-index: 10; display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 14px 18px; background: rgba(11, 16, 32, .92); backdrop-filter: blur(14px); border-bottom: 1px solid var(--border); }}
  .sticky-nav .nav-left, .sticky-nav .nav-right {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
  .sticky-nav button {{ background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: 8px 12px; cursor: pointer; }}
  .sticky-nav button:hover {{ border-color: var(--accent); }}
  .card {{ background: linear-gradient(180deg, rgba(23,33,58,.96), rgba(17,24,45,.96)); border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); }}
  .card h2 {{ margin: 0 0 10px 0; font-size: 18px; }}
  .card h3 {{ margin: 0 0 10px 0; font-size: 16px; }}
  .small-note {{ color: var(--muted); font-size: 13px; }}
  .doc {{ line-height: 1.62; font-size: 15px; }}
  .doc p {{ margin: 0 0 12px 0; }}
  .doc ul, .doc ol, .doc dl, .doc blockquote, .doc table {{ margin: 0 0 14px 0; }}
  .doc blockquote {{ padding-left: 14px; border-left: 3px solid var(--accent); color: #d8e4ff; }}
  .doc table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  .doc th, .doc td {{ border: 1px solid var(--border); padding: 8px; vertical-align: top; }}
  details {{ border: 1px solid var(--border); border-radius: 14px; padding: 10px 12px; background: rgba(9, 14, 26, .45); }}
  summary {{ cursor: pointer; font-weight: 600; }}
  .rating-grid {{ display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
  .rating-btn {{ border: 1px solid var(--border); background: var(--panel); color: var(--text); border-radius: 12px; min-height: 48px; cursor: pointer; font-weight: 700; }}
  .rating-btn:hover {{ border-color: var(--accent); transform: translateY(-1px); }}
  .rating-btn.active {{ border-color: var(--accent); background: rgba(110,168,255,.16); box-shadow: 0 0 0 1px rgba(110,168,255,.4) inset; }}
  .rating-hint {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
  .controls-grid {{ display: grid; gap: 12px; grid-template-columns: 1fr auto; align-items: start; }}
  .comment-box textarea {{ width: 100%; min-height: 130px; resize: vertical; background: #0b1227; color: var(--text); border: 1px solid var(--border); border-radius: 14px; padding: 12px; }}
  .checkline {{ display: flex; gap: 10px; align-items: center; color: var(--muted); font-size: 14px; }}
  .summary-list {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
  .summary-box {{ background: rgba(9, 14, 26, .45); border: 1px solid var(--border); border-radius: 14px; padding: 12px; }}
  .summary-box .num {{ font-size: 22px; font-weight: 700; }}
  .muted {{ color: var(--muted); }}
  .ok {{ color: var(--ok); }}
  .warn {{ color: var(--warn); }}
  .danger {{ color: var(--danger); }}
  .footer-note {{ color: var(--muted); font-size: 12px; text-align: center; padding-bottom: 24px; }}
  @media (max-width: 980px) {{
    body {{ overflow: auto; height: auto; }}
    .app {{ grid-template-columns: 1fr; height: auto; }}
    .sidebar {{ height: auto; border-right: 0; border-bottom: 1px solid var(--border); }}
    .summary-list {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .controls-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="sidebar-head">
      <div class="title">Экспертная форма</div>
      <div class="meta" id="meta"></div>
      <div class="progress-wrap">
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
      </div>
    </div>
    <div class="toolbar">
      <div class="toolbar-row">
        <span class="status-pill" id="autosaveStatus">Черновик: не сохранён</span>
      </div>
      <div class="toolbar-row">
        <button id="saveDraftBtn">Скачать черновик</button>
        <button id="loadDraftBtn">Загрузить черновик</button>
        <input id="loadDraftInput" type="file" accept="application/json" style="display:none"/>
      </div>
      <div class="toolbar-row">
        <button id="exportBtn">Скачать итоговый JSON</button>
        <button id="clearDraftBtn">Очистить черновик</button>
      </div>
      <div class="toolbar-row">
        <button id="nextUnratedBtn">Следующий неоценённый</button>
        <select id="filterSelect">
          <option value="all">Показывать: все</option>
          <option value="unrated">Показывать: неоценённые</option>
          <option value="flagged">Показывать: помеченные</option>
        </select>
      </div>
      <div class="toolbar-row">
        <input type="search" id="searchInput" placeholder="Поиск по № / eval_id / case_id" />
      </div>
    </div>
    <div class="sample-list" id="sampleList"></div>
  </aside>

  <main class="content">
    <div class="sticky-nav">
      <div class="nav-left">
        <button id="prevBtn">← Предыдущий</button>
        <button id="nextBtn">Следующий →</button>
        <button id="toggleRefBtn">Показать / скрыть эталон</button>
      </div>
      <div class="nav-right">
        <span class="status-pill" id="positionBadge">Образец 1 / 1</span>
        <span class="status-pill" id="scoreBadge">Оценка: —</span>
      </div>
    </div>

    <div class="content-inner">
      <section class="card">
        <div class="summary-list">
          <div class="summary-box"><div class="muted">Всего</div><div class="num" id="statTotal">0</div></div>
          <div class="summary-box"><div class="muted">Оценено</div><div class="num ok" id="statRated">0</div></div>
          <div class="summary-box"><div class="muted">Осталось</div><div class="num warn" id="statRemaining">0</div></div>
          <div class="summary-box"><div class="muted">Помечено</div><div class="num danger" id="statFlagged">0</div></div>
        </div>
      </section>

      <section class="card">
        <h2 id="sampleTitle">Образец</h2>
        <div class="small-note" id="sampleMeta"></div>
      </section>

      <section class="card">
        <h3>Вопрос</h3>
        <div class="doc" id="questionHtml"></div>
      </section>

      <section class="card">
        <h3>Оцениваемый ответ</h3>
        <div class="doc" id="answerHtml"></div>
      </section>

      <section class="card" id="referenceCard" style="display:none">
        <details id="referenceDetails" open>
          <summary>Эталонный / корректный ответ</summary>
          <div class="doc" id="correctAnswerHtml" style="margin-top:12px"></div>
        </details>
      </section>

      <section class="card">
        <h3>Оценка эксперта</h3>
        <div class="small-note">Шкала от 1 до 10. Горячие клавиши: 1–9, 0 = 10, N = следующий, P = предыдущий, U = следующий неоценённый.</div>
        <div class="rating-grid" id="ratingGrid"></div>
        <div class="rating-hint">Рекомендуем оценивать итоговое качество ответа с учётом корректности, полноты и понятности.</div>
        <div class="controls-grid" style="margin-top:16px">
          <div class="comment-box">
            <label for="commentInput" class="small-note">Комментарий эксперта (необязательно)</label>
            <textarea id="commentInput" placeholder="Короткий комментарий, замечания или причина низкой оценки"></textarea>
          </div>
          <div>
            <label class="checkline"><input type="checkbox" id="flagInput"/> Пометить для перепроверки</label>
          </div>
        </div>
      </section>

      <div class="footer-note">Модельные оценки не отображаются в интерфейсе и попадут только в экспортируемый JSON для последующего сравнения.</div>
    </div>
  </main>
</div>

<script>
const META = {meta_json};
const SAMPLES = {data_json};
const STORAGE_KEY = `sudresh_expert_form_${{META.form_id}}_draft_v1`;
let currentIndex = 0;
let showReference = false;
let state = {{
  responses: {{}},
  autosavedAt: null,
}};

const els = {{
  meta: document.getElementById('meta'),
  progressFill: document.getElementById('progressFill'),
  autosaveStatus: document.getElementById('autosaveStatus'),
  sampleList: document.getElementById('sampleList'),
  filterSelect: document.getElementById('filterSelect'),
  searchInput: document.getElementById('searchInput'),
  prevBtn: document.getElementById('prevBtn'),
  nextBtn: document.getElementById('nextBtn'),
  nextUnratedBtn: document.getElementById('nextUnratedBtn'),
  toggleRefBtn: document.getElementById('toggleRefBtn'),
  saveDraftBtn: document.getElementById('saveDraftBtn'),
  loadDraftBtn: document.getElementById('loadDraftBtn'),
  loadDraftInput: document.getElementById('loadDraftInput'),
  exportBtn: document.getElementById('exportBtn'),
  clearDraftBtn: document.getElementById('clearDraftBtn'),
  positionBadge: document.getElementById('positionBadge'),
  scoreBadge: document.getElementById('scoreBadge'),
  statTotal: document.getElementById('statTotal'),
  statRated: document.getElementById('statRated'),
  statRemaining: document.getElementById('statRemaining'),
  statFlagged: document.getElementById('statFlagged'),
  sampleTitle: document.getElementById('sampleTitle'),
  sampleMeta: document.getElementById('sampleMeta'),
  questionHtml: document.getElementById('questionHtml'),
  answerHtml: document.getElementById('answerHtml'),
  correctAnswerHtml: document.getElementById('correctAnswerHtml'),
  referenceCard: document.getElementById('referenceCard'),
  referenceDetails: document.getElementById('referenceDetails'),
  ratingGrid: document.getElementById('ratingGrid'),
  commentInput: document.getElementById('commentInput'),
  flagInput: document.getElementById('flagInput'),
}};

function defaultResponse(sample) {{
  return {{
    eval_id: sample.eval_id,
    score: null,
    comment: '',
    flagged: false,
    updated_at: null,
  }};
}}

function getResponse(sample) {{
  if (!state.responses[sample.eval_id]) {{
    state.responses[sample.eval_id] = defaultResponse(sample);
  }}
  return state.responses[sample.eval_id];
}}

function ratedCount() {{
  return SAMPLES.filter(s => Number.isInteger(getResponse(s).score)).length;
}}

function flaggedCount() {{
  return SAMPLES.filter(s => getResponse(s).flagged).length;
}}

function updateSummary() {{
  const rated = ratedCount();
  const flagged = flaggedCount();
  const total = SAMPLES.length;
  const remaining = total - rated;
  const pct = total ? Math.round((rated / total) * 100) : 0;
  els.progressFill.style.width = `${{pct}}%`;
  els.statTotal.textContent = String(total);
  els.statRated.textContent = String(rated);
  els.statRemaining.textContent = String(remaining);
  els.statFlagged.textContent = String(flagged);
  els.autosaveStatus.textContent = state.autosavedAt
    ? `Черновик: сохранён локально ${{new Date(state.autosavedAt).toLocaleString()}}`
    : 'Черновик: ещё не сохранён';
}}

function visibleSamples() {{
  const filter = els.filterSelect.value;
  const q = (els.searchInput.value || '').trim().toLowerCase();
  return SAMPLES.filter(sample => {{
    const resp = getResponse(sample);
    if (filter === 'unrated' && Number.isInteger(resp.score)) return false;
    if (filter === 'flagged' && !resp.flagged) return false;
    if (!q) return true;
    const hay = [
      String(sample.form_sample_index),
      sample.eval_id || '',
      sample.case_id || '',
      sample.category || '',
      sample.test_model || '',
    ].join(' ').toLowerCase();
    return hay.includes(q);
  }});
}}

function renderSidebar() {{
  const visible = visibleSamples();
  const currentSample = SAMPLES[currentIndex];
  els.sampleList.innerHTML = '';
  for (const sample of visible) {{
    const resp = getResponse(sample);
    const div = document.createElement('div');
    div.className = 'sample-item';
    if (sample.eval_id === currentSample.eval_id) div.classList.add('active');
    if (Number.isInteger(resp.score)) div.classList.add('rated');
    if (resp.flagged) div.classList.add('flagged');
    const scoreText = Number.isInteger(resp.score) ? `Оценка: ${{resp.score}}` : 'Без оценки';
    div.innerHTML = `
      <div class="sample-top">
        <span>№ ${{sample.form_sample_index}}</span>
        <span>${{scoreText}}</span>
      </div>
      <div class="sample-main">${{sample.case_id || sample.eval_id}}</div>
      <div class="sample-tags">
        ${{sample.is_shared_between_experts ? '<span class="tag">общий</span>' : ''}}
        ${{sample.category ? `<span class="tag">${{escapeHtml(sample.category)}}</span>` : ''}}
        ${{sample.test_model ? `<span class="tag">${{escapeHtml(sample.test_model)}}</span>` : ''}}
      </div>
    `;
    div.addEventListener('click', () => {{
      currentIndex = SAMPLES.findIndex(x => x.eval_id === sample.eval_id);
      renderCurrent();
    }});
    els.sampleList.appendChild(div);
  }}
}}

function renderRatingButtons(score) {{
  els.ratingGrid.innerHTML = '';
  for (let i = 1; i <= 10; i++) {{
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'rating-btn';
    if (i === score) btn.classList.add('active');
    btn.textContent = String(i);
    btn.addEventListener('click', () => setScore(i));
    els.ratingGrid.appendChild(btn);
  }}
}}

function renderCurrent() {{
  const sample = SAMPLES[currentIndex];
  const resp = getResponse(sample);
  els.positionBadge.textContent = `Образец ${{currentIndex + 1}} / ${{SAMPLES.length}}`;
  els.scoreBadge.textContent = `Оценка: ${{Number.isInteger(resp.score) ? resp.score : '—'}}`;
  els.sampleTitle.textContent = `Образец №${{sample.form_sample_index}}`;
  els.sampleMeta.innerHTML = [
    sample.case_id ? `case_id: <strong>${{escapeHtml(sample.case_id)}}</strong>` : '',
    sample.eval_id ? `eval_id: <strong>${{escapeHtml(sample.eval_id)}}</strong>` : '',
    sample.category ? `категория: <strong>${{escapeHtml(sample.category)}}</strong>` : '',
    sample.test_model ? `модель: <strong>${{escapeHtml(sample.test_model)}}</strong>` : '',
    sample.is_shared_between_experts ? '<strong>общий для двух экспертов</strong>' : 'уникальный для формы',
  ].filter(Boolean).join(' · ');
  els.questionHtml.innerHTML = sample.question_html || '<p>—</p>';
  els.answerHtml.innerHTML = sample.answer_html || '<p>—</p>';
  els.correctAnswerHtml.innerHTML = sample.correct_answer_html || '<p>—</p>';
  els.referenceCard.style.display = showReference ? '' : 'none';
  if (showReference) els.referenceDetails.open = true;
  renderRatingButtons(resp.score);
  els.commentInput.value = resp.comment || '';
  els.flagInput.checked = !!resp.flagged;
  renderSidebar();
  updateSummary();
}}

function setScore(score) {{
  const sample = SAMPLES[currentIndex];
  const resp = getResponse(sample);
  resp.score = score;
  resp.updated_at = new Date().toISOString();
  persistLocal();
  renderCurrent();
}}

function updateCurrentFromInputs() {{
  const sample = SAMPLES[currentIndex];
  const resp = getResponse(sample);
  resp.comment = els.commentInput.value || '';
  resp.flagged = !!els.flagInput.checked;
  resp.updated_at = new Date().toISOString();
  persistLocal();
  renderSidebar();
  updateSummary();
}}

function persistLocal() {{
  state.autosavedAt = new Date().toISOString();
  const payload = buildExportPayload(true);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}}

function tryRestoreLocal() {{
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {{
    applyImportedPayload(JSON.parse(raw));
  }} catch (err) {{
    console.warn('Cannot restore local draft', err);
  }}
}}

function buildExportPayload(isDraft) {{
  const responses = SAMPLES.map(sample => {{
    const resp = getResponse(sample);
    return {{
      eval_id: sample.eval_id,
      case_id: sample.case_id,
      form_sample_index: sample.form_sample_index,
      category: sample.category,
      test_model: sample.test_model,
      is_shared_between_experts: !!sample.is_shared_between_experts,
      expert_rating: resp.score,
      expert_comment: resp.comment || '',
      flagged_for_review: !!resp.flagged,
      updated_at: resp.updated_at,
      hidden_model_evaluation: sample.model_evaluation || null,
      hidden_processing_meta: sample.processing || null,
    }};
  }});
  return {{
    schema_version: META.schema_version,
    is_draft: !!isDraft,
    exported_at: new Date().toISOString(),
    form: META,
    summary: {{
      total: SAMPLES.length,
      rated: ratedCount(),
      remaining: SAMPLES.length - ratedCount(),
      flagged: flaggedCount(),
    }},
    responses,
  }};
}}

function applyImportedPayload(payload) {{
  if (!payload || !Array.isArray(payload.responses)) return;
  const byId = Object.create(null);
  for (const item of payload.responses) byId[item.eval_id] = item;
  for (const sample of SAMPLES) {{
    const incoming = byId[sample.eval_id];
    if (!incoming) continue;
    state.responses[sample.eval_id] = {{
      eval_id: sample.eval_id,
      score: Number.isInteger(incoming.expert_rating) ? incoming.expert_rating : null,
      comment: incoming.expert_comment || '',
      flagged: !!incoming.flagged_for_review,
      updated_at: incoming.updated_at || null,
    }};
  }}
  state.autosavedAt = new Date().toISOString();
  persistLocal();
  renderCurrent();
}}

function downloadJson(filename, payload) {{
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json;charset=utf-8' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}}

function exportDraft() {{
  downloadJson(`${{META.form_id}}__draft.json`, buildExportPayload(true));
}}

function exportFinal() {{
  downloadJson(`${{META.form_id}}__results.json`, buildExportPayload(false));
}}

function clearDraft() {{
  if (!confirm('Удалить локальный черновик и очистить все оценки в этой форме?')) return;
  localStorage.removeItem(STORAGE_KEY);
  state = {{ responses: {{}}, autosavedAt: null }};
  renderCurrent();
}}

function nextIndex(step) {{
  currentIndex = Math.max(0, Math.min(SAMPLES.length - 1, currentIndex + step));
  renderCurrent();
}}

function goToNextUnrated() {{
  for (let i = currentIndex + 1; i < SAMPLES.length; i++) {{
    if (!Number.isInteger(getResponse(SAMPLES[i]).score)) {{
      currentIndex = i;
      renderCurrent();
      return;
    }}
  }}
  for (let i = 0; i <= currentIndex; i++) {{
    if (!Number.isInteger(getResponse(SAMPLES[i]).score)) {{
      currentIndex = i;
      renderCurrent();
      return;
    }}
  }}
  alert('Все образцы уже оценены.');
}}

function escapeHtml(text) {{
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}}

els.prevBtn.addEventListener('click', () => nextIndex(-1));
els.nextBtn.addEventListener('click', () => nextIndex(1));
els.nextUnratedBtn.addEventListener('click', goToNextUnrated);
els.toggleRefBtn.addEventListener('click', () => {{ showReference = !showReference; renderCurrent(); }});
els.commentInput.addEventListener('input', updateCurrentFromInputs);
els.flagInput.addEventListener('change', updateCurrentFromInputs);
els.filterSelect.addEventListener('change', renderSidebar);
els.searchInput.addEventListener('input', renderSidebar);
els.saveDraftBtn.addEventListener('click', exportDraft);
els.exportBtn.addEventListener('click', exportFinal);
els.clearDraftBtn.addEventListener('click', clearDraft);
els.loadDraftBtn.addEventListener('click', () => els.loadDraftInput.click());
els.loadDraftInput.addEventListener('change', async (ev) => {{
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  try {{
    const text = await file.text();
    applyImportedPayload(JSON.parse(text));
    alert('Черновик загружен.');
  }} catch (err) {{
    alert('Не удалось загрузить черновик: ' + err);
  }} finally {{
    ev.target.value = '';
  }}
}});

document.addEventListener('keydown', (ev) => {{
  const tag = (ev.target && ev.target.tagName || '').toLowerCase();
  const isTyping = tag === 'textarea' || tag === 'input';
  if (!isTyping) {{
    if (ev.key >= '1' && ev.key <= '9') {{ setScore(parseInt(ev.key, 10)); ev.preventDefault(); return; }}
    if (ev.key === '0') {{ setScore(10); ev.preventDefault(); return; }}
    if (ev.key === 'n' || ev.key === 'N' || ev.key === 'ArrowRight') {{ nextIndex(1); ev.preventDefault(); return; }}
    if (ev.key === 'p' || ev.key === 'P' || ev.key === 'ArrowLeft') {{ nextIndex(-1); ev.preventDefault(); return; }}
    if (ev.key === 'u' || ev.key === 'U') {{ goToNextUnrated(); ev.preventDefault(); return; }}
  }}
  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 's') {{
    exportDraft();
    ev.preventDefault();
  }}
}});

function init() {{
  els.meta.innerHTML = `Эксперт: <strong>${{escapeHtml(META.expert_name)}}</strong><br>Всего образцов: <strong>${{META.total_samples}}</strong><br>Общих образцов: <strong>${{META.shared_samples}}</strong>`;
  tryRestoreLocal();
  renderCurrent();
}}

init();
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
    hf_token: Optional[str]
    hf_direct_only: bool
    num_cases: Optional[int]
    max_records: Optional[int]
    case_sampling: str
    seed: int
    case_ids: Optional[List[str]]
    model: str
    provider_name: Optional[str]
    g4f_api_key: Optional[str]
    concurrency: int
    llm_semaphore: int
    llm_max_retries: int
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_s: Optional[int]
    out: str
    forms_dir: str
    samples_per_expert: int
    shared_samples: int
    expert_a_name: str
    expert_b_name: str
    case_index: Optional[str]


async def run_pipeline(opts: Options) -> int:
    hf_token = install_hf_token_to_env(opts.hf_token)
    if opts.from_hf:
        if hf_token:
            print('[INFO] Hugging Face token detected: authenticated Hub access enabled.', file=sys.stderr)
        else:
            print('[WARN] Hugging Face token not found. Requests to the Hub will be unauthenticated.', file=sys.stderr)
    if opts.from_hf:
        bench_iter = iter_records_hf(opts.benchmark_repo, split=opts.hf_split, streaming=opts.hf_streaming, token=opts.hf_token, direct_only=opts.hf_direct_only)
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

    client = None
    provider = None
    g4f_api_key = resolve_g4f_api_key(opts.g4f_api_key)
    if AsyncClient is None:
        print("[WARN] g4f недоступен: будет использован эвристический фолбэк форматирования.", file=sys.stderr)
    else:
        provider = _resolve_model_provider(opts.model, opts.provider_name)
        if provider is not None:
            print(f"[INFO] g4f provider selected for model {opts.model}: {_provider_display_name(provider)}", file=sys.stderr)
        elif opts.provider_name:
            print(f"[WARN] requested g4f provider was not resolved: {opts.provider_name}", file=sys.stderr)
        if g4f_api_key:
            print("[INFO] g4f api key detected and will be passed to AsyncClient.", file=sys.stderr)
        client = create_g4f_async_client(provider=provider, api_key=g4f_api_key)

    cache: Dict[str, str] = {}
    processed_records: List[Dict[str, Any]] = []
    out_path = Path(opts.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    llm_sem = asyncio.Semaphore(max(1, opts.llm_semaphore))
    if client:
        print(f"[INFO] g4f concurrency gate enabled: max {max(1, opts.llm_semaphore)} in-flight formatting request(s)", file=sys.stderr)
        print(f"[INFO] g4f retries enabled: up to {opts.llm_max_retries} attempts per field", file=sys.stderr)

    async def format_field(field_name: str, text: str, context_text: str) -> Tuple[str, str]:
        key = hashlib.sha1((field_name + "\0" + context_text + "\0" + text).encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key], "cache"

        if not client:
            html_out = heuristic_format_html(text)
            cache[key] = html_out
            return html_out, "heuristic"

        try:
            async with llm_sem:
                html_out = await llm_format_html(
                    client=client,
                    target_text=text,
                    target_name=field_name,
                    context_text=context_text,
                    model=opts.model,
                    provider=provider,
                    max_retries=opts.llm_max_retries,
                    temperature=opts.llm_temperature,
                    max_tokens=opts.llm_max_tokens,
                    request_timeout_s=opts.llm_timeout_s,
                )
            cache[key] = html_out
            return html_out, "g4f"
        except asyncio.CancelledError as e:
            if _is_outer_task_cancelled():
                raise
            print(f"[WARN] Formatting fallback for {field_name}: unexpected CancelledError from provider: {e}", file=sys.stderr)
            html_out = heuristic_format_html(text)
            cache[key] = html_out
            return html_out, "heuristic"
        except Exception as e:
            print(f"[WARN] Formatting fallback for {field_name}: {e}", file=sys.stderr)
            html_out = heuristic_format_html(text)
            cache[key] = html_out
            return html_out, "heuristic"

    if opts.from_hf:
        eval_iter = iter_records_hf(opts.evaluated_repo, split=opts.hf_split, streaming=opts.hf_streaming, limit=opts.max_records, token=opts.hf_token, direct_only=opts.hf_direct_only)
    else:
        eval_iter = iter_records_local(opts.evaluated, inner=opts.evaluated_inner, limit=opts.max_records)

    record_sem = asyncio.Semaphore(max(1, opts.concurrency))

    async def process_one_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        q = rec.get("question") or ""
        a = rec.get("answer") or ""
        ca = rec.get("correct_answer") or ""
        instr_line = extract_instruction_from_question(q)

        key_exact = (instr_line, ca)
        case_id = bench_mappings.key_exact_to_case_id.get(key_exact) or bench_mappings.key_norm_to_case_id.get(
            (normalize_text_for_key(instr_line), normalize_text_for_key(ca))
        )
        if case_id is None or case_id not in allowed_case_ids:
            return None

        rid = str(rec.get("id") or "").strip()
        if not rid:
            rid = hashlib.sha1((instr_line + ca).encode()).hexdigest()

        fields: Dict[str, Dict[str, Any]] = {}
        for field_name, text, ctx in (("question", q, ""), ("answer", a, q), ("correct_answer", ca, q)):
            formatted_html, engine = await format_field(field_name, text, ctx)
            fields[field_name] = {
                "original": text,
                "formatted_html": formatted_html,
                "engine": engine,
            }

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
            "processing": {
                "question_engine": fields["question"]["engine"],
                "answer_engine": fields["answer"]["engine"],
                "correct_answer_engine": fields["correct_answer"]["engine"],
                "target_model": opts.model,
                "target_provider": _provider_display_name(provider) or opts.provider_name,
            },
            "fields": fields,
            "benchmark_source_sha1": bench_mappings.case_id_to_source_sha1.get(case_id),
        }
        if key_exact in bench_mappings.key_exact_to_benchmark_id:
            out["benchmark_record_id"] = bench_mappings.key_exact_to_benchmark_id[key_exact]
        return out

    async def schedule(rec: Dict[str, Any]):
        async with record_sem:
            return await process_one_record(rec)

    tasks: List[asyncio.Task] = []
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
                    if item is None:
                        continue
                    if not first:
                        fout.write(",\n")
                    fout.write(json.dumps(item, ensure_ascii=False))
                    first = False
                    processed_records.append(item)
                    if opts.case_index:
                        case_index[item["case_id"]]["evaluated_records"] += 1
                    if pbar is not None:
                        pbar.update(1)

        if tasks:
            done = await asyncio.gather(*tasks)
            for item in done:
                if item is None:
                    continue
                if not first:
                    fout.write(",\n")
                fout.write(json.dumps(item, ensure_ascii=False))
                first = False
                processed_records.append(item)
                if opts.case_index:
                    case_index[item["case_id"]]["evaluated_records"] += 1
                if pbar is not None:
                    pbar.update(1)

        fout.write("\n]\n")

    if pbar is not None:
        pbar.close()

    forms_dir = Path(opts.forms_dir)
    forms_dir.mkdir(parents=True, exist_ok=True)
    assignments = choose_assignments(
        processed_records=processed_records,
        samples_per_expert=opts.samples_per_expert,
        shared_samples=opts.shared_samples,
        seed=opts.seed,
    )

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    expert_1_samples = assignments["expert_1"]
    expert_2_samples = assignments["expert_2"]

    expert_1_html = build_expert_form_html(
        form_id="expert_form_1",
        expert_name=opts.expert_a_name,
        samples=expert_1_samples,
        shared_count=opts.shared_samples,
        generated_at=generated_at,
    )
    expert_2_html = build_expert_form_html(
        form_id="expert_form_2",
        expert_name=opts.expert_b_name,
        samples=expert_2_samples,
        shared_count=opts.shared_samples,
        generated_at=generated_at,
    )

    form1_path = forms_dir / "expert_form_1.html"
    form2_path = forms_dir / "expert_form_2.html"
    assignments_path = forms_dir / "assignments_manifest.json"

    form1_path.write_text(expert_1_html, encoding="utf-8")
    form2_path.write_text(expert_2_html, encoding="utf-8")
    assignments_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "samples_per_expert": opts.samples_per_expert,
                "shared_samples": opts.shared_samples,
                "expert_a_name": opts.expert_a_name,
                "expert_b_name": opts.expert_b_name,
                "expert_1_eval_ids": [x["eval_id"] for x in expert_1_samples],
                "expert_2_eval_ids": [x["eval_id"] for x in expert_2_samples],
                "shared_eval_ids": assignments["shared"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if opts.case_index:
        ordered = {cid: case_index[cid] for cid in bench_mappings.case_id_order if cid in case_index}
        with open(opts.case_index, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)

    print(f"[INFO] processed records: {len(processed_records)}")
    print(f"[INFO] form 1: {form1_path}")
    print(f"[INFO] form 2: {form2_path}")
    print(f"[INFO] assignments: {assignments_path}")
    return 0


# -----------------------------
# CLI
# -----------------------------
def parse_case_ids_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    if os.path.exists(value):
        p = Path(value)
        if p.suffix.lower() == ".json":
            with open(str(p), "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
            if isinstance(obj, dict):
                return [str(k).strip() for k in obj.keys() if str(k).strip()]
        else:
            ids = []
            with open(str(p), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ids.append(line.strip())
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
    p.add_argument("--hf-token", default=None, help="Hugging Face token. Можно передать и через HF_TOKEN / HUGGINGFACE_HUB_TOKEN.")
    p.add_argument("--hf-direct-only", action="store_true", help="Принудительно читать файлы датасета напрямую через huggingface_hub. Обычно raw .json/.jsonl/.zip и так выбираются автоматически.")
    p.add_argument("--num-cases", type=int, default=None)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--case-sampling", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--case-ids", default=None)
    p.add_argument("--case-index", default=None)

    p.add_argument("--model", default="r1-1776")
    p.add_argument("--provider", default=None, help="g4f.Provider.*. По умолчанию для r1-1776 будет автоматически выбран PerplexityLabs, если он доступен.")
    p.add_argument("--g4f-api-key", default=None, help="Необязательный API key для провайдеров g4f. Также читается из G4F_API_KEY / PUTER_API_KEY / OPENROUTER_API_KEY / GOOGLE_API_KEY.")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--llm-semaphore", type=int, default=1, help="Максимум одновременных g4f-запросов. По умолчанию 1, чтобы снизить риск блокировки provider.")
    p.add_argument("--llm-max-retries", type=int, default=10)
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-max-tokens", type=int, default=2200)
    p.add_argument("--llm-timeout-s", type=int, default=90)

    p.add_argument("--out", required=True)
    p.add_argument("--forms-dir", required=True)
    p.add_argument("--samples-per-expert", type=int, default=400)
    p.add_argument("--shared-samples", type=int, default=100)
    p.add_argument("--expert-a-name", default="Эксперт 1")
    p.add_argument("--expert-b-name", default="Эксперт 2")

    a = p.parse_args(argv)
    if not a.from_hf and (not a.benchmark or not a.evaluated):
        p.error("Specify --benchmark and --evaluated or use --from-hf")

    return Options(
        benchmark=a.benchmark,
        evaluated=a.evaluated,
        benchmark_inner=a.benchmark_inner,
        evaluated_inner=a.evaluated_inner,
        from_hf=a.from_hf,
        benchmark_repo=a.benchmark_repo,
        evaluated_repo=a.evaluated_repo,
        hf_split=a.hf_split,
        hf_streaming=a.hf_streaming,
        hf_token=a.hf_token,
        hf_direct_only=bool(a.hf_direct_only),
        num_cases=a.num_cases,
        max_records=a.max_records,
        case_sampling=a.case_sampling,
        seed=int(a.seed),
        case_ids=parse_case_ids_arg(a.case_ids),
        model=a.model,
        provider_name=a.provider,
        g4f_api_key=a.g4f_api_key,
        concurrency=max(1, int(a.concurrency)),
        llm_semaphore=max(1, int(a.llm_semaphore)),
        llm_max_retries=max(1, int(a.llm_max_retries)),
        llm_temperature=float(a.llm_temperature),
        llm_max_tokens=max(256, int(a.llm_max_tokens)),
        llm_timeout_s=(int(a.llm_timeout_s) if a.llm_timeout_s else None),
        out=a.out,
        forms_dir=a.forms_dir,
        samples_per_expert=max(1, int(a.samples_per_expert)),
        shared_samples=max(0, int(a.shared_samples)),
        expert_a_name=a.expert_a_name,
        expert_b_name=a.expert_b_name,
        case_index=a.case_index,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    opts = parse_args(argv)
    return asyncio.run(run_pipeline(opts))


if __name__ == "__main__":
    raise SystemExit(main())
