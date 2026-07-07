from typing import Any, Dict, List

from .config import CFG
from .llm_cache import cache_key_for_llm, get_llm_cache, set_llm_cache
from .models import get_openai_client
from .text_utils import count_tokens


def format_source(i: int, r: Dict[str, Any]) -> str:
    author_str = ", ".join(r.get("authors") or []) or "Неизвестный автор"
    neighbor_note = "да" if r.get("is_neighbor") else "нет"
    seq = ""
    if r.get("sequence_name"):
        seq = f"\nСерия: {r.get('sequence_name')}"
        if r.get("sequence_number"):
            seq += f", №{r.get('sequence_number')}"

    return (
        f"[Источник {i}]\n"
        f"Книга: {r['book_title']}\n"
        f"Файл: {r['book_file']}\n"
        f"Автор: {author_str}"
        f"{seq}\n"
        f"Глава/раздел: {r['chapter']}\n"
        f"Чанк в книге: {r['book_chunk_number']}\n"
        f"Соседний чанк: {neighbor_note}\n"
        f"Текст: {r['text']}"
    )


def trim_context_results_by_formatted_tokens(
    context_results: List[Dict[str, Any]],
    max_tokens: int,
) -> List[Dict[str, Any]]:
    final: List[Dict[str, Any]] = []
    for candidate in context_results:
        trial = final + [candidate]
        formatted_context = "\n\n".join(format_source(i, r) for i, r in enumerate(trial, start=1))
        if final and count_tokens(formatted_context) > max_tokens:
            break
        final = trial
    return final


def ask_llm(query: str, context_results: List[Dict[str, Any]]) -> str:
    if not context_results:
        return "В найденных фрагментах этого нет."

    context_results = trim_context_results_by_formatted_tokens(context_results, CFG.max_context_tokens)
    if not context_results:
        return "В найденных фрагментах этого нет."

    llm_cache_key = cache_key_for_llm(query, context_results)
    cached = get_llm_cache(llm_cache_key)
    if cached is not None:
        return cached

    context = "\n\n".join(format_source(i, r) for i, r in enumerate(context_results, start=1))

    system_prompt = (
        "Ты аккуратный помощник по серии книг. "
        "Отвечай только на основе предоставленного контекста. "
        "Не добавляй факты из памяти и не выдумывай."
    )

    user_prompt = f"""
Правила ответа:
1. Используй только контекст ниже.
2. Если ответа нет в контексте, напиши: "В найденных фрагментах этого нет".
3. Если ответ частичный, явно скажи, что известно только частично.
4. Если источники противоречат друг другу, укажи противоречие.
5. Отвечай по-русски, кратко, но достаточно понятно.
6. В конце укажи источники, например: [Источник 1], [Источник 3].

Контекст:
{context}

Вопрос:
{query}

Ответ:
""".strip()

    response = get_openai_client().chat.completions.create(
        model=CFG.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.15,
    )

    answer = response.choices[0].message.content.strip()
    set_llm_cache(llm_cache_key, answer)
    return answer
