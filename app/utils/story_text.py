from __future__ import annotations

import re
from typing import List


_SENTENCE_ENDINGS = ('.', '!', '?', '."', '!"', '?"', ".'", "!'", "?'", '.”', '!”', '?”', ".’", "!’", "?’")
_FRAGMENT_STARTERS = {
    "and",
    "but",
    "because",
    "which",
    "that",
    "where",
    "when",
    "while",
    "until",
    "then",
    "so",
    "or",
    "nor",
    "for",
    "yet",
    "with",
    "without",
    "inside",
    "outside",
    "under",
    "over",
    "beside",
    "between",
    "through",
    "toward",
    "towards",
    "across",
    "around",
    "behind",
    "before",
    "after",
    "into",
    "onto",
    "from",
    "near",
    "location",
}


def clean_story_text(text: str) -> str:
    if not text:
        return text
    cleaned = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    for pattern in [
        r'(?im)^\s*the end\.?\s*$',
        r'(?im)^\s*fin\.?\s*$',
        r'(?im)^\s*finis\.?\s*$',
        r'(?im)^\s*ende\.?\s*$',
        r'(?im)^\s*fine\.?\s*$',
    ]:
        cleaned = re.sub(pattern, '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    deduped: List[str] = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    return '\n'.join(deduped).strip()


def _first_visible_word(text: str) -> str:
    match = re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text or "")
    return match.group(0).lower() if match else ""


def _starts_like_fragment(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False

    if stripped.startswith(("...", "…", ",", ";", ":", "-", "—", ")", "]")):
        return True

    first_char = stripped[0]
    # Lowercase starts often indicate a sentence fragment in English/French/Spanish/Italian/German.
    if first_char.isalpha() and first_char == first_char.lower() and first_char != first_char.upper():
        return True

    return _first_visible_word(stripped) in _FRAGMENT_STARTERS


def _ends_with_sentence_boundary(text: str) -> bool:
    stripped = (text or "").strip()
    return not stripped or stripped.endswith(_SENTENCE_ENDINGS)


def log_page_boundary_warnings(pages: List[str], source: str = "story") -> None:
    """Log suspicious page boundaries without changing story text.

    This is intentionally non-mutating. Page count, narration timing, sync,
    polling, and share-card behaviour must not be affected by validation.
    """
    if not pages:
        return

    for index, page in enumerate(pages):
        page_number = index + 1
        stripped = (page or "").strip()
        if not stripped:
            print(f"[STORY_BOUNDARY_WARN] source={source} page={page_number} reason=empty_page")
            continue

        if index > 0 and _starts_like_fragment(stripped):
            preview = stripped[:90].replace("\n", " ")
            print(
                f"[STORY_BOUNDARY_WARN] source={source} page={page_number} "
                f"reason=fragment_start preview={preview!r}"
            )

        if index > 0:
            previous = (pages[index - 1] or "").strip()
            if previous and not _ends_with_sentence_boundary(previous):
                preview = previous[-90:].replace("\n", " ")
                print(
                    f"[STORY_BOUNDARY_WARN] source={source} page={page_number - 1} "
                    f"reason=previous_page_no_sentence_boundary preview={preview!r}"
                )


def postprocess_story_pages(pages: List[str], source: str = "story") -> List[str]:
    if not pages:
        return pages
    cleaned_pages = [clean_story_text(page) for page in pages]
    final_page = cleaned_pages[-1].strip()
    if final_page and not final_page.endswith(('.', '!', '?')):
        final_page += '.'
    cleaned_pages[-1] = final_page

    # Phase 10E: log suspicious page boundaries only.
    # Do not auto-merge or rewrite pages here because that can affect narration,
    # page count, polling, share-card conditions, and reader sync.
    log_page_boundary_warnings(cleaned_pages, source=source)
    return cleaned_pages


def preview_from_pages(pages: List[str]) -> str:
    first_page = pages[0] if pages else ''
    preview = re.sub(r'\[(whisper|softly|chuckle|pause|gently)\]', '', first_page).strip()
    return preview[:497] + '...' if len(preview) > 500 else preview
