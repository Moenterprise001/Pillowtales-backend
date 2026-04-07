from __future__ import annotations

import re
from typing import List


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


def postprocess_story_pages(pages: List[str]) -> List[str]:
    if not pages:
        return pages
    cleaned_pages = [clean_story_text(page) for page in pages]
    final_page = cleaned_pages[-1].strip()
    if final_page and not final_page.endswith(('.', '!', '?')):
        final_page += '.'
    cleaned_pages[-1] = final_page
    return cleaned_pages


def preview_from_pages(pages: List[str]) -> str:
    first_page = pages[0] if pages else ''
    preview = re.sub(r'\[(whisper|softly|chuckle|pause|gently)\]', '', first_page).strip()
    return preview[:497] + '...' if len(preview) > 500 else preview
