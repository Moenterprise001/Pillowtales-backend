from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class NarrationRequest(BaseModel):
    storyId: str
    narrationLanguageCode: Optional[str] = None
    voicePreference: Optional[str] = None


class NarrationResponse(BaseModel):
    status: str
    audioUrl: Optional[str] = None
    message: Optional[str] = None
