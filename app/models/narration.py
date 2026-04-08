from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

VoiceMode = Literal['standard', 'parent', 'fallback_tts']
NarrationStatus = Literal['page_ready', 'generating', 'all_ready', 'failed']


class NarrationRequest(BaseModel):
    storyId: str
    narrationLanguageCode: Optional[str] = None
    voicePreference: Optional[str] = None


class NarrationResponse(BaseModel):
    status: NarrationStatus
    audioUrl: Optional[str] = None
    message: Optional[str] = None
    jobId: Optional[str] = None
    currentPage: Optional[int] = None
    totalPages: Optional[int] = None
    pageAudioUrl: Optional[str] = None
    pagesReady: List[int] = Field(default_factory=list)
    voice_mode: Optional[VoiceMode] = None


class PageStatusResponse(BaseModel):
    storyId: str
    totalPages: int
    pagesReady: List[int] = Field(default_factory=list)
    pagesGenerating: List[int] = Field(default_factory=list)
    pagesFailed: List[int] = Field(default_factory=list)
    allReady: bool
    voice_mode: Optional[VoiceMode] = None
