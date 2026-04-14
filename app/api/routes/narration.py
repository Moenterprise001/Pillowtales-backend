from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

VoiceMode = Literal['standard', 'parent', 'fallback_tts']
NarrationStatus = Literal['page_ready', 'generating', 'all_ready', 'failed']


class NarrationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Accept BOTH old camelCase and new snake_case payloads
    story_id: str
    narration_language_code: Optional[str] = Field(default=None, alias='narrationLanguageCode')
    voice_preference: Optional[str] = Field(default=None, alias='voicePreference')

    # New frontend/internal fields
    narrator: Optional[str] = None
    lang: Optional[str] = None
    child_name_pronunciation: Optional[str] = None

    # Backward-compatible properties so existing service code
    # can keep using request.storyId / request.narrationLanguageCode / request.voicePreference
    @property
    def storyId(self) -> str:
        return self.story_id

    @property
    def narrationLanguageCode(self) -> Optional[str]:
        return self.lang or self.narration_language_code

    @property
    def voicePreference(self) -> Optional[str]:
        return self.narrator or self.voice_preference


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