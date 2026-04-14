from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

VoiceMode = Literal['standard', 'parent', 'fallback_tts']
NarrationStatus = Literal['page_ready', 'generating', 'all_ready', 'failed']


class NarrationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Accept both old camelCase and new snake_case/internal payloads
    story_id: str = Field(validation_alias=AliasChoices('story_id', 'storyId'))
    narration_language_code: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('narration_language_code', 'narrationLanguageCode', 'lang'),
    )
    voice_preference: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('voice_preference', 'voicePreference', 'narrator'),
    )
    child_name_pronunciation: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('child_name_pronunciation', 'childNamePronunciation'),
    )

    # Backward-compatible properties for existing service code
    @property
    def storyId(self) -> str:
        return self.story_id

    @property
    def narrationLanguageCode(self) -> Optional[str]:
        return self.narration_language_code

    @property
    def voicePreference(self) -> Optional[str]:
        return self.voice_preference

    @property
    def narrator(self) -> Optional[str]:
        return self.voice_preference

    @property
    def lang(self) -> Optional[str]:
        return self.narration_language_code


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
