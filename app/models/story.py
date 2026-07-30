from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StoryCharacter(BaseModel):
    name: str
    relationship: str


class GenerateStoryRequest(BaseModel):
    userId: str
    childName: str
    age: int = Field(ge=1, le=12)
    theme: str
    moral: str
    calmLevel: str
    durationMin: int = Field(default=8, ge=5, le=20)
    storyLanguageCode: str = 'en'
    narrationLanguageCode: Optional[str] = None
    continueFromStoryId: Optional[str] = None
    characters: Optional[List[StoryCharacter]] = Field(default=None, max_length=3)
    customTheme: Optional[str] = None
    customMoral: Optional[str] = Field(default=None, max_length=15)
    companionId: Optional[str] = None
    childNamePronunciation: Optional[str] = None
    gender: str = 'neutral'

    @field_validator('customMoral')
    @classmethod
    def validate_custom_moral(cls, value: Optional[str]):
        if value is None:
            return value
        cleaned = value.strip()[:15]
        return cleaned or None

    @field_validator('characters')
    @classmethod
    def validate_characters(cls, value: Optional[List[StoryCharacter]]):
        if value is None:
            return value
        cleaned: List[StoryCharacter] = []
        seen = set()
        for character in value:
            name = (character.name or '').strip()
            relationship = (character.relationship or '').strip()
            if not name or not relationship:
                continue
            key = (name.lower(), relationship.lower())
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(StoryCharacter(name=name, relationship=relationship))
            if len(cleaned) >= 3:
                break
        return cleaned or None


class StoryResponse(BaseModel):
    storyId: str
    title: str
    pages: List[str]
    generation_status: str = 'complete'
    expected_pages: int = 7
    generation_error: Optional[str] = None


class UpdateStoryRequest(BaseModel):
    isFavorite: Optional[bool] = None


class StoryFeedbackRequest(BaseModel):
    rating: Literal['loved_it', 'okay', 'didnt_enjoy']
    feedback: List[str] = Field(default_factory=list, max_length=10)
    wouldLikeSimilarStories: Optional[bool] = None
    childFellAsleep: Optional[bool] = None
    comment: Optional[str] = Field(default=None, max_length=1000)
    narrator: Optional[str] = Field(default=None, max_length=100)
    parentVoiceUsed: Optional[bool] = None

    @field_validator('feedback')
    @classmethod
    def validate_feedback(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        for item in value:
            normalised = str(item or '').strip().lower().replace(' ', '_')[:60]
            if not normalised or normalised in seen:
                continue
            seen.add(normalised)
            cleaned.append(normalised)
            if len(cleaned) >= 10:
                break
        return cleaned

    @field_validator('comment', 'narrator')
    @classmethod
    def clean_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class StoryFeedbackResponse(BaseModel):
    message: str
    feedback: dict


class UserProfileResponse(BaseModel):
    id: str
    email: str
    plan: str
    preferred_language: str
    streak_count: int = 0
    last_story_date: Optional[str] = None
    stories_this_week: int = 0
    stories_saved: int = 0
    can_generate: bool = True
    can_save_more: bool = True


class StoryRecord(BaseModel):
    model_config = ConfigDict(extra='allow')

    id: str
    user_id: str
    title: str
    child_name: str
    age: int
    theme: str
    moral: str
    calm_level: str
    duration_min: int
    language: str
    story_language_code: str = 'en'
    narration_language_code: Optional[str] = None
    child_name_pronunciation: Optional[str] = None
    pages: List[str]
    full_text: str
    generation_status: str = 'complete'
    expected_pages: int = 7
    generation_error: Optional[str] = None
    is_favorite: bool = False
    created_at: str
