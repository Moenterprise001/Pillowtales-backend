from __future__ import annotations

from typing import List, Optional

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
    companionId: Optional[str] = None
    childNamePronunciation: Optional[str] = None
    gender: str = 'neutral'

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
