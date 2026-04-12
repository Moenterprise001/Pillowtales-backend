from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    durationMin: int = Field(ge=5, le=20)
    storyLanguageCode: str = 'en'
    narrationLanguageCode: Optional[str] = None
    continueFromStoryId: Optional[str] = None
    characters: Optional[List[StoryCharacter]] = None
    customTheme: Optional[str] = None
    companionId: Optional[str] = None
    childNamePronunciation: Optional[str] = None
    gender: str = 'neutral'


class StoryResponse(BaseModel):
    storyId: str
    title: str
    pages: List[str]


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
    is_favorite: bool = False
    created_at: str
