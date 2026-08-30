from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StoryWorldRegion(BaseModel):
    code: str
    name: str


class StoryWorldCountry(BaseModel):
    code: str
    name: str
    heroUrl: Optional[str] = None


class StoryWorldArtwork(BaseModel):
    coverUrl: Optional[str] = None
    thumbnailUrl: Optional[str] = None
    iconUrl: Optional[str] = None


class StoryWorldPresentation(BaseModel):
    primaryColour: Optional[str] = None
    secondaryColour: Optional[str] = None
    sortOrder: int = 100


class StoryWorldAvailability(BaseModel):
    enabled: bool
    published: bool
    comingSoon: bool


class StoryWorldAgeRange(BaseModel):
    min: int = Field(ge=1, le=12)
    max: int = Field(ge=1, le=12)


class StoryWorldPublic(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    slug: str
    name: str
    shortDescription: str
    description: str
    region: StoryWorldRegion
    countries: List[StoryWorldCountry] = Field(default_factory=list)
    peoples: List[str] = Field(default_factory=list)
    traditions: List[str] = Field(default_factory=list)
    category: str
    worldType: str
    ageRange: StoryWorldAgeRange
    artwork: StoryWorldArtwork
    presentation: StoryWorldPresentation
    availability: StoryWorldAvailability
    supportedLanguages: List[str] = Field(default_factory=list)
    version: int = 1
    updatedAt: str


class StoryWorldListResponse(BaseModel):
    storyWorlds: List[StoryWorldPublic]
    count: int


class StoryWorldCanonCollection(BaseModel):
    """Public Canon collection metadata used to group original folklore stories."""
    slug: str
    title: str
    description: Optional[str] = None
    sortOrder: int = 100
    artwork: StoryWorldArtwork


class StoryWorldCanonSeries(BaseModel):
    """Public Canon series metadata for multi-part original folklore stories."""
    slug: str
    title: str
    summary: Optional[str] = None
    partCount: int
    sortOrder: int = 100
    chronologyGroup: Optional[str] = None
    chronologyOrder: Optional[int] = None
    artwork: StoryWorldArtwork


class StoryWorldCanonStorySource(BaseModel):
    """Public original folklore story available for a protected Canon retelling."""
    slug: str
    title: str
    summary: str
    ageRange: StoryWorldAgeRange
    artwork: StoryWorldArtwork
    coreValues: List[str] = Field(default_factory=list)
    country: Optional[StoryWorldCountry] = None

    collection: Optional[StoryWorldCanonCollection] = None
    series: Optional[StoryWorldCanonSeries] = None

    partNumber: Optional[int] = None
    partTitle: Optional[str] = None
    sortOrder: int = 100
    chronologyGroup: Optional[str] = None
    chronologyOrder: Optional[int] = None


class StoryWorldCanonCountry(BaseModel):
    """Country grouping for Story Worlds that organise Canon by country."""
    code: str
    name: str
    heroUrl: Optional[str] = None
    stories: List[StoryWorldCanonStorySource] = Field(default_factory=list)
    count: int = 0


class StoryWorldCanonStoryListResponse(BaseModel):
    storyWorldSlug: str
    stories: List[StoryWorldCanonStorySource]
    countries: List[StoryWorldCanonCountry] = Field(default_factory=list)
    count: int


class StoryWorldAdventureSource(BaseModel):
    """Public folklore source available for a PillowTales Folk Adventure."""
    slug: str
    title: str
    summary: str
    ageRange: StoryWorldAgeRange
    artwork: StoryWorldArtwork
    coreValues: List[str] = Field(default_factory=list)


class StoryWorldAdventureListResponse(BaseModel):
    storyWorldSlug: str
    adventures: List[StoryWorldAdventureSource]
    count: int
