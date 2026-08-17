# PillowTales Story World Contract v1.0

**Status:** Proposed for freeze  
**Scope:** Database, backend, API, frontend, admin, Story World content packages  
**Reference implementation:** Irish Legends  
**Compatibility target:** 500+ Story Worlds, five current languages, future language expansion without schema redesign

---

## 1. Purpose

This contract defines the permanent shape and behaviour of a PillowTales Story World.

It is the shared agreement between:

- Supabase schema
- FastAPI backend
- Story World resolver
- Existing PillowTales story engine
- Frontend discovery screens
- Admin management
- Story World content packages

The Story Engine must never contain knowledge of Ireland, Japan, India, Egypt, the Americas, Africa, Oceania, Christmas, Halloween, or any individual Story World.

The engine owns behaviour. Story Worlds own identity.

---

## 2. Frozen Product Behaviour

PillowTales has exactly three story-entry modes:

1. **Tonight's Surprise Story**
   - Parent does not choose a Story World.
   - Backend selects one enabled Story World using monthly weights.
   - If no eligible Story World exists, current non-Story-World generation continues unchanged.

2. **Create Your Own Story**
   - Parent provides the story idea.
   - Story World is not imposed automatically.
   - A future optional Story World selection may be supported, but it is not required for v1.

3. **Discover Story Worlds**
   - Parent deliberately selects a Story World.
   - Every enabled and published Story World remains available all year.
   - Monthly weighting never hides or disables a Story World in Discover mode.

---

## 3. Story World Public Object

This is the object returned to the frontend.

```json
{
  "id": "uuid",
  "slug": "irish-legends",
  "name": "Irish Legends",
  "shortDescription": "Magical adventures inspired by Ireland's folklore, legends and myths.",
  "description": "Step into ancient forests, fairy rings and legendary adventures inspired by Ireland's storytelling traditions.",
  "region": {
    "code": "europe",
    "name": "Europe"
  },
  "countries": [
    {
      "code": "IE",
      "name": "Ireland"
    }
  ],
  "peoples": [],
  "traditions": ["Celtic folklore", "Fenian Cycle"],
  "category": "folklore",
  "worldType": "hybrid",
  "ageRange": {
    "min": 1,
    "max": 12
  },
  "artwork": {
    "coverUrl": null,
    "thumbnailUrl": null,
    "iconUrl": null
  },
  "presentation": {
    "primaryColour": null,
    "secondaryColour": null,
    "sortOrder": 100
  },
  "availability": {
    "enabled": true,
    "published": true,
    "comingSoon": false
  },
  "supportedLanguages": ["en", "es", "fr", "de", "it"],
  "version": 1,
  "updatedAt": "ISO-8601 timestamp"
}
```

### Public object rules

- It contains no full editorial bible.
- It contains no system prompts.
- It contains no private research notes.
- It contains no monthly weights unless an admin endpoint requests them.
- It contains only presentation and selection data needed by the app.

---

## 4. Internal Story World Object

The backend and admin system use the full internal object.

```json
{
  "id": "uuid",
  "slug": "irish-legends",
  "worldType": "hybrid",
  "category": "folklore",
  "regionCode": "europe",
  "countryCodes": ["IE"],
  "peoples": [],
  "traditions": ["Celtic folklore", "Fenian Cycle"],
  "ageMin": 1,
  "ageMax": 12,
  "storyMix": {
    "livingWorld": 70,
    "companion": 20,
    "legendaryMoment": 10
  },
  "monthlyWeights": {
    "1": 10,
    "2": 10,
    "3": 100,
    "4": 20,
    "5": 20,
    "6": 20,
    "7": 20,
    "8": 20,
    "9": 20,
    "10": 20,
    "11": 20,
    "12": 20
  },
  "editorialBibleVersion": 1,
  "storyDnaVersion": 1,
  "promptPackVersion": 1,
  "enabled": true,
  "published": true,
  "comingSoon": false,
  "createdAt": "ISO-8601 timestamp",
  "updatedAt": "ISO-8601 timestamp"
}
```

---

## 5. Cultural Hierarchy

A Story World must not be forced into a single-country model.

The contract supports:

- Region
- Zero or more countries
- Zero or more peoples/cultures
- Zero or more storytelling traditions

Examples:

- **Irish Legends** — Europe / Ireland / Celtic folklore
- **Guarani Forest Legends** — South America / Paraguay, Brazil, Argentina / Guarani
- **Panchatantra Adventures** — Asia / India / classical animal fables
- **Ancient Egypt** — Africa / Egypt / ancient Egyptian mythology
- **Arabian Desert Legends** — Middle East / multiple countries / Arabic and Bedouin storytelling traditions
- **Polynesian Ocean Legends** — Oceania / multiple island nations / multiple Polynesian peoples

No culture is reduced to a generic continent-level Story World unless that scope is deliberately researched and approved.

---

## 6. Narrative Types

Every Story World supports the same three narrative types.

### 6.1 Living World

- Default and most common type.
- Child is the main hero.
- Original adventure set in the Story World.
- Famous legendary characters may be absent, mentioned, or appear briefly.

### 6.2 Companion

- Legendary figure remains central.
- Child travels with, helps, observes, or supports the legendary figure.
- Child never replaces the legendary figure.
- No direct questions or forced decisions are addressed to the child.

### 6.3 Legendary Moment

- Rare.
- Child experiences a known legendary moment from within the world.
- Defining events are not rewritten.
- Prior knowledge is never required.
- Necessary context is introduced naturally in one or two simple sentences.

### Story mix validation

The three percentages must total 100.

Default reference mix:

```json
{
  "livingWorld": 70,
  "companion": 20,
  "legendaryMoment": 10
}
```

A Story World may use a different mix through data. No code branch is allowed for a specific world.

---

## 7. Monthly Weighting

Monthly weighting applies only to **Tonight's Surprise Story**.

Rules:

- One weight per month, 1–12.
- Weight must be an integer from 0 to 1000.
- Weight 0 means not eligible for automatic selection that month.
- A published Story World with weight 0 remains available in Discover mode.
- Selection is weighted among enabled, published, age-compatible Story Worlds.
- There is no hardcoded March/Ireland logic.
- There is no hardcoded October/Halloween logic.
- There is no hardcoded December/Christmas logic.

The resolver accepts a month as an input for deterministic testing. Production defaults to the current application date.

---

## 8. Language Contract

Current Story World languages:

- English (`en`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Italian (`it`)

The schema must use translation rows, not language-specific columns.

### Translatable content

- Name
- Short description
- Full description
- Editorial guidance where language-specific wording is required
- Prompt pack content
- Display labels

### Fallback rule

1. Requested language
2. English
3. Story World is unavailable only if no valid published prompt pack can be resolved

Adding Portuguese, Dutch, Japanese, Chinese, Arabic, Hindi, or another language must require new data only—not schema changes.

---

## 9. Editorial Bible Contract

Every published Story World must have one active editorial bible version.

Required sections:

- Purpose
- Cultural scope
- Intended feeling
- Tone
- Atmosphere
- Core values
- Humour
- Dialogue
- Nature
- Magic
- Pacing
- Ending style
- Bedtime suitability
- Cultural respect rules
- Prior-knowledge accessibility rules
- Things never to do

The editorial bible is the human-readable source of truth.

It is not returned through public APIs.

---

## 10. Story DNA Contract

Every published Story World must have one active Story DNA version.

Required structured groups:

- Legendary figures
- Supporting characters
- Everyday people
- Creatures and spirits
- Landscapes
- Settlements
- Buildings and places
- Nature
- Animals and birds
- Objects
- Foods where appropriate
- Music and sounds
- Festivals and traditions
- Magic systems
- Core themes
- Story opportunities
- Excluded or restricted elements
- Bedtime adaptation guidance

Story DNA contains ingredients, not fixed stories.

The Golden Rule:

> A Story World is complete because it has enough ingredients to create unlimited stories—not because it contains a fixed number of stories.

---

## 11. Prompt Pack Contract

Prompt packs are versioned and language-specific.

Each active prompt pack must contain:

- World context block
- Narrative-type rules
- Cultural-respect block
- Bedtime adaptation block
- Positive story ingredients
- Negative constraints
- Accessibility context rule
- Ending guidance

Prompt packs do not replace the existing PillowTales story-quality engine.

They are merged into the existing generation prompt as an isolated Story World context block.

They must not override:

- Reading-age rules
- Oxford Reading Tree benchmark
- Page-1-first generation
- Ending validation
- Safety rules
- Story duration rules
- Language rules
- Narration rules

---

## 12. Generation Request Contract

The existing `GenerateStoryRequest` is extended additively.

```json
{
  "storyMode": "surprise | custom | discover",
  "storyWorldSlug": "irish-legends | null"
}
```

### Rules

- `surprise`: `storyWorldSlug` must be absent or ignored; backend resolves the world.
- `custom`: `storyWorldSlug` is absent in v1; existing behaviour remains unchanged.
- `discover`: `storyWorldSlug` is required and must resolve to an enabled, published Story World.
- Requests without the new fields retain current behaviour during rollout.

No existing request field is removed or renamed.

---

## 13. Generation Resolution Contract

```text
Request
  ↓
Resolve story mode
  ↓
Resolve Story World (or none)
  ↓
Resolve narrative type using Story World mix
  ↓
Resolve requested-language prompt pack, fallback to English
  ↓
Compile isolated Story World context
  ↓
Pass context to existing story generator
  ↓
Save Story World metadata with story
```

If Story World resolution fails in Surprise mode, current story generation continues without a Story World.

If resolution fails in Discover mode, the request returns a clear 404/422 and must not silently generate a different world.

---

## 14. Story Persistence Contract

Generated stories store immutable generation metadata:

- `story_mode`
- `story_world_id`
- `story_world_slug`
- `story_world_version`
- `story_world_prompt_pack_version`
- `story_world_narrative_type`

This enables:

- Library display
- Feedback analysis
- Quality comparison
- Reproduction/debugging
- Admin analytics
- Future “more like this” behaviour

These fields are additive. Existing stories remain valid with null Story World metadata.

---

## 15. Public API Contract

### `GET /api/story-worlds`

Returns enabled, published Story Worlds available for Discover mode.

Optional filters:

- `language`
- `region`
- `category`
- `age`

### `GET /api/story-worlds/{slug}`

Returns one public Story World object.

### `GET /api/story-worlds/featured`

Returns Story Worlds relevant to a month for presentation only.

This endpoint does not perform the final Surprise selection.

### Admin APIs

Admin APIs are separate and authenticated. They may expose:

- Draft worlds
- Monthly weights
- Editorial bible versions
- Story DNA versions
- Prompt pack versions
- Publication controls

---

## 16. Publication Contract

A Story World may be:

- Draft
- Coming soon
- Published
- Disabled
- Archived

A world is eligible for Discover only when:

- enabled = true
- published = true
- coming_soon = false
- a valid translation exists
- a valid active prompt pack exists
- age range matches the child

A world is eligible for Surprise only when all Discover rules pass and its current monthly weight is greater than zero.

---

## 17. Frontend Contract

Home screen adds exactly one new card:

**🌍 Discover Story Worlds**  
*Magical adventures inspired by the world's folklore, legends and myths.*

The frontend must:

- Load worlds from the API
- Never contain a hardcoded Ireland list
- Never contain a hardcoded region list that blocks new regions
- Render API-provided artwork and translated text
- Pass `storyMode=discover` and `storyWorldSlug=<slug>` during generation

Tonight's Surprise Story passes `storyMode=surprise`.

Create Your Own Story passes `storyMode=custom` or omits the field during compatibility rollout.

---

## 18. Engine Protection Rules

Story Worlds must not change or refactor:

- Page-1-first story generation
- Background completion of pages 2+
- Narration request flow
- Chunk generation
- Audio storage paths
- Playback locking
- Text polling as final page source of truth
- Share-card final-page conditions
- Feedback flow
- Keep-awake behaviour
- Existing ending engine

The Story World integration point is limited to prompt-context resolution and Story World metadata persistence.

---

## 19. No-Hardcoding Rule

The following are prohibited:

```python
if world == "irish-legends":
    ...
```

```typescript
const STORY_WORLDS = ["Ireland", "Japan", "Egypt"];
```

```python
if month == 3:
    return "irish-legends"
```

Allowed hardcoded values are platform enums and validation rules only, such as:

- Story modes
- Narrative types
- Publication states
- Supported contract versions
- Generic validation limits

Individual Story World content, availability, weighting, display, prompts, culture, or artwork must come from data.

---

## 20. Irish Legends Reference Object

Ireland validates the contract but does not alter it.

```json
{
  "slug": "irish-legends",
  "regionCode": "europe",
  "countryCodes": ["IE"],
  "peoples": [],
  "traditions": ["Celtic folklore", "Fenian Cycle"],
  "category": "folklore",
  "worldType": "hybrid",
  "ageMin": 1,
  "ageMax": 12,
  "storyMix": {
    "livingWorld": 70,
    "companion": 20,
    "legendaryMoment": 10
  },
  "monthlyWeights": {
    "1": 10,
    "2": 10,
    "3": 100,
    "4": 20,
    "5": 20,
    "6": 20,
    "7": 20,
    "8": 20,
    "9": 20,
    "10": 20,
    "11": 20,
    "12": 20
  },
  "enabled": true,
  "published": true,
  "comingSoon": false
}
```

The exact weights remain content configuration and may be adjusted through admin without a code release.

---

## 21. Acceptance Criteria for Contract Freeze

The contract is accepted only if it can represent without redesign:

- Ireland
- India and multiple Indian traditions
- Ancient Egypt
- Arabia and other Middle Eastern traditions
- Japan
- Distinct African peoples and regions
- Indigenous North American traditions
- Central America and Caribbean traditions
- Amazonian, Andean, Guarani, Tupi, Mapuche and other South American traditions
- Māori, Polynesian, Melanesian and Micronesian traditions
- Seasonal Story Worlds
- PillowTales Originals
- Five current languages
- Future languages
- 500+ Story Worlds

---

## 22. Implementation Sequence After Freeze

1. Create additive Supabase migration.
2. Add Story World domain models.
3. Add repository.
4. Add resolver and context compiler.
5. Add public Story World API.
6. Seed Irish Legends as data.
7. Extend generation request additively.
8. Integrate context into the existing story service at one isolated point.
9. Persist Story World metadata.
10. Test existing generation and narration regression suite.
11. Add Discover Story Worlds frontend.
12. Test English, Spanish, French, German and Italian.

---

## 23. Change Governance

This contract may be changed only when evidence shows one of the following:

1. It cannot be implemented reliably.
2. It prevents scaling to hundreds of Story Worlds.
3. It reduces story quality, safety, accessibility, or the bedtime experience.

Preference, code-style cleanup, a different framework, or a desire to redesign are not sufficient reasons.
