from __future__ import annotations

from typing import Any, Dict

SUPPORTED_LANGUAGES: Dict[str, str] = {
    'en': 'English (British)',
    'es': 'Spanish (Castilian/Spain)',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
}

VOICE_PRESETS: Dict[str, Dict[str, Any]] = {
    'wise_owl': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Wise Owl', 'description': 'Calm British bedtime narration, gentle and reassuring', 'icon': '🦉', 'language_code': 'en', 'tier': 'free'},
    'parent_voice': {'provider': 'elevenlabs', 'voice_id': None, 'name': 'Parent Voice', 'description': 'Your own voice reads stories', 'icon': '❤️', 'language_code': 'all', 'tier': 'add_on', 'requires_setup': True, 'purchase_required_each_story': True, 'price_eur': 2.0, 'bundle3_price_eur': 4.99},
    'night_owl_spanish': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Búho Sabio', 'description': 'Suave voz española para dormir', 'icon': '🦉', 'language_code': 'es', 'tier': 'free'},
    'night_owl_german': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Weise Eule', 'description': 'Sanfte deutsche Stimme zum Einschlafen', 'icon': '🦉', 'language_code': 'de', 'tier': 'free'},
    'night_owl_french': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Hibou Sage', 'description': 'Douce voix française pour dormir', 'icon': '🦉', 'language_code': 'fr', 'tier': 'free'},
    'night_owl_italian': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Gufo Saggio', 'description': 'Dolce voce italiana per dormire', 'icon': '🦉', 'language_code': 'it', 'tier': 'free'},
}

STORY_COMPANIONS: Dict[str, Dict[str, Any]] = {
    'luna_owl': {'name': 'Luna the Moon Owl', 'short_name': 'Luna', 'icon': '🦉', 'description': 'A wise little owl who glows softly in moonlight', 'tier': 'free'},
    'milo_fox': {'name': 'Milo the Sleepy Fox', 'short_name': 'Milo', 'icon': '🦊', 'description': 'A cozy fox who knows all the best sleeping spots', 'tier': 'free'},
    'spark_dragon': {'name': 'Spark the Tiny Dragon', 'short_name': 'Spark', 'icon': '🐉', 'description': 'A palm-sized dragon who breathes warm, sparkly light', 'tier': 'premium'},
    'stella_fairy': {'name': 'Stella the Star Fairy', 'short_name': 'Stella', 'icon': '✨', 'description': 'A tiny fairy who sprinkles sleepy stardust', 'tier': 'premium'},
    'bramble_bear': {'name': 'Bramble the Gentle Bear', 'short_name': 'Bramble', 'icon': '🐻', 'description': 'A soft, cuddly bear who gives the best hugs', 'tier': 'premium'},
}

SUBSCRIPTION_TIERS: Dict[str, Dict[str, Any]] = {
    'free': {
        'weekly_story_limit': 2,
        'weekly_narration_limit': 2,
        'max_saved_stories': 10,
        'narrators': ['wise_owl', 'night_owl_spanish', 'night_owl_german', 'night_owl_french', 'night_owl_italian'],
        'companions': ['luna_owl', 'milo_fox'],
        'parent_voice': True,
    },
    'premium': {
        'weekly_story_limit': None,
        'weekly_narration_limit': None,
        'max_saved_stories': None,
        'narrators': list(VOICE_PRESETS.keys()),
        'companions': list(STORY_COMPANIONS.keys()),
        'parent_voice': True,
    },
}

TESTER_EMAILS = {
    'qa@pillowtales.app', 'test@pillowtales.app', 'qa@pillowtales.co', 'test@pillowtales.co',
    'logintest@pillowtales.app', 'dev@pillowtales.app', 'dev@pillowtales.co',
    'support@pillowtales.co', 'hello@pillowtales.co',
}
