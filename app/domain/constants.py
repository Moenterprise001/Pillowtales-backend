from __future__ import annotations

from typing import Any, Dict

SUPPORTED_LANGUAGES: Dict[str, str] = {
    'en': 'English',
    'es': 'Spanish (Castilian/Spain)',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
}

VOICE_PRESETS: Dict[str, Dict[str, Any]] = {
    'wise_owl': {'provider': 'openai', 'voice_id': 'verse', 'name': 'Wise Owl', 'description': 'Calm British bedtime narration, gentle and reassuring', 'icon': '🦉', 'language_code': 'en', 'tier': 'free'},
    'night_owl_english': {'provider': 'openai', 'voice_id': 'sage', 'name': 'Night Owl', 'description': 'Warm American bedtime narration, calm and reassuring', 'icon': '🦉', 'language_code': 'en', 'tier': 'free'},
    'parent_voice': {'provider': 'elevenlabs', 'voice_id': None, 'name': 'Parent Voice', 'description': 'Your own voice reads stories', 'icon': '❤️', 'language_code': 'all', 'tier': 'add_on', 'requires_setup': True, 'purchase_required_each_story': True, 'price_eur': 2.0, 'bundle3_price_eur': 4.99},
    'night_owl_spanish': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Búho Sabio', 'description': 'Suave voz española para dormir', 'icon': '🦉', 'language_code': 'es', 'tier': 'free'},
    'night_owl_german': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Weise Eule', 'description': 'Sanfte deutsche Stimme zum Einschlafen', 'icon': '🦉', 'language_code': 'de', 'tier': 'free'},
    'night_owl_french': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Hibou Sage', 'description': 'Douce voix française pour dormir', 'icon': '🦉', 'language_code': 'fr', 'tier': 'free'},
    'night_owl_italian': {'provider': 'openai', 'voice_id': 'shimmer', 'name': 'Gufo Saggio', 'description': 'Dolce voce italiana per dormire', 'icon': '🦉', 'language_code': 'it', 'tier': 'free'},
}

STORY_COMPANIONS: Dict[str, Dict[str, Any]] = {
    'luna_owl': {
        'name': 'Luna the Moon Owl',
        'short_name': 'Luna',
        'icon': '🦉',
        'description': 'A wise little owl who glows softly in moonlight',
        'tier': 'free',
        'name_i18n': {
            'en': 'Luna the Moon Owl',
            'es': 'Luna, la lechuza de la luna',
            'fr': 'Luna, la chouette de la lune',
            'de': 'Luna, die Mondeule',
            'it': 'Luna, la civetta della luna',
        },
        'description_i18n': {
            'en': 'A wise little owl who glows softly in moonlight',
            'es': 'Una pequeña lechuza sabia que brilla suavemente bajo la luz de la luna',
            'fr': 'Une petite chouette sage qui brille doucement au clair de lune',
            'de': 'Eine kleine weise Eule, die im Mondlicht sanft leuchtet',
            'it': 'Una piccola civetta saggia che brilla dolcemente alla luce della luna',
        },
    },
    'milo_fox': {
        'name': 'Milo the Sleepy Fox',
        'short_name': 'Milo',
        'icon': '🦊',
        'description': 'A cozy fox who knows all the best sleeping spots',
        'tier': 'free',
        'name_i18n': {
            'en': 'Milo the Sleepy Fox',
            'es': 'Milo, el zorro dormilón',
            'fr': 'Milo, le renard somnolent',
            'de': 'Milo, der schläfrige Fuchs',
            'it': 'Milo, la volpe assonnata',
        },
        'description_i18n': {
            'en': 'A cozy fox who knows all the best sleeping spots',
            'es': 'Un zorro acogedor que conoce los rincones más tranquilos para dormir',
            'fr': 'Un renard tout doux qui connaît les meilleurs endroits pour s’endormir',
            'de': 'Ein gemütlicher Fuchs, der die besten Schlafplätze kennt',
            'it': 'Una volpe tenera che conosce tutti i posti migliori per addormentarsi',
        },
    },
    'spark_dragon': {
        'name': 'Spark the Tiny Dragon',
        'short_name': 'Spark',
        'icon': '🐉',
        'description': 'A palm-sized dragon who breathes warm, sparkly light',
        'tier': 'premium',
        'name_i18n': {
            'en': 'Spark the Tiny Dragon',
            'es': 'Spark, el pequeño dragón',
            'fr': 'Spark, le tout petit dragon',
            'de': 'Spark, der winzige Drache',
            'it': 'Spark, il piccolo drago',
        },
        'description_i18n': {
            'en': 'A palm-sized dragon who breathes warm, sparkly light',
            'es': 'Un pequeño dragón del tamaño de la palma de la mano que respira una luz cálida y brillante',
            'fr': 'Un dragon minuscule qui tient dans la main et souffle une lumière chaude et scintillante',
            'de': 'Ein handflächengroßer Drache, der warmes, funkelndes Licht atmet',
            'it': 'Un draghetto grande come una mano che soffia una luce calda e scintillante',
        },
    },
    'stella_fairy': {
        'name': 'Stella the Star Fairy',
        'short_name': 'Stella',
        'icon': '✨',
        'description': 'A tiny fairy who sprinkles sleepy stardust',
        'tier': 'premium',
        'name_i18n': {
            'en': 'Stella the Star Fairy',
            'es': 'Stella, el hada de las estrellas',
            'fr': 'Stella, la fée des étoiles',
            'de': 'Stella, die Sternenfee',
            'it': 'Stella, la fata delle stelle',
        },
        'description_i18n': {
            'en': 'A tiny fairy who sprinkles sleepy stardust',
            'es': 'Una pequeña hada que esparce polvo de estrellas para ayudar a dormir',
            'fr': 'Une minuscule fée qui répand une poussière d’étoiles pour s’endormir',
            'de': 'Eine winzige Fee, die schläfrigen Sternenstaub verstreut',
            'it': 'Una fatina minuscola che sparge polvere di stelle per la nanna',
        },
    },
    'bramble_bear': {
        'name': 'Bramble the Gentle Bear',
        'short_name': 'Bramble',
        'icon': '🐻',
        'description': 'A soft, cuddly bear who gives the best hugs',
        'tier': 'premium',
        'name_i18n': {
            'en': 'Bramble the Gentle Bear',
            'es': 'Bramble, el oso tierno',
            'fr': 'Bramble, l’ours tout doux',
            'de': 'Bramble, der sanfte Bär',
            'it': 'Bramble, l’orso gentile',
        },
        'description_i18n': {
            'en': 'A soft, cuddly bear who gives the best hugs',
            'es': 'Un oso suave y cariñoso que da los mejores abrazos',
            'fr': 'Un ours doux et câlin qui donne les meilleurs câlins',
            'de': 'Ein weicher, kuscheliger Bär, der die besten Umarmungen gibt',
            'it': 'Un orso morbido e affettuoso che dà gli abbracci più belli',
        },
    },
}

SUBSCRIPTION_TIERS: Dict[str, Dict[str, Any]] = {
    'free': {
        'weekly_story_limit': 2,
        'weekly_narration_limit': 2,
        'max_saved_stories': 10,
        'narrators': ['wise_owl', 'night_owl_english', 'night_owl_spanish', 'night_owl_german', 'night_owl_french', 'night_owl_italian'],
        'companions': ['luna_owl', 'milo_fox'],
        'parent_voice': True,
    },
    'premium': {
        'weekly_story_limit': None,
        'weekly_narration_limit': None,
        'max_saved_stories': 30,
        'narrators': list(VOICE_PRESETS.keys()),
        'companions': list(STORY_COMPANIONS.keys()),
        'parent_voice': True,
    },
}

PREMIUM_TESTER_EMAILS = {
    'qa@pillowtales.co',
    'pttest@pillowtales.co',
}

QA_PARENT_VOICE_BYPASS_EMAILS = {
    'qa@pillowtales.co',
    'pttest@pillowtales.co',
}
