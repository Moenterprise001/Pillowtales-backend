from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
from supabase import create_client, Client
from passlib.context import CryptContext
from jose import JWTError, jwt
import json
import google.generativeai as genai
GEMINI_MODEL = "gemini-3-flash-preview"

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# --- AUDIOOP SAFE IMPORT (Python 3.13 fix) ---
AUDIOOP_AVAILABLE = True

try:
    import audioop
except ImportError:
    try:
        import pyaudioop as audioop  # fallback name
    except ImportError:
        AUDIOOP_AVAILABLE = False
        audioop = None

if not AUDIOOP_AVAILABLE:
    print("[WARNING] audioop not available - audio compression disabled, will fallback to OpenAI TTS if needed")
print(f"[AUDIO] audioop available: {AUDIOOP_AVAILABLE}")

# ================== TTS Provider Configuration ==================
# OpenAI TTS (default - cheaper): ~$0.015/1K chars
# ElevenLabs (premium): ~$0.30/1K chars

TTS_PROVIDER_DEFAULT = "openai"  # Default to cheaper OpenAI
TTS_PROVIDER_PREMIUM = "elevenlabs"  # Premium option

USE_ELEVENLABS = os.getenv("USE_ELEVENLABS", "false").lower() == "true"


def elevenlabs_available() -> bool:
    key = os.getenv("ELEVENLABS_API_KEY")
    return bool(key and key != "placeholder-elevenlabs-key")


def should_use_elevenlabs(voice_preference: Optional[str] = None, voice_preset: Optional[str] = None) -> bool:
    """
    Decide whether ElevenLabs should be used.
    ElevenLabs is only used when:
    - USE_ELEVENLABS=true
    - ELEVENLABS_API_KEY is present
    - the selected narrator/preset is an ElevenLabs one
    """
    if not USE_ELEVENLABS:
        return False

    if not elevenlabs_available():
        return False

    selected = voice_preference or voice_preset or DEFAULT_NARRATOR
    preset = VOICE_PRESETS.get(selected, {})
    preset_provider = preset.get("provider", "openai")
    return preset_provider == "elevenlabs"

# ================== DEV MODE - SKIP TTS FOR DEBUGGING ==================
# ================== TTS GENERATION BLOCK ==================
# CREDIT PROTECTION: Set to True to block ALL new TTS generation
# Only cached audio will be playable - protects ElevenLabs credits during debugging
TTS_GENERATION_BLOCKED = False  # <-- RE-ENABLED FOR TESTING

# ================== NARRATION CACHING CONFIGURATION ==================
# Audio is cached per-user, per-story, per-narrator, per-language
# Storage path: {user_id}/{story_id}/chunked/{narrator}_{lang}/page_{n}.mp3
# 
# CACHE BEHAVIOR:
# 1. ALWAYS check database for existing audio before generating
# 2. NEVER regenerate if cached audio exists and is accessible
# 3. Parent Voice and System Narrators both get cached
# 4. Cache persists across devices (stored in Supabase DB + Storage)
#
# COST OPTIMIZATION:
# - First playback: ~$0.30-0.50 per story (ElevenLabs generation)
# - Subsequent playbacks: $0 (served from cache)
# - Expected savings: 90-99% reduction in TTS costs

DEV_MODE_SKIP_TTS = os.getenv("DEV_MODE_SKIP_TTS", "false").lower() == "true"
DEV_MODE_REQUIRE_CACHE = True  # If True, will error if no cached audio exists instead of generating

# ================== Bedtime-Optimized Voice Presets ==================
# OpenAI TTS voices with bedtime-appropriate settings
# DO NOT USE: alloy (sounds metallic/funnel-like)
# ================== NARRATOR PERSONALITIES ==================
# Named storyteller characters using ElevenLabs for expressive narration
# Each personality has specific voice, style settings for warm bedtime delivery
# Settings tuned for WARM, HUMAN-SOUNDING narration (not robotic)
VOICE_PRESETS = {
    # ================== BEDTIME NARRATOR PERSONALITIES ==================
    # Language-filtered narrator options for parents to choose from
    # Each narrator has a language_code: "en", "es", "de", "fr", "it", "pt", or "all" (parent_voice)
    
    # ========== CALM STORYTELLER 🌙 (DEFAULT - FOUNDER'S VOICE) ==========
    # The default narrator for all users - founder's voice cloned for bedtime
    "calm_storyteller": {
        "provider": "elevenlabs",
        "voice_id": "L9jN5cGcpJEym0VnazMG",  # Calm Storyteller - system narrator voice
        "name": "Calm Storyteller",
        "description": "Soft & soothing bedtime voice",
        "icon": "🌙",
        "personality": "calm",
        "language_code": "en",  # English narrator
        # V1 SETTINGS: Best articulation - let natural voice shine
        "stability": 0.50,
        "similarity_boost": 0.75,
        "style": 0.0,
        "speed": 1.0,
        "tier": "free",
    },
    
    # ========== WISE OWL 🦉 (UK FEMALE - CHARLOTTE) ==========
    # Alternative narrator - soft British female voice
    "wise_owl": {
        "provider": "elevenlabs",
        "voice_id": "XB0fDUnXU5powFXDhCwa",  # Charlotte - UK female
        "name": "Wise Owl",
        "description": "Soft British accent, gentle & wise",
        "icon": "🦉",
        "personality": "calm",
        "language_code": "en",  # English narrator
        # V1 SETTINGS: Best articulation - let natural voice shine
        "stability": 0.50,
        "similarity_boost": 0.75,
        "style": 0.0,
        "speed": 1.0,
        "tier": "free",
    },
    
    # ========== PARENT VOICE ❤️ (USER'S CLONED VOICE) ==========
    # Uses parent's own recorded voice - key differentiator feature
    # Works for ALL languages - the parent records in their chosen language
    "parent_voice": {
        "provider": "elevenlabs",
        "voice_id": None,  # Set dynamically from user profile
        "name": "Parent Voice",
        "description": "Your own voice reads stories",
        "icon": "❤️",
        "personality": "personal",
        "language_code": "all",  # Universal - works for any language
        # V1 SETTINGS: Neutral settings for parent's natural voice
        "stability": 0.75,          # Natural expression
        "similarity_boost": 0.50,   # Clear, close to original voice
        "style": 0.0,
        "speed": 1.0,               # Normal speed - don't slow down!
        "is_premium": True,
        "requires_setup": True,
        "tier": "premium",
    },
    
    # ========== BÚHO SABIO 🦉 - SPANISH (LOIDA - EUROPEAN SPANISH) ==========
    # Native Spanish (Spain) female narrator for Spanish language stories
    # Calm, gentle storytelling voice - perfect for bedtime
    "night_owl_spanish": {
        "provider": "elevenlabs",
        "voice_id": "HYlEvvU9GMan5YdjFYpg",  # Loida Burgos - Spanish (Spain)
        "name": "Búho Sabio",  # "Wise Owl" in Spanish
        "description": "Suave voz española para dormir",  # "Soft Spanish voice for sleep"
        "icon": "🦉",
        "personality": "calm",
        "language_code": "es",  # Spanish narrator
        # V1 SETTINGS: Best articulation - let natural voice shine
        "stability": 0.50,            # Natural expression
        "similarity_boost": 0.75,     # Clear articulation
        "style": 0.0,                 # No style modification
        "speed": 1.0,                 # Normal speed
        "tier": "free",
    },
    
    # ========== WEISE EULE 🦉 - GERMAN (SERAPHINA - GERMAN FEMALE) ==========
    # Native German female narrator - calm, gentle bedtime voice
    "night_owl_german": {
        "provider": "elevenlabs",
        "voice_id": "5Q0t7uMcjvnagumLfvZi",  # Seraphina - soft German female voice
        "name": "Weise Eule",  # "Wise Owl" in German
        "description": "Sanfte deutsche Stimme zum Einschlafen",  # "Soft German voice for sleep"
        "icon": "🦉",
        "personality": "calm",
        "language_code": "de",  # German narrator
        # V1 SETTINGS: Best articulation - let natural voice shine
        "stability": 0.50,
        "similarity_boost": 0.75,
        "style": 0.0,
        "speed": 1.0,
        "tier": "free",
    },
    
    # ========== HIBOU SAGE 🦉 - FRENCH (CHARLOTTE - FRENCH FEMALE) ==========
    # Native French female narrator - calm, gentle bedtime voice
    "night_owl_french": {
        "provider": "elevenlabs",
        "voice_id": "XB0fDUnXU5powFXDhCwa",  # Charlotte - also works well for French
        "name": "Hibou Sage",  # "Wise Owl" in French
        "description": "Douce voix française pour dormir",  # "Soft French voice for sleep"
        "icon": "🦉",
        "personality": "calm",
        "language_code": "fr",  # French narrator
        # V1 SETTINGS: Best articulation - let natural voice shine
        "stability": 0.50,
        "similarity_boost": 0.75,
        "style": 0.0,
        "speed": 1.0,
        "tier": "free",
    },
    
    # ========== GUFO SAGGIO 🦉 - ITALIAN (ARIA - ITALIAN FEMALE) ==========
    # ========== GUFO SAGGIO 🦉 - ITALIAN (MANUELA - NATIVE ITALIAN FEMALE) ==========
    # Native Italian female narrator - warm, clear voice perfect for bedtime storytelling
    "night_owl_italian": {
        "provider": "elevenlabs",
        "voice_id": "oVJbgLwL0s5pk9e2U6QH",  # Manuela - warm, clear Italian professional actress voice
        "name": "Gufo Saggio",  # "Wise Owl" in Italian
        "description": "Dolce voce italiana per dormire",  # "Sweet Italian voice for sleep"
        "icon": "🦉",
        "personality": "calm",
        "language_code": "it",  # Italian narrator
        # V1 SETTINGS: Best articulation - let natural voice shine
        "stability": 0.50,
        "similarity_boost": 0.75,
        "style": 0.0,
        "speed": 1.0,
        "tier": "free",
    },
    
    # ========== PORTUGUESE NARRATOR - DISABLED FOR LAUNCH ==========
    # Portuguese support will be reintroduced in a later update
    # "night_owl_portuguese": {
    #     "provider": "elevenlabs",
    #     "voice_id": "lWq4KDY8znfkV0DrK8Vb",  # Yasmin Alves - gentle, warm Brazilian Portuguese
    #     "name": "Coruja Sábia",  # "Wise Owl" in Portuguese
    #     "description": "Voz portuguesa suave para dormir",
    #     "icon": "🦉",
    #     "personality": "calm",
    #     "language_code": "pt",
    #     "stability": 0.50,
    #     "similarity_boost": 0.75,
    #     "style": 0.0,
    #     "speed": 1.0,
    #     "tier": "free",
    # },
}

# Default narrator - Calm Storyteller (UK female) for all stories
DEFAULT_NARRATOR = "calm_storyteller"

# ElevenLabs bedtime-optimized voice settings by personality
# PAUSE DURATIONS (in milliseconds) - calibrated for BEDTIME narration
# Goal: Help children fall asleep with soft, slow, relaxed pacing
# Think: Parent reading quietly in a dark room beside the child
ELEVENLABS_PAUSE_DURATIONS = {
    # Friendly Lion - gentle bedtime pacing
    "friendly": {
        "sentence": 1100,         # Long pause after each sentence
        "comma": 400,             # Breathing room at commas
        "paragraph": 1800,        # Extended pause between story pages
        "whisper_before": 900,    # Build anticipation before whispers
        "whisper_after": 1200,    # Let whisper moments settle deeply
        "softly_before": 700,     # Gentle pause before soft speech
        "softly_after": 1000,     # Let soft moments breathe
        "chuckle": 800,           # Calm pause after gentle humor
    },
    # Calm Storyteller - ultra-relaxing bedtime pacing
    "calm": {
        "sentence": 1400,         # Very long sentence pauses
        "comma": 500,             # Extended breathing at commas
        "paragraph": 2200,        # Long contemplative page breaks
        "whisper_before": 1100,   # Extended whisper anticipation
        "whisper_after": 1500,    # Deep settling after whispers
        "softly_before": 850,     # Gentle buildup to soft speech
        "softly_after": 1200,     # Let softness fully resonate
        "chuckle": 950,           # Thoughtful pause after humor
    },
    # Wise Owl - slowest, most meditative pacing
    "wise": {
        "sentence": 1600,         # Very deliberate sentence pauses
        "comma": 600,             # Thoughtful comma pauses
        "paragraph": 2500,        # Extended contemplative breaks
        "whisper_before": 1200,   # Long whisper anticipation
        "whisper_after": 1700,    # Full whisper resonance
        "softly_before": 950,     # Gentle wisdom buildup
        "softly_after": 1350,     # Let gentle wisdom settle
        "chuckle": 1050,          # Wise, knowing pause
    },
    # Parent Voice - natural bedtime pacing
    "personal": {
        "sentence": 1200,
        "comma": 420,
        "paragraph": 1900,
        "whisper_before": 950,
        "whisper_after": 1250,
        "softly_before": 750,
        "softly_after": 1050,
        "chuckle": 850,
    },
}

# ================== SUBSCRIPTION TIERS ==================
# Calm monetization model - generous free tier, gentle upgrade prompts
SUBSCRIPTION_TIERS = {
    "free": {
        "name": "Free",
        "daily_narration_limit": 2,  # 2 narrated stories per day
        "unlimited_text_stories": True,
        # All system narrators (language-specific) available to free users
        # Portuguese removed for launch
        "narrators": [
            "calm_storyteller", "wise_owl",  # English
            "night_owl_spanish",              # Spanish
            "night_owl_german",               # German
            "night_owl_french",               # French
            "night_owl_italian",              # Italian (Gufo Saggio - Manuela)
        ],
        "companions": ["luna_owl", "milo_fox"],  # Basic companions
        "parent_voice": False,
        "priority_generation": False,
    },
    "premium": {
        "name": "Premium",
        "daily_narration_limit": None,  # Unlimited
        "unlimited_text_stories": True,
        # All narrators including Parent Voice (works for any language)
        # Portuguese removed for launch
        "narrators": [
            "calm_storyteller", "wise_owl",  # English
            "night_owl_spanish",              # Spanish
            "night_owl_german",               # German
            "night_owl_french",               # French
            "night_owl_italian",              # Italian (Gufo Saggio - Manuela)
            "parent_voice",                   # User's own voice (all languages)
        ],
        "companions": ["luna_owl", "spark_dragon", "milo_fox", "stella_fairy", "bramble_bear"],  # All
        "parent_voice": True,
        "priority_generation": True,
    },
}

# Premium pricing
PREMIUM_PRICING = {
    "monthly": {
        "price": 6.99,
        "currency": "USD",
        "period": "month",
        "trial_days": 7,
    },
    "yearly": {
        "price": 49.00,
        "currency": "USD",
        "period": "year",
        "trial_days": 7,
        "savings_percent": 42,  # vs monthly
    },
}

# Free tier limits
FREE_DAILY_NARRATION_LIMIT = 2

# ================== TESTER/ADMIN ACCOUNTS ==================
# These accounts automatically get PERMANENT PREMIUM access for QA testing
# They bypass all subscription checks - perfect for development & testing
# 
# HOW IT WORKS:
# - Any email in this list gets full premium features
# - Bypasses RevenueCat subscription checks
# - Unlimited narrations, all voices, all features
# - Works immediately after signup with any password
#
TESTER_EMAILS = [
    # Primary QA accounts (recommended for testing)
    "qa@pillowtales.app",
    "test@pillowtales.app",
    "qa@pillowtales.co",
    "test@pillowtales.co",
    
    # Development accounts
    "logintest@pillowtales.app",
    "dev@pillowtales.app",
    "dev@pillowtales.co",
    
    # Support/Admin accounts
    "support@pillowtales.co",
    "hello@pillowtales.co",
    
    # Add more tester emails as needed below:
]

# Language-specific voice mappings for natural pronunciation
LANGUAGE_VOICES = {
    "en": {"openai": "nova", "elevenlabs": "21m00Tcm4TlvDq8ikWAM"},
    "es": {"openai": "nova", "elevenlabs": "EXAVITQu4vr4xnSDxMaL"},  # Spanish narrator
    "fr": {"openai": "nova", "elevenlabs": "ThT5KcBeYPX3keUQqHPh"},  # French narrator
    "de": {"openai": "nova", "elevenlabs": "pNInz6obpgDQGcFmaJgB"},  # German narrator
    "it": {"openai": "nova", "elevenlabs": "VR6AewLTigWG4xSOukaG"},  # Italian narrator
    "pt": {"openai": "nova", "elevenlabs": "ErXwobaYiN019PkySvjV"},  # Portuguese narrator
}

# Legacy OpenAI voice mapping (for backwards compatibility)
OPENAI_VOICES = {
    "default": "nova",
    "calm": "echo",
    "expressive": "fable",
    "gentle": "shimmer",
}

# Cost tracking (per 1K characters)
TTS_COST_PER_1K = {
    "openai": 0.015,       # $0.015 per 1K chars (tts-1 standard)
    "openai-hd": 0.030,    # $0.030 per 1K chars (tts-1-hd high definition)
    "elevenlabs": 0.30,    # $0.30 per 1K chars
}

# ================== BEDTIME RITUAL ENDING ==================
# Personalized goodnight messages in multiple languages
BEDTIME_RITUAL_TEMPLATES = {
    'en': [
        "\n\n[softly] Goodnight, {child_name}.\nSleep well tonight.",
        "\n\n[softly] Sweet dreams, {child_name}.\nSleep peacefully.",
        "\n\n[softly] Goodnight, dear {child_name}.\nMay your dreams be filled with wonder.",
    ],
    'es': [
        "\n\n[softly] Buenas noches, {child_name}.\nQue duermas bien.",
        "\n\n[softly] Dulces sueños, {child_name}.\nDuerme tranquilo.",
        "\n\n[softly] Buenas noches, querido {child_name}.\nQue tus sueños sean maravillosos.",
    ],
    'fr': [
        "\n\n[softly] Bonne nuit, {child_name}.\nDors bien ce soir.",
        "\n\n[softly] Fais de beaux rêves, {child_name}.\nDors paisiblement.",
        "\n\n[softly] Bonne nuit, cher {child_name}.\nQue tes rêves soient merveilleux.",
    ],
    'de': [
        "\n\n[softly] Gute Nacht, {child_name}.\nSchlaf gut heute Nacht.",
        "\n\n[softly] Süße Träume, {child_name}.\nSchlaf friedlich.",
        "\n\n[softly] Gute Nacht, lieber {child_name}.\nMögen deine Träume wundervoll sein.",
    ],
    'it': [
        "\n\n[softly] Buonanotte, {child_name}.\nDormi bene stanotte.",
        "\n\n[softly] Sogni d'oro, {child_name}.\nDormi serenamente.",
        "\n\n[softly] Buonanotte, caro {child_name}.\nChe i tuoi sogni siano meravigliosi.",
    ],
    'pt': [
        "\n\n[softly] Boa noite, {child_name}.\nDurma bem esta noite.",
        "\n\n[softly] Bons sonhos, {child_name}.\nDurma em paz.",
        "\n\n[softly] Boa noite, querido {child_name}.\nQue seus sonhos sejam maravilhosos.",
    ],
    'nl': [
        "\n\n[softly] Welterusten, {child_name}.\nSlaap lekker vanavond.",
        "\n\n[softly] Zoete dromen, {child_name}.\nSlaap zacht.",
    ],
    'pl': [
        "\n\n[softly] Dobranoc, {child_name}.\nŚpij dobrze tej nocy.",
        "\n\n[softly] Słodkich snów, {child_name}.\nŚpij spokojnie.",
    ],
    'ja': [
        "\n\n[softly] おやすみなさい、{child_name}。\n今夜はぐっすり眠ってね。",
        "\n\n[softly] 甘い夢を、{child_name}。\n静かにお休みなさい。",
    ],
    'zh': [
        "\n\n[softly] 晚安，{child_name}。\n今晚好好睡觉。",
        "\n\n[softly] 做个好梦，{child_name}。\n安静地睡吧。",
    ],
    'ko': [
        "\n\n[softly] 잘 자, {child_name}.\n오늘 밤 푹 자렴.",
        "\n\n[softly] 좋은 꿈 꿔, {child_name}.\n평화롭게 자렴.",
    ],
}

def get_bedtime_ritual_ending(child_name: str, language: str = 'en') -> str:
    """
    Generate a personalized bedtime ritual ending for the story.
    
    This creates a consistent, comforting ending that children come to expect.
    The ritual includes:
    - A pause (double newline)
    - A soft delivery marker [softly]
    - A personalized goodnight message with the child's name
    - A closing blessing for sleep
    """
    import random
    
    # Get templates for the language, default to English
    lang_code = language.lower()[:2] if language else 'en'
    templates = BEDTIME_RITUAL_TEMPLATES.get(lang_code, BEDTIME_RITUAL_TEMPLATES['en'])
    
    # Select a random template for variety
    template = random.choice(templates)
    
    # Personalize with child's name
    ritual_ending = template.format(child_name=child_name)
    
    logger.info(f"[BEDTIME RITUAL] Generated ritual ending for {child_name} in {lang_code}")
    
    return ritual_ending

def append_bedtime_ritual_to_story(pages: list, child_name: str, language: str = 'en') -> list:
    """
    Append the bedtime ritual ending to the last page of the story.
    
    This ensures every story ends with a consistent, comforting goodnight message.
    """
    if not pages:
        return pages
    
    # Get the ritual ending
    ritual_ending = get_bedtime_ritual_ending(child_name, language)
    
    # Append to the last page
    pages_with_ritual = pages.copy()
    pages_with_ritual[-1] = pages_with_ritual[-1] + ritual_ending
    
    logger.info(f"[BEDTIME RITUAL] Appended ritual to story for {child_name}")
    
    return pages_with_ritual


# ================== STORY COMPANIONS ==================
# Magical recurring characters that build a living story world
# Each child's story world remembers which companions they've met
STORY_COMPANIONS = {
    "luna_owl": {
        "name": "Luna the Moon Owl",
        "short_name": "Luna",
        "icon": "🦉",
        "description": "A wise little owl who glows softly in moonlight",
        "personality": "gentle, wise, speaks in soft hoots",
        "appearance": "small silver owl with glowing golden eyes",
        "catchphrase": "Hoo-hoo, little dreamer",
        "themes": ["nighttime", "wisdom", "dreams", "stars"],
        "intro_story": "Luna lives in the tallest tree, watching over sleepy children.",
        "tier": "free",  # Available to free users
    },
    "spark_dragon": {
        "name": "Spark the Tiny Dragon",
        "short_name": "Spark",
        "icon": "🐉",
        "description": "A palm-sized dragon who breathes warm, sparkly light",
        "personality": "curious, playful, a little clumsy",
        "appearance": "tiny purple dragon with golden wings",
        "catchphrase": "Let's go on an adventure!",
        "themes": ["adventure", "courage", "magic", "friendship"],
        "intro_story": "Spark hatched from a starlight egg and loves making new friends.",
        "tier": "premium",  # Premium only
    },
    "milo_fox": {
        "name": "Milo the Sleepy Fox",
        "short_name": "Milo",
        "icon": "🦊",
        "description": "A cozy fox who knows all the best sleeping spots",
        "personality": "drowsy, warm, loves cozy things",
        "appearance": "fluffy orange fox with a soft, bushy tail",
        "catchphrase": "*yawns* Time for cozy dreams",
        "themes": ["sleep", "comfort", "warmth", "kindness"],
        "intro_story": "Milo curls up in the softest places and helps everyone feel safe.",
        "tier": "free",  # Available to free users
    },
    "stella_fairy": {
        "name": "Stella the Star Fairy",
        "short_name": "Stella",
        "icon": "✨",
        "description": "A tiny fairy who sprinkles sleepy stardust",
        "personality": "gentle, magical, speaks in whispers",
        "appearance": "tiny fairy with shimmering wings made of starlight",
        "catchphrase": "Sweet dreams are made of stars",
        "themes": ["magic", "stars", "dreams", "wishes"],
        "intro_story": "Stella flies from star to star, collecting dream dust for children.",
        "tier": "premium",  # Premium only
    },
    "bramble_bear": {
        "name": "Bramble the Gentle Bear",
        "short_name": "Bramble",
        "icon": "🐻",
        "description": "A soft, cuddly bear who gives the best hugs",
        "personality": "protective, calm, loves honey and stories",
        "appearance": "fluffy brown bear with kind, sleepy eyes",
        "catchphrase": "Everything is okay now",
        "themes": ["comfort", "protection", "kindness", "nature"],
        "intro_story": "Bramble lives in a cozy cave and always has warm hugs ready.",
        "tier": "premium",  # Premium only
    },
}

# Companion appearance probability (30% chance a companion appears)
COMPANION_APPEARANCE_CHANCE = 0.30
# Chance a known companion returns vs meeting a new one (70% return)
KNOWN_COMPANION_RETURN_CHANCE = 0.70

# Log ElevenLabs API key on startup
eleven_key = os.environ.get('ELEVENLABS_API_KEY')
if eleven_key:
    print(f"ELEVEN key prefix: {eleven_key[:6]}")
else:
    print("ELEVEN key prefix: None (not set)")

# Emergent LLM Key for OpenAI TTS
emergent_key = os.environ.get('GEMINI_API_KEY')
if emergent_key:
    print(f"GEMINI_API_KEY prefix: {emergent_key[:12]}...")
else:
    print("GEMINI_API_KEY: Not set")

# Supabase configuration
supabase_url = os.environ.get('SUPABASE_URL', 'https://your-project.supabase.co')
supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', 'placeholder-key')

# Log Supabase configuration on startup
print(f"SUPABASE_URL: {supabase_url[:50]}...")
print(f"SUPABASE_SERVICE_ROLE_KEY prefix: {supabase_key[:20]}..." if supabase_key != 'placeholder-key' else "SUPABASE_SERVICE_ROLE_KEY: NOT SET")

supabase: Client = create_client(supabase_url, supabase_key)

# JWT configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'pillowtales-secret-key-change-in-production')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 30  # 30 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Create the main app without a prefix
app = FastAPI(title="PillowTales API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== Models ==================

class SignupRequest(BaseModel):
    email: str
    password: str
    preferredLanguage: Optional[str] = "en"

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    userId: str
    email: str
    preferredLanguage: str

class StoryCharacter(BaseModel):
    name: str
    relationship: str

class GenerateStoryRequest(BaseModel):
    userId: str
    childName: str
    age: int
    theme: str  # dragons, space, animals, princess, adventure, or custom theme
    moral: str  # kindness, bravery, sharing, patience
    calmLevel: str  # very_calm, calm, adventure
    durationMin: int  # 8 (Short ~8min), 11 (Bedtime ~11min), or 15 (Long Adventure ~15min)
    storyLanguageCode: str = "en"  # Language for story text
    narrationLanguageCode: Optional[str] = None  # Language for TTS, None means same as story
    # Story continuation fields
    continueFromStoryId: Optional[str] = None  # ID of story to continue from
    # Personalization fields
    characters: Optional[List[StoryCharacter]] = None  # Family, friends, pets to include
    customTheme: Optional[str] = None  # Custom theme when "other" is selected
    companionId: Optional[str] = None  # Story companion (character appearing IN the story)
    gender: Optional[str] = "neutral"  # Pronouns: "girl" (she/her), "boy" (he/him), "neutral" (they/them)
    
# Maximum parts in a story arc
MAX_STORY_ARC_PARTS = 5

# Supported languages for launch (EN, ES, FR, DE, IT)
# Portuguese will be reintroduced in a later update
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
}

# ElevenLabs voice mapping by language
ELEVENLABS_VOICES = {
    "en": {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    "es": {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
    "fr": {"voice_id": "XB0fDUnXU5powFXDhCwa", "name": "Charlotte"},
    "de": {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
    "it": {"voice_id": "oVJbgLwL0s5pk9e2U6QH", "name": "Manuela"},  # Updated to Manuela for Italian
}

# Multilingual fallback voice
MULTILINGUAL_VOICE = {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"}  # Rachel supports multiple languages

class StoryResponse(BaseModel):
    storyId: str
    title: str
    pages: List[str]

class TTSRequest(BaseModel):
    storyId: str
    text: str
    language: str

class TTSResponse(BaseModel):
    audioUrl: str

# Narration request model (optimized on-demand TTS)
class NarrationRequest(BaseModel):
    storyId: str
    narrationLanguageCode: Optional[str] = None  # If not provided, use story's language
    voicePreference: Optional[str] = None  # Optional voice_id override

class NarrationResponse(BaseModel):
    status: str  # 'ready', 'generating', 'failed'
    audioUrl: Optional[str] = None
    message: Optional[str] = None

# ==================== PAGE-1-FIRST CHUNKED NARRATION ====================
# Models for progressive page-by-page narration generation
class ChunkedNarrationResponse(BaseModel):
    status: str  # 'page_ready', 'generating', 'all_ready', 'failed'
    currentPage: int  # Which page is ready (1-indexed)
    totalPages: int
    pageAudioUrl: Optional[str] = None  # Signed URL for current page audio
    pagesReady: List[int] = []  # List of pages that have audio ready
    message: Optional[str] = None

class PageStatusResponse(BaseModel):
    storyId: str
    totalPages: int
    pagesReady: List[int]  # Pages with audio ready (1-indexed)
    pagesGenerating: List[int]  # Pages currently being generated
    pagesFailed: List[int]  # Pages that failed
    allReady: bool
    
class UpdateStoryRequest(BaseModel):
    isFavorite: Optional[bool] = None

class VoiceConsentRequest(BaseModel):
    consent: bool
    consentTextVersion: str = "v1.0"

class VoiceRecordingUpload(BaseModel):
    recordingData: str  # base64 encoded audio
    sampleText: str
    durationSeconds: float

class CreateVoiceModelRequest(BaseModel):
    voiceName: str = "Parent Voice"

class Story(BaseModel):
    id: str
    user_id: str
    title: str
    child_name: str
    age: int
    theme: str
    moral: str
    calm_level: str
    duration_min: int
    language: str  # Legacy field, kept for backwards compatibility
    story_language_code: str = "en"  # Language of story text
    narration_language_code: Optional[str] = None  # Language for TTS narration
    pages: List[str]
    full_text: str
    audio_url: Optional[str] = None
    audio_storage_path: Optional[str] = None
    audio_status: str = "none"  # none, generating, ready, failed
    audio_voice_id: Optional[str] = None
    audio_language_code: str = "en"
    audio_chars: int = 0
    audio_created_at: Optional[str] = None
    is_favorite: bool = False
    created_at: str

# User profile response with streak info
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

# Plan limits
PLAN_LIMITS = {
    "free": {
        "stories_per_week": 3,
        "max_saved_stories": 5,
        "narration_enabled": True,  # Free users CAN use narration
        "narrations_per_day": 2,    # Limited to 2 narrated stories per day
        "narrations_per_month": 60, # ~2/day for a month
        "parent_voice_enabled": False,
        "sleep_mode_enabled": False,
        "tts_provider": "elevenlabs",  # Same quality for all users
    },
    "trial": {
        "stories_per_week": 10,
        "max_saved_stories": 20,
        "narration_enabled": True,
        "narrations_per_day": 5,    # More generous for trial
        "narrations_per_month": 30,
        "parent_voice_enabled": False,
        "sleep_mode_enabled": False,
        "tts_provider": "elevenlabs",
    },
    "premium": {
        "stories_per_week": 999,  # Unlimited
        "max_saved_stories": 999,  # Unlimited
        "narration_enabled": True,
        "narrations_per_month": 50,  # 50 narrations/month
        "parent_voice_enabled": True,
        "sleep_mode_enabled": True,
        "tts_provider": "openai",  # Default to OpenAI, can choose ElevenLabs
        "monthly_soft_limit": 200,  # Backend soft limit
    }
}

# Monthly narration tracking
TRIAL_NARRATION_LIMIT = 10  # 10 narrations during trial
PREMIUM_NARRATION_LIMIT = 50  # 50 narrations per month

# ================== Auth Helpers ==================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify a JWT token - supports both backend tokens and Supabase tokens"""
    # First try to decode with our backend JWT secret
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        pass
    
    # If that fails, try to verify as a Supabase token
    try:
        # Use Supabase to verify the token
        user_response = supabase.auth.get_user(token)
        if user_response and user_response.user:
            return {
                "user_id": user_response.user.id,
                "email": user_response.user.email,
                "sub": user_response.user.id
            }
    except Exception as e:
        logger.error(f"Supabase token verification failed: {str(e)}")
    
    raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    # Support both user_id (backend JWT) and sub (Supabase JWT)
    user_id = payload.get("user_id") or payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id

# ================== Helper Functions ==================

async def get_user_profile_with_stats(user_id: str) -> dict:
    """Get user profile with streak and usage statistics"""
    try:
        # Get user profile
        profile_result = supabase.table('users_profile').select('*').eq('id', user_id).execute()
        
        if not profile_result.data or len(profile_result.data) == 0:
            return None
        
        profile = profile_result.data[0]
        plan = profile.get('plan', 'free')
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
        
        # Get streak info (with fallbacks for missing columns)
        streak_count = profile.get('streak_count', 0) or 0
        last_story_date = profile.get('last_story_date')
        
        # Count stories this week
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        stories_week_result = supabase.table('stories').select('id', count='exact').eq('user_id', user_id).gte('created_at', week_ago).execute()
        stories_this_week = stories_week_result.count if hasattr(stories_week_result, 'count') else len(stories_week_result.data)
        
        # Count total saved stories
        stories_saved_result = supabase.table('stories').select('id', count='exact').eq('user_id', user_id).execute()
        stories_saved = stories_saved_result.count if hasattr(stories_saved_result, 'count') else len(stories_saved_result.data)
        
        # Determine if user can generate more stories
        can_generate = stories_this_week < limits['stories_per_week']
        
        # Determine if user can save more stories
        can_save_more = stories_saved < limits['max_saved_stories']
        
        return {
            "id": user_id,
            "email": profile.get('email', ''),
            "plan": plan,
            "preferred_language": profile.get('preferred_language', 'en'),
            "streak_count": streak_count,
            "last_story_date": str(last_story_date) if last_story_date else None,
            "stories_this_week": stories_this_week,
            "stories_saved": stories_saved,
            "can_generate": can_generate,
            "can_save_more": can_save_more,
            "limits": limits
        }
    except Exception as e:
        logger.error(f"Error getting user profile stats: {str(e)}")
        return None

async def update_streak(user_id: str) -> int:
    """Update user's sleep streak after generating a story. Returns new streak count."""
    try:
        today = datetime.utcnow().date()
        
        # Get current streak info
        profile_result = supabase.table('users_profile').select('streak_count, last_story_date').eq('id', user_id).execute()
        
        if not profile_result.data or len(profile_result.data) == 0:
            return 0
        
        profile = profile_result.data[0]
        current_streak = profile.get('streak_count', 0) or 0
        last_date_str = profile.get('last_story_date')
        
        # Parse last story date
        last_date = None
        if last_date_str:
            try:
                if isinstance(last_date_str, str):
                    last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
                else:
                    last_date = last_date_str
            except:
                last_date = None
        
        # Calculate new streak
        if last_date is None:
            # First story ever
            new_streak = 1
        elif last_date == today:
            # Already generated today, no change
            new_streak = current_streak
        elif last_date == today - timedelta(days=1):
            # Consecutive day, increment streak
            new_streak = current_streak + 1
        else:
            # Streak broken, start fresh
            new_streak = 1
        
        # Update profile with new streak
        supabase.table('users_profile').update({
            'streak_count': new_streak,
            'last_story_date': today.isoformat()
        }).eq('id', user_id).execute()
        
        logger.info(f"Updated streak for user {user_id}: {current_streak} -> {new_streak}")
        return new_streak
        
    except Exception as e:
        logger.error(f"Error updating streak: {str(e)}")
        return 0

async def check_story_limits(user_id: str, plan: str, user_email: str = None) -> dict:
    """Check if user can generate and save stories based on their plan.
    Testers get unlimited access regardless of plan.
    """
    # Check if this is a tester account - they get unlimited access
    if user_email and user_email.lower() in [e.lower() for e in TESTER_EMAILS]:
        logger.info(f"[LIMITS] Tester account detected: {user_email} - granting unlimited access")
        return {
            "can_generate": True,
            "can_save": True,
            "stories_this_week": 0,
            "stories_saved": 0,
            "is_tester": True,
            "reason": None,
            "message": None
        }
    
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
    
    # Count stories this week
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    stories_week_result = supabase.table('stories').select('id', count='exact').eq('user_id', user_id).gte('created_at', week_ago).execute()
    stories_this_week = stories_week_result.count if hasattr(stories_week_result, 'count') else len(stories_week_result.data)
    
    # Count total saved stories
    stories_saved_result = supabase.table('stories').select('id', count='exact').eq('user_id', user_id).execute()
    stories_saved = stories_saved_result.count if hasattr(stories_saved_result, 'count') else len(stories_saved_result.data)
    
    # Check premium monthly soft limit
    if plan == 'premium':
        month_start = datetime.utcnow().replace(day=1).date()
        stories_month_result = supabase.table('stories').select('id', count='exact').eq('user_id', user_id).gte('created_at', month_start.isoformat()).execute()
        stories_this_month = stories_month_result.count if hasattr(stories_month_result, 'count') else len(stories_month_result.data)
        
        if stories_this_month >= limits.get('monthly_soft_limit', 200):
            return {
                "can_generate": False,
                "can_save": True,
                "reason": "monthly_limit",
                "message": "You've reached the monthly story limit. Please try again next month."
            }
    
    can_generate = stories_this_week < limits['stories_per_week']
    can_save = stories_saved < limits['max_saved_stories']
    
    return {
        "can_generate": can_generate,
        "can_save": can_save,
        "stories_this_week": stories_this_week,
        "stories_saved": stories_saved,
        "reason": "weekly_limit" if not can_generate else ("storage_limit" if not can_save else None),
        "message": None
    }

# ================== SUBSCRIPTION HELPERS ==================

async def get_user_subscription(user_id: str, user_email: str = None) -> dict:
    """
    Get user's subscription status and narration usage.
    Handles daily counter reset automatically.
    Also handles tester/admin bypass for testing.
    """
    try:
        # First, check if this is a tester account (bypass all restrictions)
        if user_email and user_email.lower() in [e.lower() for e in TESTER_EMAILS]:
            logger.info(f"[SUB] Tester account detected: {user_email} - granting premium access")
            return {
                "status": "premium",
                "is_premium": True,
                "is_tester": True,
                "daily_narrations_used": 0,
                "daily_limit": None,
                "can_narrate": True,
                "narrations_remaining": None,
                "trial_used": False,
            }
        
        # Try to get email from user profile if not provided
        result = supabase.table('users_profile').select(
            'email, subscription_status, daily_narrations_used, last_narration_reset'
        ).eq('id', user_id).execute()
        
        if not result.data or len(result.data) == 0:
            return {
                "status": "free",
                "is_premium": False,
                "daily_narrations_used": 0,
                "daily_limit": FREE_DAILY_NARRATION_LIMIT,
                "can_narrate": True,
            }
        
        profile = result.data[0]
        
        # Check if email is in tester list (case-insensitive)
        profile_email = profile.get('email', '')
        if profile_email and profile_email.lower() in [e.lower() for e in TESTER_EMAILS]:
            logger.info(f"[SUB] Tester account detected: {profile_email} - granting premium access")
            return {
                "status": "premium",
                "is_premium": True,
                "is_tester": True,
                "daily_narrations_used": 0,
                "daily_limit": None,
                "can_narrate": True,
                "narrations_remaining": None,
                "trial_used": False,
            }
        
        status = profile.get('subscription_status', 'free') or 'free'
        is_premium = status == 'premium'
        
        daily_used = profile.get('daily_narrations_used', 0) or 0
        last_reset = profile.get('last_narration_reset')
        
        # Check if we need to reset daily counter
        should_reset = False
        if last_reset:
            try:
                if isinstance(last_reset, str):
                    last_reset_dt = datetime.fromisoformat(last_reset.replace('Z', '+00:00'))
                else:
                    last_reset_dt = last_reset
                
                # Reset if more than 24 hours since last reset
                now = datetime.utcnow()
                if last_reset_dt.tzinfo:
                    now = now.replace(tzinfo=last_reset_dt.tzinfo)
                
                if (now - last_reset_dt).total_seconds() > 86400:  # 24 hours
                    should_reset = True
            except:
                should_reset = True
        else:
            should_reset = True
        
        if should_reset:
            daily_used = 0
            # Reset the counter in DB
            try:
                supabase.table('users_profile').update({
                    'daily_narrations_used': 0,
                    'last_narration_reset': datetime.utcnow().isoformat()
                }).eq('id', user_id).execute()
            except Exception as e:
                logger.warning(f"[SUB] Could not reset daily counter: {e}")
        
        # Determine if user can narrate
        daily_limit = None if is_premium else FREE_DAILY_NARRATION_LIMIT
        can_narrate = is_premium or daily_used < FREE_DAILY_NARRATION_LIMIT
        
        return {
            "status": status,
            "is_premium": is_premium,
            "daily_narrations_used": daily_used,
            "daily_limit": daily_limit,
            "can_narrate": can_narrate,
            "narrations_remaining": None if is_premium else max(0, FREE_DAILY_NARRATION_LIMIT - daily_used),
            "trial_used": profile.get('trial_used', False),
        }
        
    except Exception as e:
        logger.error(f"[SUB] Error getting subscription: {e}")
        return {
            "status": "free",
            "is_premium": False,
            "daily_narrations_used": 0,
            "daily_limit": FREE_DAILY_NARRATION_LIMIT,
            "can_narrate": True,
        }

async def increment_narration_usage(user_id: str) -> dict:
    """Increment daily narration counter after successful narration generation."""
    try:
        # Get current count
        result = supabase.table('users_profile').select('daily_narrations_used').eq('id', user_id).execute()
        current = 0
        if result.data and len(result.data) > 0:
            current = result.data[0].get('daily_narrations_used', 0) or 0
        
        new_count = current + 1
        
        # Update counter
        supabase.table('users_profile').update({
            'daily_narrations_used': new_count
        }).eq('id', user_id).execute()
        
        logger.info(f"[SUB] User {user_id} narration usage: {new_count}")
        
        return {"daily_narrations_used": new_count}
        
    except Exception as e:
        logger.error(f"[SUB] Error incrementing usage: {e}")
        return {"daily_narrations_used": 0}

def check_feature_access(user_subscription: dict, feature: str, item_id: str = None) -> dict:
    """
    Check if user can access a specific feature/item.
    Returns: {allowed: bool, reason: str, upgrade_required: bool}
    """
    is_premium = user_subscription.get("is_premium", False)
    tier = "premium" if is_premium else "free"
    tier_config = SUBSCRIPTION_TIERS.get(tier, SUBSCRIPTION_TIERS["free"])
    
    if feature == "narrator":
        allowed_narrators = tier_config.get("narrators", ["calm_storyteller"])
        if item_id and item_id not in allowed_narrators:
            narrator = VOICE_PRESETS.get(item_id, {})
            return {
                "allowed": False,
                "reason": f"{narrator.get('name', 'This narrator')} is a premium storyteller",
                "upgrade_required": True,
                "upgrade_message": "Unlock all storytellers with Premium",
            }
    
    elif feature == "companion":
        allowed_companions = tier_config.get("companions", ["luna_owl", "milo_fox"])
        if item_id and item_id not in allowed_companions:
            companion = STORY_COMPANIONS.get(item_id, {})
            return {
                "allowed": False,
                "reason": f"{companion.get('name', 'This friend')} is part of the Premium Story World",
                "upgrade_required": True,
                "upgrade_message": "Meet all magical companions with Premium",
            }
    
    elif feature == "parent_voice":
        if not tier_config.get("parent_voice", False):
            return {
                "allowed": False,
                "reason": "Parent Voice is a premium feature",
                "upgrade_required": True,
                "upgrade_message": "Record your voice to tell bedtime stories",
            }
    
    elif feature == "narration":
        can_narrate = user_subscription.get("can_narrate", True)
        if not can_narrate:
            return {
                "allowed": False,
                "reason": "narration_limit",
                "upgrade_required": True,
                "upgrade_message": "You've listened to tonight's stories.\n\nUpgrade to unlock unlimited bedtime narrations.",
                "narrations_used": user_subscription.get("daily_narrations_used", 0),
                "narrations_limit": FREE_DAILY_NARRATION_LIMIT,
            }
    
    return {"allowed": True, "reason": None, "upgrade_required": False}

async def check_weekly_story_limit(user_id: str, is_premium: bool) -> bool:
    """Check if user has exceeded their weekly story limit (free tier only)"""
    if is_premium:
        return True
    
    try:
        # Calculate date 7 days ago
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        
        # Count stories created in the last 7 days
        response = supabase.table('stories').select('id', count='exact').eq('user_id', user_id).gte('created_at', week_ago).execute()
        
        story_count = response.count if hasattr(response, 'count') else len(response.data)
        
        logger.info(f"User {user_id} has created {story_count} stories in the last 7 days")
        
        return story_count < 3
    except Exception as e:
        logger.error(f"Error checking story limit: {str(e)}")
        # On error, allow the request (fail open)
        return True

def select_story_companion(user_id: str, theme: str, is_premium: bool = False) -> dict | None:
    """
    Select a companion character for a story based on:
    - Random chance (30% of stories get a companion)
    - User's subscription tier (free users get basic companions only)
    - User's history (prefer returning companions)
    - Theme matching
    
    Returns companion dict or None
    """
    import random
    
    # 30% chance a companion appears
    if random.random() > COMPANION_APPEARANCE_CHANCE:
        logger.info("[COMPANION] No companion this time (random chance)")
        return None
    
    # Get user's met companions
    met_companions = []
    try:
        result = supabase.table('users_profile').select('met_companions, subscription_status').eq('id', user_id).execute()
        if result.data and len(result.data) > 0:
            met_companions = result.data[0].get('met_companions', []) or []
            # Also check subscription from DB
            sub_status = result.data[0].get('subscription_status', 'free')
            is_premium = is_premium or sub_status == 'premium'
    except Exception as e:
        logger.warning(f"[COMPANION] Could not fetch met companions: {e}")
    
    logger.info(f"[COMPANION] User has met: {met_companions}, is_premium: {is_premium}")
    
    # Get tier config for available companions
    tier = "premium" if is_premium else "free"
    tier_config = SUBSCRIPTION_TIERS.get(tier, SUBSCRIPTION_TIERS["free"])
    allowed_companions = tier_config.get("companions", ["luna_owl", "milo_fox"])
    
    # Filter companions by tier
    available_companions = {
        comp_id: comp for comp_id, comp in STORY_COMPANIONS.items()
        if comp_id in allowed_companions
    }
    
    if not available_companions:
        logger.warning(f"[COMPANION] No companions available for tier: {tier}")
        return None
    
    # Theme keywords for matching companions
    theme_lower = theme.lower() if theme else ""
    
    # Find companions that match the theme
    themed_companions = []
    for comp_id, comp in available_companions.items():
        for comp_theme in comp.get("themes", []):
            if comp_theme in theme_lower or theme_lower in comp_theme:
                themed_companions.append(comp_id)
                break
    
    # Decide: return a known companion or meet a new one?
    if met_companions and random.random() < KNOWN_COMPANION_RETURN_CHANCE:
        # Return a known companion (preference for theme match)
        themed_known = [c for c in met_companions if c in themed_companions]
        if themed_known:
            selected_id = random.choice(themed_known)
            logger.info(f"[COMPANION] Returning themed known companion: {selected_id}")
        else:
            selected_id = random.choice(met_companions)
            logger.info(f"[COMPANION] Returning known companion: {selected_id}")
        
        return {
            "id": selected_id,
            "is_new": False,
            **STORY_COMPANIONS[selected_id]
        }
    else:
        # Meet a new companion (from allowed list only)
        available_new = [c for c in available_companions.keys() if c not in met_companions]
        
        if not available_new:
            # All allowed companions met, pick any from allowed
            available_new = list(available_companions.keys())
        
        # Prefer theme-matched new companions
        themed_new = [c for c in available_new if c in themed_companions]
        if themed_new:
            selected_id = random.choice(themed_new)
        else:
            selected_id = random.choice(available_new)
        
        is_new = selected_id not in met_companions
        logger.info(f"[COMPANION] {'Meeting new' if is_new else 'Revisiting'} companion: {selected_id}")
        
        return {
            "id": selected_id,
            "is_new": is_new,
            **STORY_COMPANIONS[selected_id]
        }

async def generate_story_with_openai(request: GenerateStoryRequest, continuation_context: dict = None, companion: dict = None) -> dict:
    """
    Generate a story using OpenAI via Emergent integrations.
    Supports story continuation with previous context.
    Supports story companions (recurring magical characters).
    
    Args:
        request: Story generation parameters
        continuation_context: Optional dict with {summary, characters, setting, part_number, title} from previous story
        companion: Optional dict with companion character to include in the story
    """
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        
        # Get language name for prompt
        story_language = SUPPORTED_LANGUAGES.get(request.storyLanguageCode, "English")
        
        # Build continuation context if this is a sequel
        continuation_prompt = ""
        if continuation_context:
            part_number = continuation_context.get('part_number', 1) + 1
            prev_title = continuation_context.get('title', 'the previous story')
            prev_summary = continuation_context.get('summary', '')
            characters = continuation_context.get('characters', [])
            setting = continuation_context.get('setting', '')
            
            char_descriptions = ""
            if characters:
                char_list = [f"- {c.get('name', 'Unknown')}: {c.get('description', '')} ({c.get('role', 'character')})" for c in characters]
                char_descriptions = "\n".join(char_list)
            
            continuation_prompt = f"""
IMPORTANT: This is Part {part_number} of an ongoing story arc!

PREVIOUS STORY CONTEXT:
- Previous title: "{prev_title}"
- Story so far: {prev_summary}
- Setting: {setting}
- Recurring characters:
{char_descriptions}

CONTINUATION RULES:
1. Continue the adventure from where it left off
2. Use the SAME characters from the previous story
3. Keep the same setting/world
4. Reference events from the previous night
5. Create a new mini-adventure that advances the larger story
6. The title should indicate this is Part {part_number} (e.g., "{request.childName}'s Adventure - Part {part_number}")
7. If this is Part 5 (the final part), create a satisfying conclusion to the story arc

"""
        
        # Build companion prompt if a companion is joining this story
        companion_prompt = ""
        if companion:
            is_new = companion.get('is_new', False)
            comp_name = companion.get('name', '')
            short_name = companion.get('short_name', comp_name.split()[0])
            comp_desc = companion.get('description', '')
            comp_personality = companion.get('personality', '')
            comp_appearance = companion.get('appearance', '')
            comp_catchphrase = companion.get('catchphrase', '')
            
            if is_new:
                companion_prompt = f"""
STORY COMPANION (NEW FRIEND):
{request.childName} meets a new magical friend in this story!

- Name: {comp_name}
- Description: {comp_desc}
- Appearance: {comp_appearance}
- Personality: {comp_personality}
- Catchphrase: "{comp_catchphrase}"

INTRODUCTION RULES:
1. {short_name} appears naturally in the story (maybe found sleeping, hiding, or exploring)
2. {short_name} and {request.childName} become friends
3. Include {short_name}'s catchphrase at least once
4. {short_name} helps with the story's adventure or moral lesson
5. End with {short_name} saying goodbye and hinting they might meet again

"""
            else:
                companion_prompt = f"""
STORY COMPANION (OLD FRIEND RETURNS):
{comp_name} appears again in tonight's story!

- Name: {comp_name} (use "{short_name}" in dialogue)
- Description: {comp_desc}
- Personality: {comp_personality}
- Catchphrase: "{comp_catchphrase}"

RETURNING FRIEND RULES:
1. {short_name} appears happily when seeing {request.childName} again
2. They remember each other from before
3. Include {short_name}'s catchphrase
4. {short_name} helps with tonight's adventure
5. Their friendship feels warm and familiar

"""
        
        # Build story characters prompt if any are provided
        characters_prompt = ""
        if request.characters and len(request.characters) > 0:
            char_list = []
            for char in request.characters:
                char_list.append(f"- {char.name} ({char.relationship})")
            characters_prompt = f"""
STORY CHARACTERS (FAMILY & FRIENDS):
Include these real people/pets naturally in the story:
{chr(10).join(char_list)}

CHARACTER RULES:
1. Include these characters as important parts of the story
2. Make interactions feel warm and realistic
3. Use their names naturally in dialogue and narration
4. Show positive relationships with {request.childName}
5. Characters can appear in supportive, helping, or playful roles

"""
        
        # Build gender/pronoun instruction based on selection
        gender = getattr(request, 'gender', 'neutral') or 'neutral'
        if gender == 'girl':
            gender_instruction = f"""
PRONOUN RULES:
When referring to {request.childName}, use she/her pronouns consistently throughout the story.
Examples: "she smiled", "her eyes sparkled", "she felt proud of herself"
"""
        elif gender == 'boy':
            gender_instruction = f"""
PRONOUN RULES:
When referring to {request.childName}, use he/him pronouns consistently throughout the story.
Examples: "he smiled", "his eyes sparkled", "he felt proud of himself"
"""
        else:  # neutral
            gender_instruction = f"""
PRONOUN RULES:
When referring to {request.childName}, use they/them pronouns OR structure sentences to avoid pronouns where possible.
Examples: "they smiled", "their eyes sparkled", "they felt proud of themselves"
OR: "{request.childName} smiled", "{request.childName}'s eyes sparkled"
Keep the narration smooth and natural while using gender-neutral language.
"""
        
        # Determine the effective theme (might be custom)
        effective_theme = request.theme
        if request.customTheme:
            effective_theme = request.customTheme
        
        # ================== STORY INFLUENCER SYSTEM ==================
        # Rotate between 16 different storytelling styles for maximum variety
        # Each influencer guides tone, pacing, imagery, and narrative feel
        import random
        
        STORY_INFLUENCERS = [
            # 1. Woodland Animal Tale
            {
                "name": "Woodland Animal Tale",
                "description": "gentle animal storytelling in the tradition of Beatrix Potter and 'Wind in the Willows'",
                "style_notes": "Anthropomorphic woodland creatures with gentle personalities. Cozy burrows, hollow trees, and forest paths. Animals wear tiny clothes and have warm homes. Focus on small, meaningful moments between animal friends.",
                "pacing": "Pastoral and unhurried, like a walk through autumn woods"
            },
            # 2. Cozy Friendship Story
            {
                "name": "Cozy Friendship Story",
                "description": "heartwarming tales about the comfort and joy of true friendship",
                "style_notes": "Focus on the small gestures that show friendship: sharing, helping, being there. Warm dialogue between friends. Cozy settings like treehouse hideouts or blanket forts. The friend could be a stuffed animal, imaginary companion, or creature.",
                "pacing": "Warm and conversational, like chatting with a best friend"
            },
            # 3. Magical Bedtime Adventure
            {
                "name": "Magical Bedtime Adventure",
                "description": "gentle adventures that happen in the quiet hours when the world is asleep",
                "style_notes": "The adventure comes to the child's bedroom or happens just outside. Magic that feels safe and wondrous. Discoveries made by moonlight. The adventure ends with returning to the warm bed.",
                "pacing": "Quietly adventurous, building gentle wonder"
            },
            # 4. Moonlight Dream Journey
            {
                "name": "Moonlight Dream Journey",
                "description": "dreamlike travels guided by moonlight and starshine",
                "style_notes": "The moon as a gentle guide or companion. Floating, drifting, gliding sensations. Silver and blue imagery. Visiting peaceful places bathed in moonlight. Reality gently blurs into dream.",
                "pacing": "Floaty and ethereal, like drifting on a cloud"
            },
            # 5. Curious Exploration Story
            {
                "name": "Curious Exploration Story",
                "description": "stories driven by gentle curiosity and the joy of discovery",
                "style_notes": "The child notices something small and wondrous. Following curiosity leads to peaceful discoveries. 'I wonder...' moments. Finding magic in ordinary things like a glowing flower or whispering stream.",
                "pacing": "Thoughtful and observant, like examining a snowflake"
            },
            # 6. Gentle Magical Mystery
            {
                "name": "Gentle Magical Mystery",
                "description": "soft mysteries with delightful, heartwarming solutions",
                "style_notes": "A small, cozy mystery: Where did the sparkles come from? Who left the tiny footprints? The mystery leads to meeting a friendly magical being. The answer is always something wonderful and kind.",
                "pacing": "Curious and gentle, building to a warm revelation"
            },
            # 7. Whimsical Imagination Story
            {
                "name": "Whimsical Imagination Story",
                "description": "playful, imaginative tales where anything is possible",
                "style_notes": "Surreal but soft imagery. Toys that come alive. Clouds that can be walked upon. Gentle absurdity that feels dreamlike. The rules of the world are flexible and kind.",
                "pacing": "Playful and dreamy, like a gentle daydream"
            },
            # 8. Nighttime Guardian Story
            {
                "name": "Nighttime Guardian Story",
                "description": "stories about gentle protectors who watch over sleeping children",
                "style_notes": "A guardian figure: a friendly star, a sleepy owl, a dream-keeper, or night fairy. They ensure the child is safe and peaceful. Emphasis on being watched over with love. The guardian helps with small nighttime worries.",
                "pacing": "Reassuring and protective, like a warm embrace"
            },
            # 9. Family Love Story
            {
                "name": "Family Love Story",
                "description": "tender stories celebrating the love between family members",
                "style_notes": "Moments of connection: a parent's goodnight kiss, a sibling's shared secret, grandparent's wisdom. Focus on feeling loved and belonging. Could include family traditions or special moments. Warmth and security.",
                "pacing": "Tender and intimate, like a bedtime hug"
            },
            # 10. Calm Rhythmic Story
            {
                "name": "Calm Rhythmic Story",
                "description": "lyrical storytelling with gentle, soothing rhythms",
                "style_notes": "Subtle rhythmic patterns in the prose. Soft repetition of calming phrases. Almost musical quality. Words chosen for their soothing sounds. Natural flow that lulls the listener.",
                "pacing": "Musical and flowing, like a gentle lullaby"
            },
            # 11. Tiny Hero Story
            {
                "name": "Tiny Hero Story",
                "description": "stories where small acts of kindness make a big difference",
                "style_notes": "The child helps someone or something small: a lost firefly, a worried mouse, a wilting flower. The help is gentle and kind. Small deeds ripple outward with positive effects. Empowerment through gentleness.",
                "pacing": "Empowering but gentle, building quiet pride"
            },
            # 12. Magical Creature Companion
            {
                "name": "Magical Creature Companion",
                "description": "stories featuring a special bond with a magical creature",
                "style_notes": "A unique magical friend: a sleepy dragon, a tiny phoenix, a cloud bunny. The creature has endearing quirks. They share a special moment or small adventure together. Deep connection and understanding.",
                "pacing": "Warm and bonding, like meeting a soulmate friend"
            },
            # 13. Wonder of Nature Story
            {
                "name": "Wonder of Nature Story",
                "description": "stories celebrating the gentle magic found in the natural world",
                "style_notes": "Natural phenomena as magical moments: dewdrops catching light, flowers blooming at night, gentle rain. The child connects with nature's rhythms. Seasons, weather, plants, and animals as gentle characters.",
                "pacing": "Observant and peaceful, like watching a sunset"
            },
            # 14. Cozy Village Story
            {
                "name": "Cozy Village Story",
                "description": "stories set in warm, friendly little communities",
                "style_notes": "A tiny village where everyone knows each other. Friendly shopkeepers, kind neighbors. Cobblestone streets, glowing windows, church bells. A sense of belonging and community warmth.",
                "pacing": "Homey and welcoming, like coming home"
            },
            # 15. Dreamland Adventure
            {
                "name": "Dreamland Adventure",
                "description": "journeys through the gentle landscapes of dreams",
                "style_notes": "Entering a dream world as the child falls asleep. Soft, impossible landscapes: cotton candy clouds, chocolate rivers, pillow mountains. Everything is safe and wonderful. Gentle adventures in sleep.",
                "pacing": "Surreal and soothing, like slipping into a dream"
            },
            # 16. Calm Bedtime Reflection
            {
                "name": "Calm Bedtime Reflection",
                "description": "quiet, contemplative stories perfect for winding down",
                "style_notes": "Reflective moments: watching the stars, thinking about the day, feeling grateful. Slow, peaceful observations. The child and a companion share quiet thoughts. Emphasis on stillness and contentment.",
                "pacing": "Very slow and meditative, like a deep breath"
            }
        ]
        
        # Select a random influencer for this story
        selected_influencer = random.choice(STORY_INFLUENCERS)
        logger.info(f"[STORY] Selected influencer: {selected_influencer['name']}")
        
        # ================== VARIATION CONTROL SYSTEM ==================
        # Rotate story elements to prevent repetitive storytelling
        # This significantly increases variety for returning users
        
        STORY_ARCS = ["Discovery", "Friendship", "Courage", "Mystery", "Journey", "Helping"]
        STORY_SETTINGS = [
            "enchanted forest", "magical ocean", "misty mountains", "lush jungle",
            "floating sky islands", "cozy village", "crystal caves", "moonlit meadow",
            "ancient library", "cloud kingdom", "starlit desert", "underwater kingdom",
            "train journey", "boat on calm river", "magical treehouse", "secret garden"
        ]
        MAGICAL_ELEMENTS = [
            "a friendly dragon", "a talking animal companion", "an enchanted glowing map",
            "a secret portal", "a wishing star", "a magical lantern", "a wise old owl",
            "a tiny fairy guide", "a gentle giant", "a shape-shifting cloud", 
            "a music box that plays dreams", "a compass that points to kindness"
        ]
        
        # Select varied elements (random selection for now - can be enhanced with user history tracking)
        selected_arc = random.choice(STORY_ARCS)
        selected_setting = random.choice(STORY_SETTINGS)
        selected_magical_element = random.choice(MAGICAL_ELEMENTS)
        
        logger.info(f"[STORY] Variation - Arc: {selected_arc}, Setting: {selected_setting}")
        logger.info(f"[STORY] Variation - Magical Element: {selected_magical_element}")
        
        # ================== LIGHT STORY CONTINUITY SYSTEM ==================
        # Creates a connected PillowTales world while keeping stories standalone
        # Occasionally includes familiar characters, locations, or light references
        
        PILLOWTALES_CHARACTERS = [
            {"name": "Luna the Moon Owl", "desc": "a wise silver owl who watches over sleeping children and knows the secrets of the night sky"},
            {"name": "Milo the Gentle Fox", "desc": "a curious and kind red fox with a fluffy tail who loves exploring and making new friends"},
            {"name": "Spark the Tiny Dragon", "desc": "a small, playful dragon no bigger than a cat, with warm golden scales and a gentle flame"},
            {"name": "Captain Cloud", "desc": "a jolly explorer who sails through the sky on a ship made of clouds, always ready for a gentle adventure"},
            {"name": "Pip the Dream Keeper", "desc": "a tiny glowing creature who collects happy dreams in a silver pouch and shares them with children"},
            {"name": "Willow the Wise Tree", "desc": "an ancient talking tree with gentle branches who tells stories and offers comfort to travelers"}
        ]
        
        PILLOWTALES_LOCATIONS = [
            {"name": "Whispering Forest", "desc": "a magical forest where the trees whisper gentle secrets and fireflies light the paths"},
            {"name": "Starfall Lake", "desc": "a peaceful lake where stars come to rest on the water's surface each night"},
            {"name": "Cloud Island", "desc": "a soft, floating island made entirely of clouds, where everything feels light and dreamy"},
            {"name": "The Moonlit Meadow", "desc": "a peaceful meadow that glows silver in the moonlight, where magical creatures gather"},
            {"name": "Cozy Hollow", "desc": "a warm, hidden valley where all the forest animals come to rest safely"},
            {"name": "The Dream Bridge", "desc": "a shimmering rainbow bridge that appears at twilight, connecting the waking world to dreamland"}
        ]
        
        # Decide whether to include continuity elements (about 40% of stories)
        include_continuity = random.random() < 0.4
        
        continuity_prompt = ""
        if include_continuity:
            # Randomly decide what type of continuity to include
            continuity_type = random.choice(["character", "location", "both"])
            
            if continuity_type == "character" or continuity_type == "both":
                selected_character = random.choice(PILLOWTALES_CHARACTERS)
                continuity_prompt += f"\n- Consider including {selected_character['name']} ({selected_character['desc']}) as a helper or guide in this story."
                logger.info(f"[STORY] Continuity - Including character: {selected_character['name']}")
            
            if continuity_type == "location" or continuity_type == "both":
                selected_location = random.choice(PILLOWTALES_LOCATIONS)
                continuity_prompt += f"\n- Consider setting part of the story in {selected_location['name']} ({selected_location['desc']})."
                logger.info(f"[STORY] Continuity - Including location: {selected_location['name']}")
        else:
            logger.info("[STORY] Continuity - None (standalone story)")
        
        # ================== SIGNATURE MAGIC RULE SYSTEM ==================
        # Each story includes one unique magical idea that makes the world special
        
        SIGNATURE_MAGIC_RULES = [
            "a forest where trees whisper ancient stories to those who listen quietly",
            "clouds that remember and replay the dreams of sleeping children",
            "stars that blink and twinkle when someone tells the truth",
            "rivers that glow with soft golden light when someone is brave",
            "fireflies that carry tiny glowing messages between friends",
            "flowers that hum gentle melodies when the moon rises",
            "stones that feel warm when held by someone with a kind heart",
            "shadows that dance and play when no one is watching",
            "rain that falls in colors when something magical is about to happen",
            "bridges that only appear for those who truly believe",
            "lanterns that light themselves when a child needs guidance",
            "leaves that turn silver and float upward to become stars",
            "waves that sing lullabies to anyone standing on the shore",
            "mountains that purr like giant sleeping cats",
            "paths that rearrange themselves to lead lost travelers home",
            "snowflakes that carry whispered wishes to the sky",
            "moonbeams that can be collected in jars and saved for dark nights",
            "echoes that remember and repeat kind words spoken long ago",
            "puddles that reflect not your face but your happiest memory",
            "wind that carries the sound of distant laughter from other dreamers"
        ]
        
        selected_magic_rule = random.choice(SIGNATURE_MAGIC_RULES)
        logger.info(f"[STORY] Signature Magic Rule: {selected_magic_rule}")
        
        # Get child's age for age-appropriate guidance
        age = request.age
        
        logger.info(f"[STORY] Age-appropriate guidance for age {age}")
        
        # ================== MASTER STORY GENERATION PROMPT ==================
        # Clean, comprehensive prompt for high-quality bedtime storytelling
        
        # Build age-specific language guidance with sentence density rules
        if age <= 4:
            age_language = """Age 3-4 LANGUAGE RULES:
• Use very simple vocabulary (words a 3-year-old knows)
• 1-2 sentences per page ONLY
• 5-8 words per sentence maximum
• Gentle repetition for comfort
• Clear, simple emotions
• Concrete, familiar objects (moon, stars, blanket, teddy bear)
• Soothing, rhythmic phrasing"""
        elif age <= 7:
            age_language = """Age 5-7 LANGUAGE RULES:
• Use simple, familiar vocabulary
• 2-4 sentences per page
• 8-12 words per sentence maximum
• Imaginative but understandable settings
• Friendly magical characters
• Clear story progression
• Avoid abstract concepts - keep descriptions concrete"""
        elif age <= 9:
            age_language = """Age 8-9 LANGUAGE RULES:
• Use richer but accessible vocabulary
• 3-5 sentences per page
• 10-14 words per sentence maximum
• More developed characters with personality
• Light adventure and discovery themes
• Can include gentle problem-solving"""
        else:
            age_language = """Age 10-11 LANGUAGE RULES:
• Use deeper vocabulary while remaining child-friendly
• 4-6 sentences per page
• 12-16 words per sentence maximum
• Stronger story arcs and character development
• Can include more complex emotions and themes
• Richer descriptive language"""
        
        # Log the story generation configuration for quality monitoring
        logger.info("[STORY] ========== STORY GENERATION CONFIG ==========")
        logger.info("[STORY] Structure: Hook → Discovery → Companion → Challenge → Hero → Resolution → Gentle Ending")
        logger.info(f"[STORY] Influencer style: {selected_influencer['name']}")
        logger.info(f"[STORY] Story arc: {selected_arc}")
        logger.info(f"[STORY] Setting: {selected_setting}")
        logger.info(f"[STORY] Magical element: {selected_magical_element}")
        logger.info(f"[STORY] Age group: {age} ({age_language.split(':')[0].strip()})")
        logger.info(f"[STORY] Theme: {effective_theme}, Moral: {request.moral}")
        logger.info(f"[STORY] Gender/Pronouns: {gender} ({'she/her' if gender == 'girl' else 'he/him' if gender == 'boy' else 'they/them'})")
        logger.info(f"[STORY] Continuity: {'Yes - ' + continuity_prompt.strip() if continuity_prompt else 'No (standalone)'}")
        logger.info("[STORY] ================================================")
        
        # Create the master system message
        system_message = f"""You are a professional children's storyteller and author who creates magical bedtime stories for children aged 3-11.

Your stories should feel like classic children's literature — warm, imaginative, adventurous, with a calm bedtime ending that helps children relax and fall asleep.

=== GOAL ===
Create stories that feel like they were written by professional children's authors and storytellers, combining imagination, warmth, gentle adventure, and peaceful endings.

=== SETTING RULE ===
Stories can take place ANYWHERE: forests, oceans, sky islands, magical towns, jungles, rivers, clouds, distant lands, pirate ships, space journeys, underwater kingdoms, train rides, or any imaginative setting.

Stories do NOT need to start in a bedroom.

However, stories must ALWAYS end with a calm, peaceful emotional tone suitable for bedtime.

Adventure is welcome — but the ending should feel safe, warm, and peaceful.

=== STORY REQUIREMENTS ===
- Child's name: {request.childName} (the hero of the story)
- Age: {request.age}
- Theme: {effective_theme}
- Moral lesson: {request.moral}
- Story length: approximately {request.durationMin} minutes when read aloud slowly
- Language: {story_language} (write the ENTIRE story in this language)
{continuation_prompt}{companion_prompt}{characters_prompt}
{gender_instruction}

=== STORYTELLING INFLUENCE ===
For this story, draw inspiration from: {selected_influencer['name']}
Style: {selected_influencer['description']}
{selected_influencer['style_notes']}

=== AGE-APPROPRIATE LANGUAGE ===
{age_language}

=== STORY STRUCTURE ===
Follow this emotional arc:

1. STRONG OPENING (First 1-2 sentences)
   Begin with a magical or intriguing hook that immediately captures the child's imagination.
   Set the scene in: {selected_setting}
   Example: "The moment {request.childName} touched the old oak tree, it began to glow softly..."

2. MAGICAL DISCOVERY
   {request.childName} discovers something unusual or magical that begins the adventure.
   This could be: a glowing path, mysterious creature, floating island, magical object, hidden map, etc.
   Include: {selected_magical_element}
   Build gentle wonder and curiosity.

3. MAGICAL COMPANION
   A friendly companion appears to guide or help {request.childName}.
   {continuity_prompt if continuity_prompt else "Create a warm, kind, supportive magical companion."}
   Give them personality and warmth. They help navigate the adventure together.

4. CHALLENGE MOMENT
   {request.childName} encounters a small obstacle or moment of uncertainty.
   This should NEVER feel scary — just a gentle challenge.
   Examples: solving a puzzle, crossing a glowing bridge, helping a friend, finding a hidden path.

5. HERO MOMENT
   {request.childName} shows {request.moral} to overcome the challenge.
   The child demonstrates bravery, kindness, curiosity, or clever thinking.
   Make this moment feel earned and meaningful. {request.childName} should feel proud.

6. POSITIVE RESOLUTION
   The situation resolves in a satisfying way.
   Reinforce the positive value: {request.moral}
   The world feels safe, kind, and magical.

7. GENTLE BEDTIME ENDING
   Slow the story emotionally.
   End with a calm, peaceful closing moment suitable for bedtime.
   Examples: moonlight, quiet stars, gentle wind, rocking water, returning safely home, a warm goodbye.
   The child should feel calm, safe, and ready to drift toward sleep.

=== STORY PACING RULE ===
Write in short, rhythmic segments suitable for narration.
{age_language}
Keep narration calm, digestible, and age-appropriate.
Avoid long paragraphs or overly complex descriptions.

=== STORYTELLING VOICE ===
Stories should be written in the voice of a warm bedtime storyteller speaking gently to a child.
The narration should feel imaginative, comforting, and magical — never mechanical or factual.

VOICE GUIDELINES:
• Use warm, descriptive language that paints vivid pictures
• Include gentle sensory imagery (soft moonlight, warm breezes, twinkling stars, rustling leaves, flowing water)
• Write with natural storytelling rhythm — vary sentence length and flow
• Make the child feel like they're being told a story by someone who loves them
• Create moments of quiet wonder that spark imagination
• The story should feel like a storyteller sharing a magical bedtime adventure

TONE & TEXTURE:
• Imaginative and dreamy, not educational or preachy
• Comforting and reassuring, never anxious or rushed
• Magical and wondrous, not ordinary or mundane
• Personal and intimate, like a whispered secret

AVOID:
• Robotic or repetitive phrasing ("Then... Then... Then...")
• Mechanical sentence structures
• Overly complex or academic language
• Listing facts or explaining things didactically
• Frightening, overwhelming, or stimulating moments

=== RECURRING PILLOWTALES CHARACTERS ===
You may include these beloved characters as companions:
• Luna the Moon Owl - wise silver owl who watches over sleeping children
• Milo the Gentle Fox - curious, kind fox who loves exploring
• Spark the Tiny Dragon - playful dragon with warm golden scales
• Captain Cloud - jolly explorer sailing on a cloud ship
• Pip the Dream Keeper - tiny glowing creature who collects happy dreams
• Willow the Wise Tree - ancient talking tree who tells stories

=== MAGICAL PILLOWTALES LOCATIONS ===
Familiar places that may appear:
• Whispering Forest - where trees whisper secrets and fireflies light paths
• Starfall Lake - peaceful lake where stars rest on the water
• Cloud Island - soft floating island made of clouds
• The Moonlit Meadow - glows silver under moonlight
• Cozy Hollow - warm hidden valley for resting
• The Dream Bridge - shimmering rainbow bridge to dreamland

=== LIGHT STORY CONTINUITY ===
Stories must ALWAYS work as complete standalone bedtime stories.
Any references to recurring characters or locations should be light and not require prior knowledge.

=== NARRATION MARKERS ===
Use sparingly (2-4 total) to guide the narrator's voice:
[whisper] - For magical, quiet moments
[softly] - For tender, sleepy moments
Use especially in the final pages.

=== OUTPUT FORMAT ===
CRITICAL: Respond with ONLY valid JSON in this exact format:
{{
  "title": "Story title in {story_language}",
  "pages": ["Page 1 text...", "Page 2 text...", "Page 3 text..."]
}}

PAGE GUIDELINES:
- Create {"8-9" if request.durationMin == 8 else "11-12" if request.durationMin == 11 else "15-17"} pages
- Story length: approximately {"1000-1200" if request.durationMin == 8 else "1400-1600" if request.durationMin == 11 else "1900-2200"} words total
- Each page: 8-12 sentences maximum (maintains good pacing for bedtime reading)
- Language should be clear, warm, and engaging
- FINAL PAGE must create a peaceful, calming emotional ending

=== FINAL GOAL ===
Create a magical bedtime adventure told by a caring storyteller — combining imagination, gentle adventure, emotional warmth, and a peaceful ending that helps {request.childName} relax and fall asleep.

Respond with ONLY the JSON, no other text."""

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""
        {system_message}

        Generate a bedtime story for {request.childName}, age {request.age}, with theme '{request.theme}' and moral '{request.moral}'. Write entirely in {story_language}.
        """
        response = model.generate_content(prompt)
        response_text = response.text
               
        logger.info(f"Gemini response: {response_text}")
        
        # Parse the JSON response
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            story_data = json.loads(cleaned_response)
            
            if "title" not in story_data or "pages" not in story_data:
                raise ValueError("Invalid story format")
            
            return story_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {str(e)}")
            logger.error(f"Response was: {response}")
            raise HTTPException(status_code=500, detail="Failed to generate story - invalid format")
            
    except Exception as e:
        logger.error(f"Error generating story with OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate story: {str(e)}")

# ================== TTS Text Normalization ==================

def normalize_text_for_tts(text: str) -> str:
    """
    Normalize text for better TTS narration quality.
    Applies rules to improve pacing and natural flow.
    """
    import re
    
    result = text
    changes_made = []
    
    # ============================================================
    # ACTION TEXT CLEANUP - Convert *actions* to natural narration
    # ============================================================
    # These patterns convert stage directions to natural descriptions
    # for smoother bedtime story narration
    
    action_conversions = {
        r'\*yawns?\*': 'yawned softly...',
        r'\*sighs?\*': 'sighed gently...',
        r'\*giggles?\*': 'giggled softly...',
        r'\*laughs?\*': 'laughed warmly...',
        r'\*whispers?\*': 'whispered...',
        r'\*gasps?\*': 'gasped softly...',
        r'\*stretche?s?\*': 'stretched...',
        r'\*sniffles?\*': 'sniffled...',
        r'\*smiles?\*': 'smiled...',
        r'\*nods?\*': 'nodded...',
        r'\*hugs?\*': 'hugged warmly...',
        r'\*yawning\*': 'yawning softly...',
        r'\*sighing\*': 'sighing gently...',
        r'\*giggling\*': 'giggling...',
        r'\*laughing\*': 'laughing...',
        r'\*whispering\*': 'whispering...',
    }
    
    for pattern, replacement in action_conversions.items():
        before = result
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if before != result:
            changes_made.append(f"action: {pattern} -> {replacement}")
    
    # Generic fallback: convert any remaining *action* to "action..."
    # This handles cases we didn't explicitly define
    before = result
    result = re.sub(r'\*([a-zA-Z]+(?:ing|s|ed)?)\*', r'\1...', result)
    if before != result:
        changes_made.append("action: generic *action* -> action...")
    
    # ============================================================
    # PUNCTUATION AND PACING CLEANUP
    # ============================================================
    
    # 1. Replace em dashes (—) with commas for natural pauses
    # "Emily walked forward — slowly" -> "Emily walked forward, slowly"
    result = re.sub(r'\s*—\s*', ', ', result)
    result = re.sub(r'\s*–\s*', ', ', result)  # en-dash too
    
    # 2. OXFORD COMMA REMOVAL - Multiple patterns for robustness
    # Pattern 1: Standard Oxford comma ", and" (with any whitespace)
    # "the lion, monkey, and hippo" -> "the lion, monkey and hippo"
    before = result
    result = re.sub(r',\s+(and)\b', r' \1', result, flags=re.IGNORECASE)
    if before != result:
        oxford_count = len(re.findall(r',\s+and\b', before, re.IGNORECASE))
        changes_made.append(f"oxford comma (and): removed {oxford_count}")
    
    # Pattern 2: Oxford comma with "or"
    # "red, green, or blue" -> "red, green or blue"
    before = result
    result = re.sub(r',\s+(or)\b', r' \1', result, flags=re.IGNORECASE)
    if before != result:
        oxford_count = len(re.findall(r',\s+or\b', before, re.IGNORECASE))
        changes_made.append(f"oxford comma (or): removed {oxford_count}")
    
    # 3. Ensure spacing after commas
    result = re.sub(r',(\w)', r', \1', result)
    
    # 4. Normalize quotes to standard double quotes
    result = result.replace('"', '"').replace('"', '"')
    result = result.replace(''', "'").replace(''', "'")
    
    # 5. Remove duplicate punctuation
    result = re.sub(r',,+', ',', result)
    result = re.sub(r'\.\.\.\.+', '...', result)  # Keep max 3 dots
    result = re.sub(r'\.\.(?!\.)', '.', result)   # Two dots -> one dot
    result = re.sub(r'\?\?+', '?', result)
    result = re.sub(r'!!+', '!', result)
    
    # 6. Replace parentheses with commas for smoother reading
    # "(text)" -> ", text,"
    result = re.sub(r'\s*\(([^)]+)\)\s*', r', \1, ', result)
    
    # 7. Clean up multiple spaces
    result = re.sub(r'  +', ' ', result)
    
    # 8. Clean up comma-space-comma patterns
    result = re.sub(r',\s*,', ',', result)
    
    # 9. Clean up leading/trailing commas in sentences
    result = re.sub(r'^\s*,\s*', '', result)
    result = re.sub(r'\s*,\s*$', '', result)
    
    # Log what was changed
    if changes_made:
        logger.info(f"[TTS-NORMALIZE] Changes: {', '.join(changes_made)}")
    logger.info(f"[TTS-NORMALIZE] Text normalized: {len(text)} -> {len(result)} chars")
    
    return result.strip()

# ================== Translation Helper ==================

async def translate_text_for_narration(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate story text from source language to target language using Gemini.
    Used when narration language differs from story language.
    """
    if source_lang == target_lang:
        return text

    source_name = SUPPORTED_LANGUAGES.get(source_lang, "English")
    target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

    logger.info(f"[TRANSLATE] Translating {len(text)} chars from {source_name} to {target_name}")

    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            f"Translate the following children's bedtime story from {source_name} to {target_name}.\n\n"
            "CRITICAL RULES:\n"
            "1. Preserve the calming, gentle tone suitable for bedtime\n"
            "2. Keep the story structure and flow intact\n"
            "3. Maintain any character names as-is (don't translate names)\n"
            f"4. Ensure the translation is natural and fluent in {target_name}\n"
            "5. Output ONLY the translated text, no explanations or notes\n\n"
            f"{text}"
        )

        response = model.generate_content(prompt)
        translated_text = getattr(response, "text", None)

        if not translated_text or not isinstance(translated_text, str) or not translated_text.strip():
            logger.warning("[TRANSLATE] Invalid response, falling back to original text")
            return text

        logger.info(f"[TRANSLATE] Success: {len(text)} chars -> {len(translated_text)} chars")
        return translated_text.strip()

    except Exception as e:
        logger.error(f"[TRANSLATE] Failed: {str(e)}")
        return text

def clean_text_for_narration(text: str) -> str:
    """
    Clean up text for smoother TTS narration.
    Removes unnecessary punctuation that sounds unnatural when spoken.
    """
    import re
    
    original_len = len(text)
    result = text
    changes = []
    
    # Rule 1: Remove Oxford comma before "and" when not joining independent clauses
    # Pattern: ", and [verb]" → " and [verb]"
    # This catches cases like "visited, and left" → "visited and left"
    oxford_comma_pattern = r',\s+and\s+([a-z]+ed|[a-z]+ing|[a-z]+s)\b'
    if re.search(oxford_comma_pattern, result, re.IGNORECASE):
        result = re.sub(oxford_comma_pattern, r' and \1', result, flags=re.IGNORECASE)
        changes.append("removed Oxford comma before 'and [verb]'")
    
    # Rule 2: Remove comma before "and" at end of lists when followed by a verb phrase
    # "looked up, and saw" → "looked up and saw"
    list_comma_pattern = r'(\w+),\s+and\s+(saw|found|felt|heard|noticed|realized|discovered|began|started)'
    if re.search(list_comma_pattern, result, re.IGNORECASE):
        result = re.sub(list_comma_pattern, r'\1 and \2', result, flags=re.IGNORECASE)
        changes.append("removed comma before 'and [verb]'")
    
    if changes:
        logger.info(f"[NARRATION-CLEANUP] Applied: {', '.join(changes)}")
        logger.info(f"[NARRATION-CLEANUP] Text cleaned: {original_len} -> {len(result)} chars")
    
    return result


async def extract_story_metadata(story_text: str, title: str) -> dict:
    """
    Extract story metadata for continuation feature.
    Returns: {summary, characters, setting}
    """
    logger.info(f"[METADATA] Extracting metadata from story: {title}")
    try:
        system_message = """You are a story analyst. Extract key information from children's bedtime stories."""

    Return ONLY valid JSON in this exact format:
    {
    "summary": "2-3 sentence recap of the story plot",
    "characters": [
    {"name": "Character Name", "description": "brief description", "role": "protagonist/friend/mentor/etc"}
     ],
    "setting": "Description of the story world/location"
    }

    Keep descriptions brief and child-friendly. Focus on elements that would help continue the story tomorrow."""

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""
    {system_message}

    Story Title: {title}

    Story Text:
    {story_text}
    """

        response = model.generate_content(prompt)
        response_text = response.text

        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            metadata = json.loads(json_match.group())
            logger.info(f"[METADATA] Extracted: {len(metadata.get('characters', []))} characters, summary length: {len(metadata.get('summary', ''))}")
            return metadata
        else:
            logger.warning("[METADATA] Could not parse JSON from response")
            return {"summary": "", "characters": [], "setting": ""}

    except Exception as e:
        logger.error(f"[METADATA] Extraction failed: {str(e)}")
        return {"summary": "", "characters": [], "setting": ""}

    # ================== API Endpoints ==================

    @api_router.get("/")
    async def root():
    return {"message": "PillowTales API is running"}

    @api_router.get("/debug/supabase-role")
    async def debug_supabase_role():
    """Debug endpoint to check Supabase connection and role"""
    try:
        # Try to read from users_profile to check read access
        read_result = supabase.table('users_profile').select('id').limit(1).execute()
        can_read = len(read_result.data) if read_result.data else 0
        
        # Try to check stories table
        stories_result = supabase.table('stories').select('id').limit(1).execute()
        can_read_stories = len(stories_result.data) if stories_result.data else 0
        
        return {
            "status": "connected",
            "supabase_url": supabase_url[:50] + "...",
            "service_key_prefix": supabase_key[:20] + "...",
            "can_read_users_profile": can_read,
            "can_read_stories": can_read_stories,
            "message": "Service role should bypass RLS"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "supabase_url": supabase_url[:50] + "...",
            "service_key_prefix": supabase_key[:20] + "..."
        }

    @api_router.post("/debug/test-insert")
    async def debug_test_insert():
    """Debug endpoint to test INSERT into stories table"""
    import uuid
    test_user_id = "48b46b9c-8821-458b-9e4b-6f7695ae7767"  # A known user ID
    
    test_record = {
        "user_id": test_user_id,
        "title": "Test Story - DELETE ME",
        "child_name": "Test",
        "age": 5,
        "theme": "test",
        "moral": "test",
        "calm_level": "calm",
        "duration_min": 3,
        "language": "en",
        "pages": ["Page 1", "Page 2"],
        "full_text": "Page 1 Page 2",
        "audio_url": None,
        "is_favorite": False,
        "created_at": datetime.utcnow().isoformat()
    }
    
    try:
        result = supabase.table('stories').insert(test_record).execute()
        
        # Delete the test record immediately
        if result.data and len(result.data) > 0:
            story_id = result.data[0]['id']
            supabase.table('stories').delete().eq('id', story_id).execute()
            return {
                "status": "success",
                "message": f"INSERT worked! Created and deleted story {story_id}",
                "story_id": story_id
            }
        else:
            return {
                "status": "unknown",
                "message": "INSERT returned no data",
                "result": str(result)
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

    @api_router.get("/debug/test-signed-url")
    async def debug_test_signed_url(story_id: str):
    """Debug endpoint to test signed URL generation WITHOUT auth"""
    try:
        # Get the storage path from the story
        story_result = supabase.table('stories').select('audio_url, user_id').eq('id', story_id).execute()
        
        if not story_result.data:
            return {"error": "Story not found"}
        
        story = story_result.data[0]
        storage_path = story.get('audio_url')
        
        if not storage_path:
            return {"error": "No audio_url in story", "story": story}
        
        # Check if it's a path or URL
        is_url = storage_path.startswith('http')
        
        if is_url:
            return {
                "error": "audio_url contains a full URL instead of path",
                "audio_url": storage_path,
                "needs_fix": True
            }
        
        # Try to create signed URL
        logger.info(f"[DEBUG] Creating signed URL for path: {storage_path}")
        signed_url_result = supabase.storage.from_('story-audio').create_signed_url(storage_path, 3600)
        
        return {
            "status": "success",
            "storage_path": storage_path,
            "signed_url_response": str(signed_url_result),
            "signed_url": signed_url_result.get('signedURL') or signed_url_result.get('signed_url') or signed_url_result.get('signedUrl'),
            "raw_keys": list(signed_url_result.keys()) if isinstance(signed_url_result, dict) else None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

    @api_router.post("/auth/signup", response_model=AuthResponse)
    async def signup(request: SignupRequest):
    """Sign up a new user"""
    try:
        # Create user with Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password
        })
        
        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Failed to create auth user")
        
        user_id = auth_response.user.id
        
        # Create user profile
        user_data = {
            "id": user_id,
            "preferred_language": request.preferredLanguage,
            "bedtime_mode": False,
            "plan": "free"
        }
        
        result = supabase.table('users_profile').insert(user_data).execute()
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        user = result.data[0]
        
        # Create JWT token
        token = create_access_token({"user_id": user['id'], "email": user['email']})
        
        return AuthResponse(
            token=token,
            userId=user['id'],
            email=user['email'],
            preferredLanguage=user['preferred_language']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@api_router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login an existing user"""
    try:
        # Use Supabase Auth to login
        auth_response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if not auth_response.user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        user_id = auth_response.user.id
        
        # Get user profile
        result = supabase.table('users_profile').select('*').eq('id', user_id).execute()
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        user = result.data[0]
        
        # Create JWT token (password already verified by Supabase Auth)
        token = create_access_token({"user_id": user['id'], "email": auth_response.user.email})
        
        return AuthResponse(
            token=token,
            userId=user['id'],
            email=auth_response.user.email,
            preferredLanguage=user.get('preferred_language', 'en')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@api_router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages for story and narration"""
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in SUPPORTED_LANGUAGES.items()
        ],
        "voices": {
            code: voice["name"]
            for code, voice in ELEVENLABS_VOICES.items()
        }
    }

# ================== SUBSCRIPTION ENDPOINTS ==================

@api_router.get("/subscription")
async def get_subscription_status(user_id: str = Depends(get_current_user)):
    # Get user's subscription status, usage, and available features.
    # Get user's email for tester check
    user_email = None
    try:
        user_data = supabase.table('users_profile').select('email').eq('id', user_id).execute()
        if user_data.data:
            user_email = user_data.data[0].get('email')
        else:
            # Try to get from auth.users if not in profile
            auth_user = supabase.auth.admin.get_user_by_id(user_id)
            if auth_user and auth_user.user:
                user_email = auth_user.user.email
    except Exception as e:
        logger.warning(f"[SUB] Could not get user email: {e}")
    
    subscription = await get_user_subscription(user_id, user_email)
    
    # Get tier config
    tier = "premium" if subscription["is_premium"] else "free"
    tier_config = SUBSCRIPTION_TIERS.get(tier, SUBSCRIPTION_TIERS["free"])
    
    # Get available narrators and companions for this tier
    available_narrators = []
    for narrator_id in tier_config.get("narrators", []):
        if narrator_id in VOICE_PRESETS:
            narrator = VOICE_PRESETS[narrator_id]
            available_narrators.append({
                "id": narrator_id,
                "name": narrator["name"],
                "icon": narrator["icon"],
                "description": narrator["description"],
            })
    
    available_companions = []
    for comp_id in tier_config.get("companions", []):
        if comp_id in STORY_COMPANIONS:
            comp = STORY_COMPANIONS[comp_id]
            available_companions.append({
                "id": comp_id,
                "name": comp["name"],
                "icon": comp["icon"],
            })
    
    return {
        "subscription": subscription,
        "tier": tier,
        "tier_name": tier_config.get("name", "Free"),
        "features": {
            "unlimited_text_stories": tier_config.get("unlimited_text_stories", True),
            "daily_narration_limit": tier_config.get("daily_narration_limit"),
            "parent_voice": tier_config.get("parent_voice", False),
            "priority_generation": tier_config.get("priority_generation", False),
        },
        "available_narrators": available_narrators,
        "available_companions": available_companions,
        "pricing": PREMIUM_PRICING,
    }

@api_router.get("/subscription/check-feature")
async def check_feature(
    feature: str,
    item_id: str = None,
    user_id: str = Depends(get_current_user)
):
    """
    Check if user can access a specific feature.
    Features: narrator, companion, parent_voice, narration
    """
    subscription = await get_user_subscription(user_id)
    result = check_feature_access(subscription, feature, item_id)
    return result

@api_router.post("/subscription/upgrade")
async def start_upgrade(user_id: str = Depends(get_current_user)):
    """
    Start premium upgrade flow (placeholder for payment integration).
    In production, this would redirect to Stripe/RevenueCat.
    """
    # For now, return upgrade info
    subscription = await get_user_subscription(user_id)
    
    return {
        "current_status": subscription["status"],
        "pricing": PREMIUM_PRICING,
        "trial_available": not subscription.get("trial_used", False),
        "upgrade_url": None,  # Would be Stripe checkout URL in production
        "message": "Premium upgrade coming soon!",
    }

@api_router.get("/voices")
async def get_voice_presets(user_id: str = Depends(get_current_user)):
    """Get available narrator personalities for bedtime stories"""
    
    # Check if user has parent voice set up
    parent_voice_id = None
    parent_voice_status = 'none'
    try:
        user_result = supabase.table('users_profile').select('parent_voice_id, parent_voice_status').eq('id', user_id).execute()
        if user_result.data and len(user_result.data) > 0:
            parent_voice_id = user_result.data[0].get('parent_voice_id')
            parent_voice_status = user_result.data[0].get('parent_voice_status', 'none') or 'none'
    except Exception as e:
        logger.warning(f"[VOICES] Error fetching parent voice status: {e}")
    
    # Build narrator list from presets
    narrators = []
    for preset_id, preset in VOICE_PRESETS.items():
        narrator_info = {
            "id": preset_id,
            "name": preset["name"],
            "description": preset["description"],
            "icon": preset["icon"],
            "provider": preset["provider"],
            "is_premium": preset.get("is_premium", False),
            "requires_setup": preset.get("requires_setup", False),
            "personality": preset.get("personality", "default"),
            "language_code": preset.get("language_code", "en"),  # Language code for filtering
        }
        
        # Handle Parent Voice specially - check if user has recorded it
        if preset_id == "parent_voice":
            is_ready = parent_voice_id is not None and parent_voice_status == 'ready'
            narrator_info["is_ready"] = is_ready
            narrator_info["voice_id"] = parent_voice_id
            narrator_info["status"] = parent_voice_status
            # Show helpful description based on status
            if not is_ready:
                narrator_info["description"] = "Record your voice to read stories"
        else:
            narrator_info["is_ready"] = True
        
        narrators.append(narrator_info)
    
    return {
        "narrators": narrators,
        "default_narrator": DEFAULT_NARRATOR,
        "has_parent_voice": parent_voice_id is not None and parent_voice_status == 'ready',
        "parent_voice_status": parent_voice_status,
    }

@api_router.get("/companions")
async def get_story_companions(user_id: str = Depends(get_current_user)):
    """Get story companions - magical recurring characters in the story world"""
    
    # Get user's subscription and met companions
    met_companions = []
    is_premium = False
    try:
        result = supabase.table('users_profile').select('met_companions, subscription_status').eq('id', user_id).execute()
        if result.data and len(result.data) > 0:
            met_companions = result.data[0].get('met_companions', []) or []
            is_premium = result.data[0].get('subscription_status') == 'premium'
    except Exception as e:
        logger.warning(f"[COMPANIONS] Could not fetch user data: {e}")
    
    # Get tier config
    tier = "premium" if is_premium else "free"
    tier_config = SUBSCRIPTION_TIERS.get(tier, SUBSCRIPTION_TIERS["free"])
    allowed_companions = tier_config.get("companions", ["luna_owl", "milo_fox"])
    
    # Build companion list with tier/lock info
    companions = []
    for comp_id, comp in STORY_COMPANIONS.items():
        is_locked = comp_id not in allowed_companions
        companion_info = {
            "id": comp_id,
            "name": comp["name"],
            "short_name": comp["short_name"],
            "icon": comp["icon"],
            "description": comp["description"],
            "personality": comp["personality"],
            "catchphrase": comp["catchphrase"],
            "themes": comp["themes"],
            "has_met": comp_id in met_companions,
            "tier": comp.get("tier", "free"),
            "is_locked": is_locked,
            "lock_message": "Part of the Premium Story World ✨" if is_locked else None,
        }
        companions.append(companion_info)
    
    # Sort: available first (met, then unmet), locked last
    companions.sort(key=lambda x: (x["is_locked"], not x["has_met"], x["name"]))
    
    return {
        "companions": companions,
        "met_count": len(met_companions),
        "total_count": len(STORY_COMPANIONS),
        "available_count": len(allowed_companions),
        "is_premium": is_premium,
        "appearance_chance": COMPANION_APPEARANCE_CHANCE,
    }

@api_router.get("/user/profile", response_model=UserProfileResponse)
async def get_user_profile(user_id: str = Depends(get_current_user)):
    """Get user profile with streak and usage statistics"""
    try:
        profile = await get_user_profile_with_stats(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return UserProfileResponse(
            id=profile['id'],
            email=profile['email'],
            plan=profile['plan'],
            preferred_language=profile['preferred_language'],
            streak_count=profile['streak_count'],
            last_story_date=profile['last_story_date'],
            stories_this_week=profile['stories_this_week'],
            stories_saved=profile['stories_saved'],
            can_generate=profile['can_generate'],
            can_save_more=profile['can_save_more']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user profile")

@api_router.post("/generateStory", response_model=StoryResponse)
async def generate_story(request: GenerateStoryRequest, user_id: str = Depends(get_current_user)):
    """Generate a personalized bedtime story with optional continuation support"""
    print("===========================================")
    print("[STORY] /api/generateStory HIT")
    print(f"[STORY] user_id from auth: {user_id}")
    print(f"[STORY] request.userId: {request.userId}")
    print(f"[STORY] continueFromStoryId: {request.continueFromStoryId}")
    print("===========================================")
    
    try:
        # Verify user_id matches the authenticated user
        if request.userId != user_id:
            print(f"[STORY] ERROR: user_id mismatch! auth={user_id}, request={request.userId}")
            raise HTTPException(status_code=403, detail="Unauthorized: user_id mismatch")
        
        # Get user's plan and email
        profile_result = supabase.table('users_profile').select('plan, email').eq('id', user_id).execute()
        plan = 'free'
        user_email = None
        if profile_result.data and len(profile_result.data) > 0:
            plan = profile_result.data[0].get('plan', 'free')
            user_email = profile_result.data[0].get('email')
        
        # Check story limits based on plan (testers get unlimited access)
        limits_check = await check_story_limits(user_id, plan, user_email)
        
        if not limits_check['can_generate']:
            raise HTTPException(
                status_code=403,
                detail=json.dumps({
                    "error": "PAYWALL", 
                    "reason": limits_check['reason'],
                    "message": limits_check.get('message', "Story limit reached. Upgrade to Premium for unlimited stories!")
                })
            )
        
        # Check if user can save more stories (for free tier)
        if not limits_check['can_save']:
            raise HTTPException(
                status_code=403,
                detail=json.dumps({
                    "error": "STORAGE_LIMIT", 
                    "reason": "storage_limit",
                    "message": "You've reached the maximum number of saved stories. Delete some stories or upgrade to Premium!"
                })
            )
        
        # ==================== CONTINUATION HANDLING ====================
        continuation_context = None
        story_arc_id = None
        part_number = 1
        continues_from_id = None
        
        if request.continueFromStoryId:
            logger.info(f"[STORY] Fetching continuation context from story: {request.continueFromStoryId}")
            
            # Fetch the previous story for context
            prev_story_result = supabase.table('stories').select(
                'id, title, story_summary, characters, setting, story_arc_id, part_number, user_id'
            ).eq('id', request.continueFromStoryId).execute()
            
            if prev_story_result.data and len(prev_story_result.data) > 0:
                prev_story = prev_story_result.data[0]
                
                # Verify the previous story belongs to this user
                if prev_story.get('user_id') != user_id:
                    raise HTTPException(status_code=403, detail="Cannot continue a story that doesn't belong to you")
                
                prev_part_number = prev_story.get('part_number', 1) or 1
                
                # Check if we've reached the max parts (5)
                if prev_part_number >= MAX_STORY_ARC_PARTS:
                    raise HTTPException(
                        status_code=400,
                        detail=json.dumps({
                            "error": "ARC_COMPLETE",
                            "message": f"This story arc is complete! {request.childName}'s adventure has reached its grand finale. Start a new story tonight!"
                        })
                    )
                
                # Build continuation context
                continuation_context = {
                    'title': prev_story.get('title', ''),
                    'summary': prev_story.get('story_summary', ''),
                    'characters': prev_story.get('characters', []),
                    'setting': prev_story.get('setting', ''),
                    'part_number': prev_part_number
                }
                
                # Use existing arc_id or create new one from the first story
                story_arc_id = prev_story.get('story_arc_id') or prev_story.get('id')
                part_number = prev_part_number + 1
                continues_from_id = request.continueFromStoryId
                
                logger.info(f"[STORY] Continuing arc {story_arc_id}, Part {part_number}")
            else:
                logger.warning(f"[STORY] Previous story {request.continueFromStoryId} not found, creating new story")
        
        # ==================== COMPANION SELECTION ====================
        # Use user-selected companion if provided, otherwise random selection
        selected_companion = None
        companion_id = None
        companion_name = None
        
        if request.companionId:
            # User selected a specific companion - use it
            if request.companionId in STORY_COMPANIONS:
                comp_data = STORY_COMPANIONS[request.companionId]
                selected_companion = {
                    'id': request.companionId,
                    'name': comp_data.get('name'),
                    'short_name': comp_data.get('short_name'),
                    'description': comp_data.get('description'),
                    'personality': comp_data.get('personality'),
                    'appearance': comp_data.get('appearance'),
                    'catchphrase': comp_data.get('catchphrase'),
                    'themes': comp_data.get('themes', []),
                    'is_new': False,  # User explicitly selected, so not a "surprise"
                }
                companion_id = request.companionId
                companion_name = comp_data.get('name')
                logger.info(f"[STORY] User selected companion: {companion_name}")
            else:
                logger.warning(f"[STORY] Unknown companion ID: {request.companionId}")
        else:
            # No companion selected - use random selection based on probability
            selected_companion = select_story_companion(user_id, request.theme)
            if selected_companion:
                companion_id = selected_companion.get('id')
                companion_name = selected_companion.get('name')
                logger.info(f"[STORY] Random companion selected: {companion_name} (is_new: {selected_companion.get('is_new')})")
        
        if not selected_companion:
            logger.info("[STORY] No companion for this story")
        
        # Generate story using OpenAI (with continuation context and companion if available)
        story_data = await generate_story_with_openai(request, continuation_context, selected_companion)
        
        # ==================== APPEND BEDTIME RITUAL ENDING ====================
        # Add personalized goodnight message to every story
        story_data["pages"] = append_bedtime_ritual_to_story(
            story_data["pages"], 
            request.childName, 
            request.storyLanguageCode
        )
        
        # Save story to database
        full_text = " ".join(story_data["pages"])
        
        # ==================== STORY METRICS LOGGING ====================
        story_chars = len(full_text)
        page_count = len(story_data["pages"])
        avg_chars_per_page = story_chars // page_count if page_count > 0 else 0
        estimated_tts_cost_openai = (story_chars / 1000) * TTS_COST_PER_1K["openai"]
        estimated_tts_cost_eleven = (story_chars / 1000) * TTS_COST_PER_1K["elevenlabs"]
        
        logger.info("=" * 60)
        logger.info("[STORY METRICS]")
        logger.info(f"  title: {story_data['title']}")
        logger.info(f"  story_chars: {story_chars}")
        logger.info(f"  page_count: {page_count}")
        logger.info(f"  avg_chars_per_page: {avg_chars_per_page}")
        logger.info(f"  story_language: {request.storyLanguageCode}")
        logger.info(f"  narration_language: {request.narrationLanguageCode}")
        logger.info(f"  is_continuation: {continuation_context is not None}")
        logger.info(f"  part_number: {part_number}")
        logger.info(f"  story_arc_id: {story_arc_id}")
        logger.info(f"  estimated_tts_cost_openai: ${estimated_tts_cost_openai:.4f}")
        logger.info(f"  estimated_tts_cost_eleven: ${estimated_tts_cost_eleven:.4f}")
        logger.info("=" * 60)
        
        # Determine narration language (default to story language if not specified)
        narration_lang = request.narrationLanguageCode if request.narrationLanguageCode else request.storyLanguageCode
        
        # Generate new arc_id for first story in an arc
        import uuid
        if not story_arc_id:
            story_arc_id = str(uuid.uuid4())
        
        story_record = {
            "user_id": request.userId,
            "title": story_data["title"],
            "child_name": request.childName,
            "age": request.age,
            "theme": request.theme,
            "moral": request.moral,
            "calm_level": request.calmLevel,
            "duration_min": request.durationMin,
            "language": request.storyLanguageCode,
            "story_language_code": request.storyLanguageCode,
            "narration_language_code": narration_lang,
            "pages": story_data["pages"],
            "full_text": full_text,
            "audio_url": None,
            "is_favorite": False,
            "created_at": datetime.utcnow().isoformat(),
            # Continuation fields
            "is_continuation": continuation_context is not None,
            "continues_from_story_id": continues_from_id,
            "story_arc_id": story_arc_id,
            "part_number": part_number,
            # Companion fields
            "companion_id": companion_id,
            "companion_name": companion_name,
        }
        
        # Update user's met_companions if this is a new companion
        if selected_companion and selected_companion.get('is_new'):
            try:
                # Get current met_companions
                profile_result = supabase.table('users_profile').select('met_companions').eq('id', user_id).execute()
                met_companions = []
                if profile_result.data and len(profile_result.data) > 0:
                    met_companions = profile_result.data[0].get('met_companions', []) or []
                
                # Add new companion if not already met
                if companion_id not in met_companions:
                    met_companions.append(companion_id)
                    supabase.table('users_profile').update({
                        'met_companions': met_companions
                    }).eq('id', user_id).execute()
                    logger.info(f"[STORY] User met new companion: {companion_name}. Total met: {len(met_companions)}")
            except Exception as comp_err:
                logger.warning(f"[STORY] Could not update met_companions: {comp_err}")
        
        print(f"[STORY] Inserting story record with user_id: {request.userId}")
        print(f"[STORY] Story record: {json.dumps({k: v for k, v in story_record.items() if k != 'full_text' and k != 'pages'}, default=str)}")
        
        # Try to insert the story
        result = None
        insert_error = None
        
        try:
            result = supabase.table('stories').insert(story_record).execute()
            print(f"[STORY] Insert result: {result}")
        except Exception as e1:
            print(f"[STORY] INSERT attempt 1 failed: {e1}")
            insert_error = e1
            
            try:
                from supabase import create_client
                fresh_supabase = create_client(supabase_url, supabase_key)
                result = fresh_supabase.table('stories').insert(story_record).execute()
                print("[STORY] Insert with fresh client succeeded")
                insert_error = None
            except Exception as e2:
                print(f"[STORY] INSERT attempt 2 also failed: {e2}")
        
        if insert_error:
            error_str = str(insert_error)
            if 'row-level security' in error_str.lower() or '42501' in error_str:
                raise HTTPException(
                    status_code=500, 
                    detail=json.dumps({
                        "error": "RLS_ERROR",
                        "message": "Database security policy is blocking story creation. Please run the RLS fix script in your Supabase dashboard.",
                        "technical": error_str[:200]
                    })
                )
            raise HTTPException(status_code=500, detail=f"Failed to save story: {error_str}")
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to save story")
        
        saved_story = result.data[0]
        story_id = saved_story['id']
        
        # ==================== EXTRACT AND SAVE METADATA FOR FUTURE CONTINUATIONS ====================
        # Do this in background (don't block the response)
        try:
            metadata = await extract_story_metadata(full_text, story_data["title"])
            
            if metadata.get('summary') or metadata.get('characters'):
                supabase.table('stories').update({
                    'story_summary': metadata.get('summary', ''),
                    'characters': json.dumps(metadata.get('characters', [])),
                    'setting': metadata.get('setting', '')
                }).eq('id', story_id).execute()
                logger.info(f"[STORY] Saved metadata for future continuation: {len(metadata.get('characters', []))} characters")
        except Exception as meta_error:
            # Don't fail the request if metadata extraction fails
            logger.warning(f"[STORY] Metadata extraction failed (non-blocking): {str(meta_error)}")
        
        # Update user's sleep streak
        new_streak = await update_streak(user_id)
        logger.info(f"Story saved for user {user_id}, streak now: {new_streak}")
        
        return StoryResponse(
            storyId=saved_story['id'],
            title=story_data["title"],
            pages=story_data["pages"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Story generation error: {error_msg}")
        print(f"[STORY] ERROR: {error_msg}")
        
        if 'row-level security' in error_msg.lower() or '42501' in error_msg:
            raise HTTPException(
                status_code=500, 
                detail=f"Database permission error. Please contact support. (RLS: {error_msg[:200]})"
            )
        
        raise HTTPException(status_code=500, detail=f"Failed to generate story: {error_msg}")

@api_router.post("/tts", response_model=TTSResponse)
async def generate_tts(request: TTSRequest, user_id: str = Depends(get_current_user)):
    """Generate narration audio with plan-based limits (Premium: 90/month, Trial: 15 total, Free: 0)"""
    try:
        logger.info(f"TTS request for story {request.storyId} by user {user_id}")
        
        # STEP 1: Check if story exists and belongs to user
        story_result = supabase.table('stories').select('*').eq('id', request.storyId).eq('user_id', user_id).execute()
        
        if not story_result.data or len(story_result.data) == 0:
            logger.warning(f"Story {request.storyId} not found for user {user_id}")
            raise HTTPException(status_code=404, detail="Story not found")
        
        story = story_result.data[0]
        
        # STEP 2: If audio already exists, return cached URL (NO API CALL, NO USAGE INCREMENT)
        if story.get('audio_url'):
            logger.info(f"Returning cached audio for story {request.storyId}")
            return TTSResponse(audioUrl=story['audio_url'])
        
        # STEP 3: Get user plan and trial info
        user_result = supabase.table('users_profile').select('plan, trial_end, trial_narrations_used').eq('id', user_id).execute()
        
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_result.data[0]
        user_plan = user_data.get('plan', 'free')
        trial_end = user_data.get('trial_end')
        trial_narrations_used = user_data.get('trial_narrations_used', 0)
        
        # STEP 4: Determine effective plan (check if trial expired)
        effective_plan = user_plan
        if user_plan == 'trial' and trial_end:
            # Parse trial_end and check if expired
            try:
                from dateutil import parser
                trial_end_dt = parser.isoparse(trial_end) if isinstance(trial_end, str) else trial_end
                if datetime.utcnow() > trial_end_dt.replace(tzinfo=None):
                    effective_plan = 'free'
                    logger.info(f"User {user_id} trial expired, treating as free")
            except Exception as e:
                logger.error(f"Error parsing trial_end: {e}")
                effective_plan = 'free'
        
        logger.info(f"User {user_id} - plan: {user_plan}, effective_plan: {effective_plan}")
        
        # STEP 5: Check limits based on effective plan
        
        # FREE PLAN: No narration allowed
        if effective_plan == 'free':
            logger.warning(f"Free user {user_id} attempted narration")
            raise HTTPException(
                status_code=403,
                detail=json.dumps({
                    "error": "PAYWALL",
                    "reason": "narration_premium_only",
                    "message": "Narration is a Premium feature. Upgrade to generate audio.",
                    "plan": "free"
                })
            )
        
        # TRIAL PLAN: 15 total narrations during trial period
        elif effective_plan == 'trial':
            TRIAL_LIMIT = 15
            
            if trial_narrations_used >= TRIAL_LIMIT:
                logger.warning(f"Trial user {user_id} hit limit: {trial_narrations_used}/{TRIAL_LIMIT}")
                raise HTTPException(
                    status_code=429,
                    detail=json.dumps({
                        "error": "LIMIT_REACHED",
                        "used": trial_narrations_used,
                        "limit": TRIAL_LIMIT,
                        "plan": "trial",
                        "message": f"Trial narration limit reached ({TRIAL_LIMIT} narrations). Upgrade to Premium for unlimited narrations."
                    })
                )
            
            logger.info(f"Trial user {user_id} usage: {trial_narrations_used}/{TRIAL_LIMIT}")
        
        # PREMIUM PLAN: 90 narrations per month
        elif effective_plan == 'premium':
            # Get current month usage
            month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0).date()
            usage_result = supabase.table('narration_usage').select('*').eq('user_id', user_id).eq('month_start', month_start.isoformat()).execute()
            
            current_usage = 0
            if usage_result.data and len(usage_result.data) > 0:
                current_usage = usage_result.data[0].get('narrations_used', 0)
            
            MONTHLY_LIMIT = 90
            
            if current_usage >= MONTHLY_LIMIT:
                logger.warning(f"Premium user {user_id} hit monthly limit: {current_usage}/{MONTHLY_LIMIT}")
                raise HTTPException(
                    status_code=429,
                    detail=json.dumps({
                        "error": "LIMIT_REACHED",
                        "used": current_usage,
                        "limit": MONTHLY_LIMIT,
                        "plan": "premium",
                        "message": f"Monthly narration limit reached ({MONTHLY_LIMIT} narrations). Limit resets next month."
                    })
                )
            
            logger.info(f"Premium user {user_id} monthly usage: {current_usage}/{MONTHLY_LIMIT}")
        
        # STEP 6: Select appropriate voice for narration language
        narration_lang = request.language  # Use the language from TTS request
        
        # Get voice for the narration language
        if narration_lang in ELEVENLABS_VOICES:
            selected_voice = ELEVENLABS_VOICES[narration_lang]
            logger.info(f"Using language-specific voice: {selected_voice['name']} for {narration_lang}")
        else:
            # Fallback to multilingual voice
            selected_voice = MULTILINGUAL_VOICE
            logger.info(f"Using multilingual fallback voice: {selected_voice['name']} for {narration_lang}")
        
        # STEP 7: Generate audio (mocked - in production call ElevenLabs)
        # TODO: Replace with actual ElevenLabs API call using selected_voice
        # audio_url = await call_elevenlabs_api(request.text, selected_voice['voice_id'], narration_lang)
        mock_audio_url = f"https://cdn.pillowtales.co/narrations/{request.storyId}.mp3"
        logger.info(f"Generated audio URL: {mock_audio_url} with voice {selected_voice['name']}")
        
        # STEP 8: Save audio URL to story
        supabase.table('stories').update({"audio_url": mock_audio_url}).eq('id', request.storyId).execute()
        logger.info(f"Saved audio_url to story {request.storyId}")
        
        # STEP 8: Increment usage counter based on plan
        if effective_plan == 'trial':
            # Increment trial narrations counter
            new_trial_count = trial_narrations_used + 1
            supabase.table('users_profile').update({
                "trial_narrations_used": new_trial_count
            }).eq('id', user_id).execute()
            logger.info(f"Trial user {user_id} incremented: {new_trial_count}/15")
            
        elif effective_plan == 'premium':
            # Increment monthly narrations counter
            month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0).date()
            
            if usage_result.data and len(usage_result.data) > 0:
                # Update existing record
                new_usage = current_usage + 1
                supabase.table('narration_usage').update({
                    "narrations_used": new_usage,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq('user_id', user_id).eq('month_start', month_start.isoformat()).execute()
                logger.info(f"Premium user {user_id} incremented: {new_usage}/90")
            else:
                # Create new record
                supabase.table('narration_usage').insert({
                    "user_id": user_id,
                    "month_start": month_start.isoformat(),
                    "narrations_used": 1,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }).execute()
                logger.info(f"Premium user {user_id} new month: 1/90")
        
        logger.info(f"TTS generation complete for story {request.storyId}")
        return TTSResponse(audioUrl=mock_audio_url)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

@api_router.get("/stories")
async def get_user_stories(user_id: str = Depends(get_current_user)):
    """Get all stories for a user"""
    try:
        result = supabase.table('stories').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        
        return {"stories": result.data}
        
    except Exception as e:
        logger.error(f"Error fetching stories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stories: {str(e)}")

@api_router.get("/stories/continuable/latest")
async def get_latest_continuable_story(user_id: str = Depends(get_current_user)):
    """
    Get the most recent story that can be continued.
    Returns the story if it has less than 5 parts in its arc.
    """
    try:
        # Get the most recent story for this user
        result = supabase.table('stories').select(
            'id, title, child_name, theme, story_summary, characters, setting, story_arc_id, part_number, created_at'
        ).eq('user_id', user_id).order('created_at', desc=True).limit(1).execute()
        
        if not result.data or len(result.data) == 0:
            return {"canContinue": False, "story": None, "reason": "no_stories"}
        
        story = result.data[0]
        part_number = story.get('part_number', 1) or 1
        
        # Check if the story arc is complete (5 parts max)
        if part_number >= MAX_STORY_ARC_PARTS:
            return {
                "canContinue": False,
                "story": story,
                "reason": "arc_complete",
                "message": f"{story.get('child_name', 'Your child')}'s adventure is complete! Start a new story tonight."
            }
        
        # Story can be continued
        return {
            "canContinue": True,
            "story": story,
            "partNumber": part_number,
            "nextPartNumber": part_number + 1,
            "message": f"Continue {story.get('child_name', 'the')}'s Adventure? Part {part_number + 1}",
            "summary": story.get('story_summary', ''),
        }
        
    except Exception as e:
        logger.error(f"Error fetching continuable story: {str(e)}")
        return {"canContinue": False, "story": None, "reason": "error", "error": str(e)}

@api_router.get("/stories/{story_id}")
async def get_story(story_id: str, user_id: str = Depends(get_current_user)):
    """Get a specific story"""
    try:
        # Try with regular client first
        result = supabase.table('stories').select('*').eq('id', story_id).eq('user_id', user_id).execute()
        
        if not result.data or len(result.data) == 0:
            # Retry with fresh client (service role) in case of RLS issues
            try:
                from supabase import create_client
                fresh_supabase = create_client(supabase_url, supabase_key)
                result = fresh_supabase.table('stories').select('*').eq('id', story_id).eq('user_id', user_id).execute()
                if result.data and len(result.data) > 0:
                    logger.info(f"[GET_STORY] Found story {story_id} with fresh client")
                    return result.data[0]
            except Exception as retry_err:
                logger.warning(f"[GET_STORY] Retry with fresh client failed: {retry_err}")
            
            raise HTTPException(status_code=404, detail="Story not found")
        
        return result.data[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching story: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch story: {str(e)}")

@api_router.put("/stories/{story_id}")
async def update_story(story_id: str, request: UpdateStoryRequest, user_id: str = Depends(get_current_user)):
    """Update a story (e.g., toggle favorite)"""
    try:
        update_data = {}
        if request.isFavorite is not None:
            update_data["is_favorite"] = request.isFavorite
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")
        
        result = supabase.table('stories').update(update_data).eq('id', story_id).eq('user_id', user_id).execute()
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="Story not found")
        
        return {"message": "Story updated successfully", "story": result.data[0]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating story: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update story: {str(e)}")

@api_router.delete("/stories/{story_id}")
async def delete_story(story_id: str, user_id: str = Depends(get_current_user)):
    """Delete a story"""
    try:
        # Verify story belongs to user
        check_result = supabase.table('stories').select('id').eq('id', story_id).eq('user_id', user_id).execute()
        
        if not check_result.data or len(check_result.data) == 0:
            raise HTTPException(status_code=404, detail="Story not found")
        
        # Delete the story
        supabase.table('stories').delete().eq('id', story_id).eq('user_id', user_id).execute()
        
        logger.info(f"Deleted story {story_id} for user {user_id}")
        return {"message": "Story deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting story: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete story: {str(e)}")

# ================== PUBLIC STORY PREVIEW (NO AUTH REQUIRED) ==================
# Used for shared story links - allows anyone to see a preview without logging in
@api_router.get("/story-preview/{story_id}")
async def get_story_preview(story_id: str):
    """
    Get a public preview of a story for sharing purposes.
    
    This is a PUBLIC endpoint - no authentication required.
    Returns limited data: title, child name, first paragraph only.
    Used by the /story/{id} web page for shared story links.
    """
    try:
        # Fetch only the necessary fields for preview (no sensitive data)
        result = supabase.table('stories').select(
            'id, title, child_name, pages, language, duration_min, created_at'
        ).eq('id', story_id).execute()
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="Story not found")
        
        story = result.data[0]
        pages = story.get('pages', [])
        
        # Get just the first paragraph (strip any narration markers)
        first_paragraph = ""
        if pages and len(pages) > 0:
            # Clean the first page text - remove narration markers
            first_page = pages[0]
            first_paragraph = first_page.replace('[whisper]', '').replace('[softly]', '').replace('[chuckle]', '').replace('[pause]', '').replace('[gently]', '').strip()
            # Limit to first 500 chars for preview
            if len(first_paragraph) > 500:
                first_paragraph = first_paragraph[:497] + "..."
        
        # Calculate estimated duration
        page_count = len(pages)
        duration_min = story.get('duration_min', 8)
        duration_str = f"~{duration_min} min"
        
        logger.info(f"[STORY-PREVIEW] Served preview for story {story_id}: {story.get('title', 'Unknown')}")
        
        return {
            "id": story_id,
            "title": story.get('title', 'A Bedtime Story'),
            "childName": story.get('child_name', 'Little One'),
            "firstParagraph": first_paragraph,
            "pageCount": page_count,
            "duration": duration_str,
            "language": story.get('language', 'en'),
            "createdAt": story.get('created_at', ''),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STORY-PREVIEW] Error fetching story preview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch story preview")


@api_router.get("/user/settings")
async def get_user_settings(user_id: str = Depends(get_current_user)):
    """Get user settings"""
    try:
        result = supabase.table('users_profile').select('preferred_language, bedtime_mode, plan').eq('id', user_id).execute()
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return result.data[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch settings: {str(e)}")

@api_router.put("/user/settings")
async def update_user_settings(
    preferred_language: Optional[str] = None,
    bedtime_mode: Optional[bool] = None,
    user_id: str = Depends(get_current_user)
):
    """Update user settings"""
    try:
        update_data = {}
        if preferred_language is not None:
            update_data["preferred_language"] = preferred_language
        if bedtime_mode is not None:
            update_data["bedtime_mode"] = bedtime_mode
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")
        
        result = supabase.table('users_profile').update(update_data).eq('id', user_id).execute()
        
        return {"message": "Settings updated successfully", "settings": result.data[0]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

# ================== Voice Management Endpoints ==================

CONSENT_TEXT_VERSION = "v1.0"

@api_router.post("/voice/consent")
async def grant_voice_consent(request: VoiceConsentRequest, user_id: str = Depends(get_current_user)):
    """Grant or revoke consent for voice recording and cloning"""
    try:
        if request.consent:
            # Grant consent
            update_data = {
                "parent_voice_consent_at": datetime.utcnow().isoformat(),
                "consent_text_version": request.consentTextVersion
            }
        else:
            # Revoke consent - also delete voice model and recordings
            update_data = {
                "parent_voice_consent_at": None,
                "consent_text_version": None,
                "parent_voice_id": None
            }
            # Delete all voice recordings
            supabase.table('voice_recordings').delete().eq('user_id', user_id).execute()
        
        result = supabase.table('users_profile').update(update_data).eq('id', user_id).execute()
        
        return {
            "message": "Consent updated successfully",
            "consent": request.consent,
            "consent_at": update_data.get("parent_voice_consent_at")
        }
        
    except Exception as e:
        logger.error(f"Error updating voice consent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update consent: {str(e)}")

@api_router.get("/voice/status")
async def get_voice_status(user_id: str = Depends(get_current_user)):
    """Get user's voice consent and model status"""
    try:
        user_result = supabase.table('users_profile').select(
            'parent_voice_id, parent_voice_consent_at, consent_text_version'
        ).eq('id', user_id).execute()
        
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_result.data[0]
        
        # Get recording count
        recordings_result = supabase.table('voice_recordings').select('id', count='exact').eq('user_id', user_id).execute()
        recording_count = recordings_result.count if hasattr(recordings_result, 'count') else len(recordings_result.data)
        
        return {
            "hasConsent": user_data.get('parent_voice_consent_at') is not None,
            "consentAt": user_data.get('parent_voice_consent_at'),
            "consentVersion": user_data.get('consent_text_version'),
            "hasVoiceModel": user_data.get('parent_voice_id') is not None,
            "voiceModelId": user_data.get('parent_voice_id'),
            "recordingCount": recording_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting voice status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get voice status: {str(e)}")

@api_router.post("/voice/upload")
async def upload_voice_recording(request: VoiceRecordingUpload, user_id: str = Depends(get_current_user)):
    """Upload a voice recording sample (requires consent, max 20 samples)"""
    try:
        # Check if user has granted consent and is premium
        user_result = supabase.table('users_profile').select('parent_voice_consent_at, plan').eq('id', user_id).execute()
        
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_result.data[0]
        
        if not user_data.get('parent_voice_consent_at'):
            raise HTTPException(status_code=403, detail="Voice recording consent required")
        
        if user_data.get('plan') != 'premium':
            raise HTTPException(status_code=403, detail="Parent Voice is a Premium feature")
        
        # Check recording limit (max 20 samples)
        existing_result = supabase.table('voice_recordings').select('id', count='exact').eq('user_id', user_id).execute()
        existing_count = existing_result.count if hasattr(existing_result, 'count') else len(existing_result.data)
        
        if existing_count >= 20:
            raise HTTPException(status_code=400, detail="Maximum 20 voice samples allowed. Delete some to add more.")
        
        # Get next sample number
        sample_number = existing_count + 1
        sample_filename = f"sample_{sample_number:02d}.opus"
        storage_path = f"{user_id}/{sample_filename}"
        
        # Decode base64 audio data
        import base64
        try:
            audio_bytes = base64.b64decode(request.recordingData)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid audio data format")
        
        # Upload to Supabase Storage (parent-voices bucket)
        try:
            storage_result = supabase.storage.from_('parent-voices').upload(
                storage_path,
                audio_bytes,
                {"content-type": "audio/opus"}
            )
            
            # Get public URL
            recording_url = supabase.storage.from_('parent-voices').get_public_url(storage_path)
            logger.info(f"Uploaded voice sample to: {storage_path}")
        except Exception as storage_error:
            logger.error(f"Storage upload failed: {str(storage_error)}")
            # Fallback to placeholder if storage fails
            recording_url = f"parent-voices/{storage_path}"
        
        # Save recording metadata
        recording_data = {
            "user_id": user_id,
            "recording_url": recording_url,
            "storage_path": storage_path,
            "duration_seconds": request.durationSeconds,
            "sample_text": request.sampleText,
            "sample_number": sample_number,
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table('voice_recordings').insert(recording_data).execute()
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to save recording")
        
        return {
            "recordingId": result.data[0]['id'],
            "sampleNumber": sample_number,
            "totalSamples": sample_number,
            "maxSamples": 20,
            "message": "Recording uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading voice recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload recording: {str(e)}")

@api_router.post("/voice/create-model")
async def create_voice_model(request: CreateVoiceModelRequest, user_id: str = Depends(get_current_user)):
    """Create a voice model from uploaded recordings (Premium feature)"""
    try:
        # Check if user has granted consent
        user_result = supabase.table('users_profile').select('parent_voice_consent_at, plan').eq('id', user_id).execute()
        
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_result.data[0]
        
        if not user_data.get('parent_voice_consent_at'):
            raise HTTPException(status_code=403, detail="Voice recording consent required")
        
        # Check if user is premium (in production, verify with RevenueCat)
        if user_data.get('plan') != 'premium':
            raise HTTPException(
                status_code=403,
                detail=json.dumps({"error": "PAYWALL", "reason": "premium_feature"})
            )
        
        # Get user's recordings
        recordings_result = supabase.table('voice_recordings').select('*').eq('user_id', user_id).execute()
        
        if not recordings_result.data or len(recordings_result.data) < 3:
            raise HTTPException(status_code=400, detail="At least 3 voice recordings required")
        
        # In production, call ElevenLabs API to create voice model
        # For now, create a placeholder voice model ID
        voice_model_id = f"voice_model_{user_id}_{datetime.utcnow().timestamp()}"
        
        # Update user with voice model ID
        update_result = supabase.table('users_profile').update({
            "parent_voice_id": voice_model_id
        }).eq('id', user_id).execute()
        
        logger.info(f"Voice model created for user {user_id}: {voice_model_id}")
        
        return {
            "voiceModelId": voice_model_id,
            "message": "Voice model created successfully",
            "note": "In production, this would call ElevenLabs API"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating voice model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create voice model: {str(e)}")

@api_router.delete("/voice/model")
async def delete_voice_model(user_id: str = Depends(get_current_user)):
    """Delete user's voice model (GDPR right to be forgotten)"""
    try:
        # In production, call ElevenLabs API to delete voice model
        # Get current voice model ID
        user_result = supabase.table('users_profile').select('parent_voice_id').eq('id', user_id).execute()
        
        if user_result.data and len(user_result.data) > 0:
            voice_model_id = user_result.data[0].get('parent_voice_id')
            if voice_model_id:
                logger.info(f"Deleting voice model {voice_model_id} for user {user_id}")
                # TODO: Call ElevenLabs API to delete voice model
        
        # Remove voice model ID from user
        supabase.table('users_profile').update({"parent_voice_id": None}).eq('id', user_id).execute()
        
        return {"message": "Voice model deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting voice model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete voice model: {str(e)}")

@api_router.delete("/voice/recordings")
async def delete_all_recordings(user_id: str = Depends(get_current_user)):
    """Delete all voice recordings (GDPR right to be forgotten)"""
    try:
        # Get all recordings with storage paths
        recordings_result = supabase.table('voice_recordings').select('id, recording_url, storage_path').eq('user_id', user_id).execute()
        
        if recordings_result.data:
            # Delete files from Supabase Storage
            for recording in recordings_result.data:
                storage_path = recording.get('storage_path')
                if storage_path:
                    try:
                        supabase.storage.from_('parent-voices').remove([storage_path])
                        logger.info(f"Deleted recording from storage: {storage_path}")
                    except Exception as storage_error:
                        logger.warning(f"Failed to delete from storage: {storage_path} - {str(storage_error)}")
        
        # Delete recording records from database
        supabase.table('voice_recordings').delete().eq('user_id', user_id).execute()
        
        # Also delete voice model if exists
        supabase.table('users_profile').update({
            "parent_voice_id": None,
            "parent_voice_enabled": False,
            "parent_voice_url": None
        }).eq('id', user_id).execute()
        
        return {"message": "All voice recordings deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting voice recordings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete recordings: {str(e)}")

@api_router.delete("/voice/recordings/{recording_id}")
async def delete_single_recording(recording_id: str, user_id: str = Depends(get_current_user)):
    """Delete a single voice recording"""
    try:
        # Get the recording
        recording_result = supabase.table('voice_recordings').select('id, storage_path').eq('id', recording_id).eq('user_id', user_id).execute()
        
        if not recording_result.data or len(recording_result.data) == 0:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        recording = recording_result.data[0]
        
        # Delete from storage
        storage_path = recording.get('storage_path')
        if storage_path:
            try:
                supabase.storage.from_('parent-voices').remove([storage_path])
                logger.info(f"Deleted recording from storage: {storage_path}")
            except Exception as storage_error:
                logger.warning(f"Failed to delete from storage: {storage_path} - {str(storage_error)}")
        
        # Delete from database
        supabase.table('voice_recordings').delete().eq('id', recording_id).execute()
        
        return {"message": "Recording deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete recording: {str(e)}")

@api_router.delete("/user/account")
async def delete_user_account(user_id: str = Depends(get_current_user)):
    """Delete user account and all associated data (GDPR compliance)"""
    try:
        logger.info(f"Starting account deletion for user {user_id}")
        
        # 1. Delete voice recordings from storage
        recordings_result = supabase.table('voice_recordings').select('storage_path').eq('user_id', user_id).execute()
        if recordings_result.data:
            storage_paths = [r['storage_path'] for r in recordings_result.data if r.get('storage_path')]
            if storage_paths:
                try:
                    supabase.storage.from_('parent-voices').remove(storage_paths)
                    logger.info(f"Deleted {len(storage_paths)} voice recordings from storage")
                except Exception as e:
                    logger.warning(f"Failed to delete voice recordings from storage: {str(e)}")
        
        # 2. Delete voice recordings table entries
        supabase.table('voice_recordings').delete().eq('user_id', user_id).execute()
        
        # 3. Delete stories
        supabase.table('stories').delete().eq('user_id', user_id).execute()
        
        # 4. Delete narration usage
        supabase.table('narration_usage').delete().eq('user_id', user_id).execute()
        
        # 5. Delete user profile
        supabase.table('users_profile').delete().eq('id', user_id).execute()
        
        # 6. Delete auth user (requires service role key)
        try:
            supabase.auth.admin.delete_user(user_id)
            logger.info(f"Deleted auth user {user_id}")
        except Exception as auth_error:
            logger.warning(f"Failed to delete auth user: {str(auth_error)}")
        
        logger.info(f"Account deletion completed for user {user_id}")
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting account: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete account: {str(e)}")

@api_router.get("/voice/recordings")
async def get_user_recordings(user_id: str = Depends(get_current_user)):
    """Get all user's voice recordings"""
    try:
        result = supabase.table('voice_recordings').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        
        return {"recordings": result.data}
        
    except Exception as e:
        logger.error(f"Error fetching recordings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch recordings: {str(e)}")

@api_router.get("/narration/usage")
async def get_narration_usage(user_id: str = Depends(get_current_user)):
    """Get current narration usage based on plan (Premium: monthly, Trial: total, Free: 0)"""
    try:
        # Get user plan and trial info
        user_result = supabase.table('users_profile').select('plan, trial_end, trial_narrations_used').eq('id', user_id).execute()
        
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_result.data[0]
        user_plan = user_data.get('plan', 'free')
        trial_end = user_data.get('trial_end')
        trial_narrations_used = user_data.get('trial_narrations_used', 0)
        
        # Determine effective plan (check if trial expired)
        effective_plan = user_plan
        if user_plan == 'trial' and trial_end:
            try:
                from dateutil import parser
                trial_end_dt = parser.isoparse(trial_end) if isinstance(trial_end, str) else trial_end
                if datetime.utcnow() > trial_end_dt.replace(tzinfo=None):
                    effective_plan = 'free'
            except:
                effective_plan = 'free'
        
        # FREE PLAN
        if effective_plan == 'free':
            return {
                "plan": "free",
                "isPremium": False,
                "used": 0,
                "limit": 0,
                "remaining": 0,
                "percentUsed": 0,
                "message": "Narration is a Premium feature"
            }
        
        # TRIAL PLAN
        elif effective_plan == 'trial':
            TRIAL_LIMIT = 15
            remaining = max(0, TRIAL_LIMIT - trial_narrations_used)
            percent_used = int((trial_narrations_used / TRIAL_LIMIT) * 100)
            
            return {
                "plan": "trial",
                "isPremium": False,
                "isTrial": True,
                "used": trial_narrations_used,
                "limit": TRIAL_LIMIT,
                "remaining": remaining,
                "percentUsed": percent_used,
                "trialEnd": trial_end
            }
        
        # PREMIUM PLAN
        elif effective_plan == 'premium':
            # Get current month usage
            month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0).date()
            usage_result = supabase.table('narration_usage').select('*').eq('user_id', user_id).eq('month_start', month_start.isoformat()).execute()
            
            current_usage = 0
            if usage_result.data and len(usage_result.data) > 0:
                current_usage = usage_result.data[0].get('narrations_used', 0)
            
            MONTHLY_LIMIT = 90
            remaining = max(0, MONTHLY_LIMIT - current_usage)
            percent_used = int((current_usage / MONTHLY_LIMIT) * 100)
            
            return {
                "plan": "premium",
                "isPremium": True,
                "used": current_usage,
                "limit": MONTHLY_LIMIT,
                "remaining": remaining,
                "percentUsed": percent_used,
                "monthStart": month_start.isoformat()
            }
        
        # Default (shouldn't reach here)
        return {
            "plan": "unknown",
            "isPremium": False,
            "used": 0,
            "limit": 0,
            "remaining": 0,
            "percentUsed": 0
        }
        
    except Exception as e:
        logger.error(f"Error fetching narration usage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch usage: {str(e)}")

# ================== Optimized Narration (On-Demand TTS with Caching) ==================

# Cost metrics logging
def log_tts_metrics(story_id: str, provider: str, chars: int, language: str, voice: str, success: bool, error: str = None):
    """Log TTS metrics for cost tracking and analytics"""
    cost_per_1k = TTS_COST_PER_1K.get(provider, 0.015)
    estimated_cost = (chars / 1000) * cost_per_1k
    
    logger.info("=" * 60)
    logger.info("[TTS METRICS]")
    logger.info(f"  story_id: {story_id}")
    logger.info(f"  provider: {provider}")
    logger.info(f"  tts_chars: {chars}")
    logger.info(f"  language: {language}")
    logger.info(f"  voice: {voice}")
    logger.info(f"  cost_per_1k: ${cost_per_1k:.4f}")
    logger.info(f"  estimated_cost: ${estimated_cost:.4f}")
    logger.info(f"  success: {success}")
    if error:
        logger.info(f"  error: {error}")
    logger.info("=" * 60)
    
    return estimated_cost

# ================== AUDIO COMPRESSION HELPER ==================
# Supabase Storage has size limits. For long stories (10-15 min), audio files
# can exceed the limit. This compresses audio to a smaller file size.

def compress_audio_for_upload(audio_bytes: bytes, target_bitrate: str = "48k") -> bytes:
    """
    Compress audio to smaller file size for faster uploads.
    Uses pydub for audio processing.

    NOTE: Python 3.13+ removed the audioop module that pydub depends on.
    For Python 3.13+, install: pip install audioop-lts

    Args:
        audio_bytes: Raw audio data
        target_bitrate: Target MP3 bitrate (e.g., "48k", "64k", "32k")

    Returns:
        Compressed MP3 audio bytes (or original if compression fails)
    """
    if not audio_bytes:
        logger.warning("[AUDIO] No audio bytes provided for compression")
        return audio_bytes

    import io

    original_size = len(audio_bytes)
    logger.info(f"[AUDIO] Compressing: {original_size/1024/1024:.1f}MB → target {target_bitrate}")

    try:
        # Try to import pydub - may fail on Python 3.13+ without audioop-lts
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        output_buffer = io.BytesIO()
        audio.export(output_buffer, format="mp3", bitrate=target_bitrate)
        compressed_bytes = output_buffer.getvalue()

        compressed_size = len(compressed_bytes)
        reduction = (1 - compressed_size / original_size) * 100 if original_size else 0

        logger.info(
            f"[AUDIO] Compressed: {original_size/1024/1024:.1f}MB → "
            f"{compressed_size/1024/1024:.1f}MB ({reduction:.0f}% reduction)"
        )
        return compressed_bytes

    except ImportError as e:
        logger.warning(f"[AUDIO] pydub/audioop not available: {str(e)}")
        logger.warning("[AUDIO] For Python 3.13+, add 'audioop-lts' to requirements.txt")
        logger.info("[AUDIO] Skipping compression, using original audio")
        return audio_bytes

    except Exception as e:
        logger.error(f"[AUDIO] Compression failed: {str(e)} - using original")
        return audio_bytes

async def generate_tts_audio_openai(
    text: str,
    language_code: str,
    voice: str = "nova",
    story_id: str = None,
    voice_preset: str = None
) -> tuple:
    """
    Generate TTS audio using OpenAI TTS API.
    This is the fallback/default path when ElevenLabs is unavailable or disabled.

    Returns: (audio_bytes, voice_used, char_count, estimated_cost)
    """
    import httpx

    char_count = len(text)
    lang_code = language_code.lower()[:2] if language_code else "en"
    openai_key = os.getenv("OPENAI_API_KEY")

    logger.info("=" * 50)
    logger.info("[TTS-OPENAI] GENERATION REQUEST")
    logger.info(f"[TTS-OPENAI]   story_id: {story_id}")
    logger.info(f"[TTS-OPENAI]   chars: {char_count}")
    logger.info(f"[TTS-OPENAI]   language: {lang_code}")
    logger.info(f"[TTS-OPENAI]   voice: {voice}")
    logger.info("=" * 50)

    if not openai_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    # Keep voice simple/stable for bedtime
    selected_voice = voice or "nova"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini-tts",
                    "voice": selected_voice,
                    "input": text,
                    "format": "mp3",
                },
            )

        if response.status_code != 200:
            error_text = response.text[:500]
            log_tts_metrics(
                story_id=story_id,
                provider="openai",
                chars=char_count,
                language=lang_code,
                voice=selected_voice,
                success=False,
                error=error_text,
            )
            raise HTTPException(status_code=500, detail=f"OpenAI TTS failed: {error_text}")

        audio_bytes = response.content
        estimated_cost = log_tts_metrics(
            story_id=story_id,
            provider="openai",
            chars=char_count,
            language=lang_code,
            voice=selected_voice,
            success=True,
        )

        logger.info(f"[TTS-OPENAI] SUCCESS: Generated {len(audio_bytes)} bytes, cost ~${estimated_cost:.4f}")
        return (audio_bytes, selected_voice, char_count, estimated_cost)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TTS-OPENAI] Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI TTS exception: {str(e)}")
 
async def generate_tts_elevenlabs_expressive(text: str, language_code: str, voice_preset: str = None, story_id: str = None, voice_id_override: str = None) -> tuple:
    """
    Generate warm, expressive bedtime narration using ElevenLabs.
    
    Key features:
    - SENTENCE-BY-SENTENCE generation for natural pacing
    - Comma pauses for breathing room within sentences
    - Expressive markers ([whisper], [softly], [chuckle]) with proper handling
    - Paragraph pauses between story pages
    - Warm, calm voice settings optimized for bedtime
    - LANGUAGE-CONSISTENT PACING: All languages get same calm bedtime speed
    
    Args:
        voice_id_override: For parent voice, use their cloned voice ID directly
    
    Returns: (audio_bytes, voice_id, char_count, estimated_cost)
    """
    import httpx
    import re
    import io

    # Try to import pydub - may fail on Python 3.13+ without audioop-lts
    try:
        from pydub import AudioSegment
        PYDUB_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"[TTS-ELEVEN] pydub not available: {e}")
        logger.warning("[TTS-ELEVEN] For Python 3.13+, add 'audioop-lts' to requirements.txt")
        PYDUB_AVAILABLE = False

    # If pydub is not available, fall back to OpenAI TTS
    if not PYDUB_AVAILABLE:
        logger.warning("[TTS-ELEVEN] pydub not available - falling back to OpenAI TTS")
        return await generate_tts_audio_openai(text, language_code, "shimmer", story_id, voice_preset)

    lang_code = language_code.lower()[:2] if language_code else 'en'

    if not USE_ELEVENLABS:
        logger.info("[TTS-ELEVEN] ElevenLabs disabled by USE_ELEVENLABS=false, using OpenAI fallback")
        return await generate_tts_audio_openai(text, language_code, "nova", story_id, voice_preset)

    # Get narrator personality settings
    preset = VOICE_PRESETS.get(voice_preset, VOICE_PRESETS[DEFAULT_NARRATOR])
    personality = preset.get("personality", "friendly")
    voice_id = preset.get("voice_id")
    
    # Log the voice selection for debugging
    logger.info(f"[TTS-ELEVEN] Voice selection: preset_requested='{voice_preset}', preset_name='{preset.get('name')}', voice_id='{voice_id}'")
    
    # If voice_id_override is provided (e.g., for parent voice), use it
    if voice_id_override:
        voice_id = voice_id_override
        logger.info(f"[TTS-ELEVEN] Using voice_id_override (Parent Voice): {voice_id}")
    
    # ==================== PARENT VOICE: USE NEUTRAL SETTINGS ====================
    # Parent Voice should use neutral/natural settings because:
    # 1. The user's voice sample already has their natural pacing
    # 2. Slowing it down artificially makes it sound unnatural
    # 3. The cloned voice already captures their tone/style
    is_parent_voice = voice_preset == "parent_voice" or voice_id_override is not None
    
    if is_parent_voice:
        logger.info("[TTS-ELEVEN] PARENT VOICE DETECTED - Using neutral settings")
        # Use neutral settings that let the parent's natural voice shine through
        PARENT_VOICE_SETTINGS = {
            'stability': 0.75,           # Slightly lower = more natural expression
            'similarity_boost': 0.50,    # Higher = clearer, closer to original voice
            'style': 0.0,                # No style modification
            'speed': 1.0,                # Normal speed - don't slow down!
            'pause_mult': 1.0,           # Normal pauses
            'notes': 'Parent Voice - natural settings'
        }
        # Force use these settings instead of language-specific ones
        base_stability = PARENT_VOICE_SETTINGS['stability']
        similarity_boost = PARENT_VOICE_SETTINGS['similarity_boost']
        base_style = PARENT_VOICE_SETTINGS['style']
        speed_adjustment = PARENT_VOICE_SETTINGS['speed']
        pause_multiplier = PARENT_VOICE_SETTINGS['pause_mult']
        
        logger.info(f"[TTS-ELEVEN]   stability: {base_stability}")
        logger.info(f"[TTS-ELEVEN]   similarity_boost: {similarity_boost}")
        logger.info(f"[TTS-ELEVEN]   speed: {speed_adjustment} (normal)")
        logger.info(f"[TTS-ELEVEN]   pause_mult: {pause_multiplier}")
    else:
        # Fall through to language-specific settings below (will be applied after)
        pass
    
    # ==================== BEDTIME NARRATION SETTINGS ====================
    # These settings ensure calm, consistent pacing across ALL languages
    # Spanish, French, German etc. should feel the same as English
    
    # ==================== LANGUAGE-SPECIFIC VOICE TUNING ====================
    # Each language needs specific adjustments to sound equally calm and smooth
    LANGUAGE_VOICE_SETTINGS = {
        'en': {
            # English - baseline, already sounds good
            'stability': 0.80,
            'similarity_boost': 0.35,
            'style': 0.0,
            'speed': 1.0,
            'pause_mult': 1.0,
            'notes': 'Baseline - warm and calm'
        },
        'es': {
            # Spanish - needs smoother, less robotic delivery
            # Increase stability for calmness, reduce similarity for naturalness
            'stability': 0.85,          # Higher for smoother, calmer delivery
            'similarity_boost': 0.30,   # Lower for more natural flow
            'style': 0.05,              # Tiny bit of warmth
            'speed': 0.85,              # Slower for bedtime feel
            'pause_mult': 1.3,          # Longer pauses for relaxed pacing
            'notes': 'Smoother, softer, more bedtime-friendly'
        },
        'it': {
            # Italian - also robotic, needs more natural pacing
            # Similar treatment to Spanish
            'stability': 0.85,          # Higher for smoother delivery
            'similarity_boost': 0.32,   # Slightly higher for clarity
            'style': 0.05,              # Minimal expressiveness
            'speed': 0.86,              # Slower pace
            'pause_mult': 1.25,         # Longer pauses
            'notes': 'More natural, relaxed Italian delivery'
        },
        'de': {
            # German - fuzzy/unclear, needs clearer articulation
            # Boost similarity for clarity, keep calm
            'stability': 0.82,          # Good stability
            'similarity_boost': 0.45,   # Higher for clearer articulation
            'style': 0.0,               # No extra expressiveness
            'speed': 0.88,              # Slightly slower
            'pause_mult': 1.15,         # Moderate pauses
            'notes': 'Clearer articulation, calm German'
        },
        'fr': {
            # French - generally good, minor tuning
            'stability': 0.82,
            'similarity_boost': 0.38,
            'style': 0.05,
            'speed': 0.88,
            'pause_mult': 1.2,
            'notes': 'Smooth French bedtime delivery'
        },
        'pt': {
            # Portuguese - similar to Spanish
            'stability': 0.85,
            'similarity_boost': 0.32,
            'style': 0.05,
            'speed': 0.85,
            'pause_mult': 1.25,
            'notes': 'Smooth Portuguese bedtime delivery'
        },
        'nl': {
            # Dutch
            'stability': 0.82,
            'similarity_boost': 0.40,
            'style': 0.0,
            'speed': 0.90,
            'pause_mult': 1.15,
            'notes': 'Clear Dutch delivery'
        },
        'pl': {
            # Polish
            'stability': 0.83,
            'similarity_boost': 0.38,
            'style': 0.0,
            'speed': 0.88,
            'pause_mult': 1.2,
            'notes': 'Calm Polish delivery'
        },
        'ja': {
            # Japanese
            'stability': 0.80,
            'similarity_boost': 0.40,
            'style': 0.0,
            'speed': 0.92,
            'pause_mult': 1.1,
            'notes': 'Natural Japanese pacing'
        },
        'zh': {
            # Chinese
            'stability': 0.80,
            'similarity_boost': 0.42,
            'style': 0.0,
            'speed': 0.92,
            'pause_mult': 1.1,
            'notes': 'Clear Chinese delivery'
        },
        'ko': {
            # Korean
            'stability': 0.80,
            'similarity_boost': 0.40,
            'style': 0.0,
            'speed': 0.90,
            'pause_mult': 1.15,
            'notes': 'Natural Korean pacing'
        },
    }
    
    # Get language-specific settings (fallback to English-like defaults)
    lang_settings = LANGUAGE_VOICE_SETTINGS.get(lang_code, {
        'stability': 0.82,
        'similarity_boost': 0.38,
        'style': 0.0,
        'speed': 0.88,
        'pause_mult': 1.15,
    })
    
    # Apply language-specific overrides to base settings
    base_stability = lang_settings.get('stability', preset.get("stability", 0.80))
    similarity_boost = lang_settings.get('similarity_boost', preset.get("similarity_boost", 0.35))
    base_style = lang_settings.get('style', preset.get("style", 0.0))
    speed_adjustment = lang_settings.get('speed', 0.88)
    pause_multiplier = lang_settings.get('pause_mult', 1.15)
    
    # Keep legacy speed/pause arrays for backwards compatibility
    LANGUAGE_SPEED_ADJUSTMENTS = {
        # V1 SETTINGS: Normal speed for all languages
        # Narrators are hand-picked for calm bedtime quality, no need to slow down
        'en': 1.0,      # English
        'es': 1.0,      # Spanish
        'fr': 1.0,      # French 
        'de': 1.0,      # German
        'it': 1.0,      # Italian
        'pt': 1.0,      # Portuguese
        'pl': 1.0,      # Polish
        'nl': 1.0,      # Dutch
        'ja': 1.0,      # Japanese
        'zh': 1.0,      # Chinese
        'ko': 1.0,      # Korean
    }
    
    LANGUAGE_PAUSE_MULTIPLIERS = {
        # V1 SETTINGS: Normal pauses for all languages
        'en': 1.0,
        'es': 1.0,
        'fr': 1.0,
        'de': 1.0,
        'it': 1.0,
        'pt': 1.0,
        'pl': 1.0,
        'nl': 1.0,
        'ja': 1.0,
        'zh': 1.0,
        'ko': 1.0,
    }
    
    # Language-specific whisper keywords for detecting soft delivery triggers
    WHISPER_KEYWORDS = {
        'en': ['whispered', 'whisper', 'softly', 'gently', 'quietly', 'hushed'],
        'es': ['susurró', 'susurro', 'suavemente', 'gentilmente', 'silenciosamente', 'en voz baja'],
        'fr': ['chuchota', 'murmura', 'doucement', 'gentiment', 'silencieusement'],
        'de': ['flüsterte', 'flüstern', 'sanft', 'leise', 'still'],
        'it': ['sussurrò', 'sussurro', 'dolcemente', 'gentilmente', 'silenziosamente'],
        'pt': ['sussurrou', 'sussurro', 'suavemente', 'gentilmente', 'silenciosamente'],
    }
    
    speed_adjustment = LANGUAGE_SPEED_ADJUSTMENTS.get(lang_code, 0.90)
    pause_multiplier = LANGUAGE_PAUSE_MULTIPLIERS.get(lang_code, 1.15)
    whisper_keywords = WHISPER_KEYWORDS.get(lang_code, WHISPER_KEYWORDS['en'])
    
    # ==================== PARENT VOICE OVERRIDE ====================
    # Parent Voice should use NEUTRAL settings because:
    # 1. The user's voice sample already has their natural pacing
    # 2. Slowing it down artificially makes it sound unnatural
    # 3. The cloned voice already captures their tone/style
    if is_parent_voice:
        logger.info("[TTS-ELEVEN] PARENT VOICE - Overriding to neutral settings")
        base_stability = 0.75           # Slightly lower = more natural expression
        similarity_boost = 0.50         # Higher = clearer, closer to original voice
        base_style = 0.0                # No style modification
        speed_adjustment = 1.0          # Normal speed - DON'T slow down!
        pause_multiplier = 1.0          # Normal pauses
        logger.info(f"[TTS-ELEVEN]   Parent Voice overrides: speed=1.0, stability=0.75, similarity=0.50")
    
    # Get base pause durations and apply language multiplier
    base_pauses = ELEVENLABS_PAUSE_DURATIONS.get(personality, ELEVENLABS_PAUSE_DURATIONS["friendly"])
    pauses = {key: int(value * pause_multiplier) for key, value in base_pauses.items()}
    
    eleven_labs_key = os.environ.get('ELEVENLABS_API_KEY')
    if not eleven_labs_key or eleven_labs_key == 'placeholder-elevenlabs-key':
        logger.warning("[TTS-ELEVEN] API key not configured, falling back to OpenAI")
        return await generate_tts_audio_openai(text, language_code, "shimmer", story_id, voice_preset)
    
    if not voice_id:
        voice_id = "XB0fDUnXU5powFXDhCwa"  # Charlotte (default - warm, friendly)
    
    logger.info("=" * 60)
    logger.info("[TTS-ELEVEN] BEDTIME NARRATION - LANGUAGE-SPECIFIC TUNING")
    logger.info(f"[TTS-ELEVEN]   narrator: {preset.get('name', 'Unknown')}")
    logger.info(f"[TTS-ELEVEN]   language: {lang_code}")
    logger.info(f"[TTS-ELEVEN]   voice_id: {voice_id}")
    logger.info("[TTS-ELEVEN]   --- Voice Settings (language-tuned) ---")
    logger.info(f"[TTS-ELEVEN]   stability: {base_stability} (higher=calmer)")
    logger.info(f"[TTS-ELEVEN]   similarity_boost: {similarity_boost} (higher=clearer)")
    logger.info(f"[TTS-ELEVEN]   style: {base_style} (lower=smoother)")
    logger.info(f"[TTS-ELEVEN]   speed_adjustment: {speed_adjustment} (lower=slower)")
    logger.info(f"[TTS-ELEVEN]   pause_multiplier: {pause_multiplier} (higher=longer pauses)")
    logger.info(f"[TTS-ELEVEN]   pauses: sentence={pauses['sentence']}ms, comma={pauses['comma']}ms")
    logger.info(f"[TTS-ELEVEN]   notes: {lang_settings.get('notes', 'default settings')}")
    logger.info("=" * 60)
    
    # ==================== PARSE TEXT INTO SEGMENTS ====================
    def parse_story_text(text):
        """
        Parse story text into segments with markers and pause information.
        Handles: [whisper], [softly], [chuckle] markers
        Returns list of dicts: {text, marker, is_paragraph_start}
        """
        segments = []
        
        # First, split by story pages (double newlines indicate paragraph breaks)
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            is_paragraph_start = para_idx > 0  # First paragraph doesn't need extra pause
            
            # Find and extract markers: [whisper], [softly], [chuckle]
            # Markers affect the sentence that follows them
            current_marker = None
            
            # Split by markers, keeping the markers
            parts = re.split(r'(\[whisper\]|\[softly\]|\[chuckle\])', paragraph, flags=re.IGNORECASE)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                part_lower = part.lower()
                if part_lower == '[whisper]':
                    current_marker = 'whisper'
                    continue
                elif part_lower == '[softly]':
                    current_marker = 'softly'
                    continue
                elif part_lower == '[chuckle]':
                    current_marker = 'chuckle'
                    continue
                
                # Split this part into sentences
                sentences = re.split(r'(?<=[.!?])\s+', part)
                
                for sent_idx, sentence in enumerate(sentences):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    segments.append({
                        "text": sentence,
                        "marker": current_marker,
                        "is_paragraph_start": is_paragraph_start and sent_idx == 0,
                    })
                    
                    # Only apply marker to the first sentence after it
                    if current_marker != 'chuckle':  # chuckle affects current sentence
                        current_marker = None
        
        return segments
    
    segments = parse_story_text(text)
    
    if not segments:
        raise HTTPException(status_code=400, detail="No text to narrate")
    
    logger.info(f"[TTS-ELEVEN]   total segments: {len(segments)}")
    
    # ==================== GENERATE AUDIO FOR EACH SEGMENT ====================
    audio_segments = []
    total_chars = 0
    
    async def generate_sentence_audio(sentence_text: str, stability: float, style: float) -> AudioSegment:
        """Generate audio for a single sentence using ElevenLabs API
        
        Applies language-specific speed adjustment for consistent bedtime pacing.
        """
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": eleven_labs_key
        }
        
        # For non-English, use slightly higher stability for calmer delivery
        adjusted_stability = stability
        if lang_code != 'en':
            adjusted_stability = min(stability + 0.05, 0.95)  # Bump stability for calmness
        
        data = {
            "text": sentence_text,
            "model_id": "eleven_multilingual_v2",
            "language_code": lang_code,
            "voice_settings": {
                "stability": adjusted_stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": True
            },
            "output_format": "mp3_44100_128"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                audio = AudioSegment.from_mp3(io.BytesIO(response.content))
                
                # Apply speed adjustment for non-English languages
                # This ensures consistent bedtime pacing across all languages
                if speed_adjustment < 1.0 and lang_code != 'en':
                    # Slow down the audio using pydub
                    # Lower playback rate = slower, calmer narration
                    # We achieve this by changing the frame rate
                    original_frame_rate = audio.frame_rate
                    # Slow down by reducing effective playback speed
                    # speed_adjustment of 0.88 means play at 88% speed
                    slowed_frame_rate = int(original_frame_rate * speed_adjustment)
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": slowed_frame_rate
                    }).set_frame_rate(original_frame_rate)
                    
                return audio
            elif response.status_code == 401 and "quota_exceeded" in response.text.lower():
                # Quota exceeded - raise specific error to stop processing
                logger.error("[TTS-ELEVEN] QUOTA EXCEEDED - stopping generation")
                raise Exception("QUOTA_EXCEEDED: ElevenLabs credits exhausted")
            else:
                logger.error(f"[TTS-ELEVEN] API error {response.status_code}: {response.text[:200]}")
                return None
    
    # Helper function to check for whisper keywords in text (multilingual)
    def contains_whisper_keyword(text: str) -> bool:
        """Check if text contains whisper/soft delivery keywords in any supported language"""
        text_lower = text.lower()
        for keyword in whisper_keywords:
            if keyword.lower() in text_lower:
                return True
        return False
    
    for i, segment in enumerate(segments):
        sentence_text = segment["text"]
        marker = segment["marker"]
        is_para_start = segment["is_paragraph_start"]
        
        # Add paragraph pause if this is start of a new paragraph
        if is_para_start and audio_segments:
            para_silence = AudioSegment.silent(duration=pauses["paragraph"])
            audio_segments.append(para_silence)
            logger.info(f"[TTS-ELEVEN]   [PARAGRAPH BREAK] +{pauses['paragraph']}ms")
        
        # Determine voice settings and pre-pause based on marker
        sent_stability = base_stability
        sent_style = base_style
        pre_pause = 0
        post_pause = pauses["sentence"]
        
        # Check for whisper keywords in the sentence (multilingual support)
        # Words like "whispered", "susurró" (Spanish), "chuchota" (French) trigger soft delivery
        has_whisper_keyword = contains_whisper_keyword(sentence_text)
        
        if marker == "whisper" or has_whisper_keyword:
            # Whisper: higher stability for consistency, lower style for calmness
            sent_stability = min(base_stability + 0.15, 0.92)
            sent_style = max(base_style - 0.08, 0.05)
            pre_pause = pauses["whisper_before"]
            post_pause = pauses["whisper_after"]
            logger.info(f"[TTS-ELEVEN]   [{i+1}] [WHISPER] {sentence_text[:50]}...")
        elif marker == "softly":
            # Softly: slightly higher stability
            sent_stability = min(base_stability + 0.10, 0.88)
            sent_style = max(base_style - 0.05, 0.08)
            pre_pause = pauses["softly_before"]
            post_pause = pauses["softly_after"]
            logger.info(f"[TTS-ELEVEN]   [{i+1}] [SOFTLY] {sentence_text[:50]}...")
        elif marker == "chuckle":
            # Chuckle: add pause after for effect
            post_pause = pauses["chuckle"]
            logger.info(f"[TTS-ELEVEN]   [{i+1}] [chuckle] {sentence_text[:50]}...")
        else:
            logger.info(f"[TTS-ELEVEN]   [{i+1}] {sentence_text[:50]}...")
        
        # Add pre-pause for expressive moments
        if pre_pause > 0:
            pre_silence = AudioSegment.silent(duration=pre_pause)
            audio_segments.append(pre_silence)
        
        total_chars += len(sentence_text)
        
        try:
            # Generate audio for this sentence
            audio = await generate_sentence_audio(sentence_text, sent_stability, sent_style)
            
            if audio:
                audio_segments.append(audio)
                
                # Add comma pauses within the sentence by analyzing the audio duration
                # For sentences with commas, we add tiny gaps in post-processing
                # This is approximated by slightly longer post-sentence pause
                comma_count = sentence_text.count(',')
                if comma_count > 0:
                    extra_pause = min(comma_count * (pauses["comma"] // 2), 400)
                    post_pause += extra_pause
                
                # Add post-sentence pause
                if i < len(segments) - 1:  # Don't add pause after last sentence
                    post_silence = AudioSegment.silent(duration=post_pause)
                    audio_segments.append(post_silence)
        except Exception as e:
            error_str = str(e)
            # Check for quota exceeded - fallback to OpenAI gracefully
            if "QUOTA_EXCEEDED" in error_str or "quota" in error_str.lower() or "credit" in error_str.lower():
                logger.warning("[TTS-ELEVEN] QUOTA/CREDITS EXHAUSTED - falling back to OpenAI TTS")
                return await generate_tts_audio_openai(text, language_code, "shimmer", story_id, voice_preset)
            # Check for rate limiting
            if "rate" in error_str.lower() or "429" in error_str:
                logger.warning("[TTS-ELEVEN] Rate limited - falling back to OpenAI TTS")
                return await generate_tts_audio_openai(text, language_code, "shimmer", story_id, voice_preset)
            logger.error(f"[TTS-ELEVEN] Error on segment {i+1}: {error_str}")
            continue
    
    if not audio_segments:
        logger.error("[TTS-ELEVEN] No audio generated, falling back to OpenAI")
        return await generate_tts_audio_openai(text, language_code, "shimmer", story_id, voice_preset)
    
    # ==================== CONCATENATE AND POST-PROCESS ====================
    logger.info(f"[TTS-ELEVEN] Concatenating {len(audio_segments)} segments...")
    
    combined_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        combined_audio = combined_audio + segment
    
    # Add a gentle fade at the very end for a calm conclusion
    combined_audio = combined_audio.fade_out(duration=500)
    
    # Export as high-quality MP3
    output_buffer = io.BytesIO()
    combined_audio.export(output_buffer, format="mp3", bitrate="128k")
    audio_bytes = output_buffer.getvalue()
    
    duration_seconds = len(combined_audio) / 1000
    logger.info(f"[TTS-ELEVEN] Final audio: {duration_seconds:.1f} seconds, {total_chars} chars")
    
    estimated_cost = log_tts_metrics(
        story_id=story_id,
        provider="elevenlabs",
        chars=total_chars,
        language=lang_code,
        voice=voice_id,
        success=True
    )
    
    logger.info(f"[TTS-ELEVEN] SUCCESS: {len(audio_bytes)} bytes, {total_chars} chars, ~${estimated_cost:.4f}")
    logger.info("=" * 60)
    
    return (audio_bytes, voice_id, total_chars, estimated_cost)



async def generate_tts_audio(text: str, language_code: str, voice_id: str = None, story_id: str = None, provider: str = "elevenlabs", voice_preset: str = None, voice_id_override: str = None) -> tuple:
    """
    Main TTS generation function - uses ElevenLabs for expressive bedtime narration.
    
    Default: ElevenLabs with sentence-by-sentence generation for natural prosody
    Fallback: OpenAI if ElevenLabs is unavailable
    
    Args:
        voice_id_override: If provided (e.g., for parent voice), use this voice ID directly
    
    Returns: (audio_bytes, voice_used, char_count, estimated_cost)
    """
    char_count = len(text)
    
    # Determine voice preset (narrator personality)
    narrator_preset = voice_preset or voice_id or DEFAULT_NARRATOR
    if narrator_preset not in VOICE_PRESETS:
        narrator_preset = DEFAULT_NARRATOR
    
    preset = VOICE_PRESETS.get(narrator_preset, {})
    preset_provider = preset.get("provider", "elevenlabs")
    
    logger.info(f"[TTS] Routing: preset={narrator_preset}, provider={preset_provider}, chars={char_count}, voice_override={voice_id_override}")
    
    # Use ElevenLabs expressive generation for all narrator personalities
    if preset_provider == "elevenlabs":
        return await generate_tts_elevenlabs_expressive(
            text=text,
            language_code=language_code,
            voice_preset=narrator_preset,
            story_id=story_id,
            voice_id_override=voice_id_override  # Pass parent voice ID if provided
        )
    else:
        # Fallback to OpenAI
        return await generate_tts_audio_openai(text, language_code, "shimmer", story_id, narrator_preset)

@api_router.post("/narration/request", response_model=NarrationResponse)
async def request_narration(request: NarrationRequest, user_id: str = Depends(get_current_user)):
    """
    Request narration for a story with MULTILINGUAL support.
    - Checks subscription limits (free: 2/day, premium: unlimited)
    - Stores audio per language: {userId}/{storyId}/{lang}.mp3
    - Translates text on-the-fly if narration language differs from story language
    - Uses ElevenLabs eleven_multilingual_v2 model with language_code parameter
    - NEVER fails silently - always records error details.
    """
    story_id = request.storyId
    narration_language_code = request.narrationLanguageCode
    
    # ==================== SUBSCRIPTION CHECK ====================
    subscription = await get_user_subscription(user_id)
    narration_access = check_feature_access(subscription, "narration")
    
    if not narration_access["allowed"]:
        logger.info(f"[NARRATION] User {user_id} hit narration limit")
        raise HTTPException(
            status_code=403,
            detail={
                "error": "narration_limit_reached",
                "message": narration_access.get("upgrade_message", "Daily narration limit reached"),
                "upgrade_required": True,
                "narrations_used": narration_access.get("narrations_used", 0),
                "narrations_limit": narration_access.get("narrations_limit", FREE_DAILY_NARRATION_LIMIT),
            }
        )
    
    # Check narrator access (premium narrators)
    if request.voicePreference and request.voicePreference != "calm_storyteller":
        narrator_access = check_feature_access(subscription, "narrator", request.voicePreference)
        if not narrator_access["allowed"]:
            logger.info(f"[NARRATION] User {user_id} tried premium narrator: {request.voicePreference}")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "premium_narrator",
                    "message": narrator_access.get("reason", "Premium narrator required"),
                    "upgrade_required": True,
                }
            )
    
    # ==================== PARENT VOICE HANDLING ====================
    # If parent_voice is selected, fetch the user's actual voice ID from the database
    parent_voice_id_override = None
    if request.voicePreference == "parent_voice":
        try:
            user_result = supabase.table('users_profile').select('parent_voice_id, parent_voice_status').eq('id', user_id).execute()
            if user_result.data and len(user_result.data) > 0:
                parent_voice_id_override = user_result.data[0].get('parent_voice_id')
                parent_voice_status = user_result.data[0].get('parent_voice_status', 'none')
                
                if not parent_voice_id_override or parent_voice_status != 'ready':
                    logger.warning(f"[NARRATION] Parent voice not ready for user {user_id}: status={parent_voice_status}")
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "parent_voice_not_ready",
                            "message": "Parent Voice is not set up yet. Please record your voice in Settings first.",
                            "setup_required": True,
                        }
                    )
                logger.info(f"[NARRATION] Using parent voice: {parent_voice_id_override}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[NARRATION] Error fetching parent voice: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to load Parent Voice. Please try again."
            )
    
    # ==================== LANGUAGE RESTRICTION FOR LAUNCH ====================
    # System narrators (Calm Storyteller, Wise Owl) only support English for quality
    # Parent Voice can narrate in any language
    # If non-English language requested with system narrator, fall back to English
    original_narration_language = narration_language_code
    language_fallback_applied = False
    
    # NOTE: Language restriction removed - we now have language-specific narrators
    # The frontend auto-selects a compatible narrator for the language
    # (e.g., night_owl_spanish for Spanish, night_owl_german for German, etc.)
    
    # ==================== VERIFICATION LOGGING ====================
    logger.info("=" * 60)
    logger.info("[NARRATION] REQUEST RECEIVED")
    logger.info(f"[NARRATION] story_id: {story_id}")
    logger.info(f"[NARRATION] narration_language: {narration_language_code}")
    logger.info(f"[NARRATION] voicePreference (narrator): {request.voicePreference}")
    logger.info(f"[NARRATION] user_id: {user_id}")
    logger.info(f"[NARRATION] subscription: {subscription['status']}, narrations_used: {subscription['daily_narrations_used']}")
    
    # Debug: Check if voicePreference is a valid narrator
    if request.voicePreference:
        preset = VOICE_PRESETS.get(request.voicePreference)
        if preset:
            logger.info(f"[NARRATION] Voice preset found: name={preset.get('name')}, voice_id={preset.get('voice_id')}")
        else:
            logger.warning(f"[NARRATION] WARNING: voicePreference '{request.voicePreference}' NOT FOUND in VOICE_PRESETS!")
            logger.info(f"[NARRATION] Available presets: {list(VOICE_PRESETS.keys())}")
    else:
        logger.info(f"[NARRATION] No voicePreference provided, will use DEFAULT_NARRATOR: {DEFAULT_NARRATOR}")
    logger.info("=" * 60)
    
    def build_error_string(err: Exception, context: str = "") -> str:
        """Build a readable error string from any exception"""
        status = getattr(err, 'status_code', None) or getattr(err, 'status', None) or 'unknown'
        body = ""
        if hasattr(err, 'response'):
            try:
                if hasattr(err.response, 'text'):
                    body = str(err.response.text)[:500]
                elif hasattr(err.response, 'data'):
                    body = json.dumps(err.response.data)[:500]
            except:
                pass
        msg = str(err) if str(err) else 'unknown error'
        full = f"{status} | {context}: {msg}"
        if body:
            full += f" | {body}"
        return full[:1000]
    
    def update_story_error(error_msg: str, lang: str = None):
        """Update story with failed status and error message"""
        try:
            update_data = {'audio_status': 'failed', 'audio_error': error_msg}
            # Also update per-language status if JSONB columns exist
            if lang:
                try:
                    # Get current JSONB data
                    current = supabase.table('stories').select('audio_status_by_lang').eq('id', story_id).execute()
                    if current.data and len(current.data) > 0:
                        status_by_lang = current.data[0].get('audio_status_by_lang') or {}
                        if isinstance(status_by_lang, str):
                            status_by_lang = json.loads(status_by_lang)
                        status_by_lang[lang] = 'failed'
                        update_data['audio_status_by_lang'] = json.dumps(status_by_lang)
                except:
                    pass
            supabase.table('stories').update(update_data).eq('id', story_id).execute()
            logger.info(f"[NARRATION] Updated error status for story {story_id}")
        except Exception as db_err:
            try:
                supabase.table('stories').update({'audio_status': 'failed'}).eq('id', story_id).execute()
            except:
                logger.error(f"[NARRATION] Failed to update story status: {str(db_err)}")
    
    try:
        # STEP 1: Fetch story with per-language audio columns (with fallbacks)
        try:
            story_result = supabase.table('stories').select(
                'id, user_id, title, pages, full_text, language, audio_url, audio_path_by_lang, audio_status_by_lang'
            ).eq('id', story_id).eq('user_id', user_id).execute()
        except:
            # Fall back to basic columns if JSONB columns don't exist
            story_result = supabase.table('stories').select(
                'id, user_id, title, pages, full_text, language, audio_url'
            ).eq('id', story_id).eq('user_id', user_id).execute()
        
        if not story_result.data or len(story_result.data) == 0:
            error_msg = "404 | Story not found or not owned by user"
            logger.error(f"[NARRATION] {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        story = story_result.data[0]
        story_language = story.get('language', 'en') or 'en'
        logger.info(f"[NARRATION] Story found: '{story.get('title')}' (story_language={story_language})")
        
        # STEP 2: Determine narration language
        narration_lang = (narration_language_code or '').strip().lower()[:2]
        if not narration_lang or narration_lang == 'sa':  # 'sa' from 'same_as_story'
            narration_lang = story_language
        logger.info(f"[NARRATION] Resolved narration_language: {narration_lang}")
        
        # STEP 3: Check if audio for THIS NARRATOR + LANGUAGE combination already exists
        # Cache key format: {narrator_id}_{lang} to ensure different narrators have separate audio
        audio_path_by_lang = story.get('audio_path_by_lang') or {}
        if isinstance(audio_path_by_lang, str):
            try:
                audio_path_by_lang = json.loads(audio_path_by_lang)
            except:
                audio_path_by_lang = {}
        
        audio_status_by_lang = story.get('audio_status_by_lang') or {}
        if isinstance(audio_status_by_lang, str):
            try:
                audio_status_by_lang = json.loads(audio_status_by_lang)
            except:
                audio_status_by_lang = {}
        
        # Build the NARRATOR-SPECIFIC cache key
        narrator_id_for_cache = request.voicePreference or DEFAULT_NARRATOR
        cache_key = f"{narrator_id_for_cache}_{narration_lang}"
        
        existing_path = audio_path_by_lang.get(cache_key)
        existing_status = audio_status_by_lang.get(cache_key)
        
        logger.info(f"[NARRATION] Cache check: key='{cache_key}', path={existing_path is not None}, status={existing_status}")
        
        # Legacy fallback ONLY if narrator matches default and no narrator-specific audio exists
        # This ensures switching narrators triggers regeneration, not legacy fallback
        legacy_audio_url = story.get('audio_url')
        if not existing_path and legacy_audio_url and narrator_id_for_cache == DEFAULT_NARRATOR and narration_lang == story_language:
            # Check if legacy path matches the expected narrator
            if DEFAULT_NARRATOR in str(legacy_audio_url):
                existing_path = legacy_audio_url
                existing_status = 'ready'
                logger.info(f"[NARRATION] Using legacy audio_url for default narrator: {legacy_audio_url}")
        
        if existing_path and existing_status == 'ready':
            # VERIFY the file actually exists in Supabase Storage before returning cache hit
            # This prevents returning broken cache if file was deleted but DB wasn't updated
            try:
                # Try to create a signed URL - if the file doesn't exist, this will fail
                verify_result = supabase.storage.from_('story-audio').create_signed_url(existing_path, 60)
                if verify_result and verify_result.get('signedURL'):
                    logger.info(f"[NARRATION] CACHE HIT ✓ Audio file verified at: {existing_path}")
                    logger.info(f"[NARRATION] Returning cached audio for narrator='{narrator_id_for_cache}', lang='{narration_lang}'")
                    logger.info("[NARRATION] Skipping ElevenLabs API call - saving costs!")
                    return NarrationResponse(
                        status='ready',
                        audioUrl=None,  # Frontend will request signed URL
                        message="Audio ready (cached)"
                    )
                else:
                    logger.warning(f"[NARRATION] CACHE MISS - File not found in storage: {existing_path}")
                    # File doesn't exist, clear the stale cache entry
                    audio_path_by_lang.pop(cache_key, None)
                    audio_status_by_lang.pop(cache_key, None)
            except Exception as verify_err:
                logger.warning(f"[NARRATION] CACHE MISS - Storage verification failed: {str(verify_err)}")
                # File likely doesn't exist, clear stale cache
                audio_path_by_lang.pop(cache_key, None)
                audio_status_by_lang.pop(cache_key, None)
        
        # DUPLICATE REQUEST PROTECTION: If audio is currently being generated, don't start another
        if existing_status == 'generating':
            logger.info(f"[NARRATION] DUPLICATE BLOCKED: Audio already being generated for narrator='{narrator_id_for_cache}', lang='{narration_lang}'")
            return NarrationResponse(
                status='generating',
                audioUrl=None,
                message="Narration is already being generated. Please wait..."
            )
        
        logger.info(f"[NARRATION] CACHE MISS - Generating new audio for narrator='{narrator_id_for_cache}', lang='{narration_lang}'")
        
        # ==================== TTS GENERATION BLOCK ====================
        # If TTS generation is blocked, return a helpful message
        if TTS_GENERATION_BLOCKED:
            logger.warning(f"[NARRATION] TTS GENERATION BLOCKED - No cached audio for {cache_key}")
            raise HTTPException(
                status_code=503,
                detail="Narration generation is temporarily paused to preserve credits. Please use a story with existing cached narration, or wait until generation is re-enabled."
            )
        # ==============================================================
        
        # STEP 4: Check user plan and limits
        user_result = supabase.table('users_profile').select('plan, email, trial_narrations_used').eq('id', user_id).execute()
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        user_data = user_result.data[0]
        user_plan = user_data.get('plan', 'free')
        user_email = user_data.get('email', '')
        
        # Check if this is a tester account - bypass all restrictions
        is_tester = user_email and user_email.lower() in [e.lower() for e in TESTER_EMAILS]
        if is_tester:
            logger.info(f"[NARRATION] Tester account ({user_email}) - bypassing plan restrictions")
            user_plan = 'premium'  # Treat testers as premium
        
        logger.info(f"[NARRATION] User plan: {user_plan} (tester: {is_tester})")
        
        if user_plan == 'free':
            # Free users get 2 narrations per day
            can_narrate = subscription.get("can_narrate", True)
            if not can_narrate:
                raise HTTPException(
                    status_code=403,
                    detail=json.dumps({"error": "DAILY_LIMIT", "message": "You've used your daily narrations. Come back tomorrow or upgrade to Premium!"})
                )
        
        if user_plan == 'trial':
            trial_used = user_data.get('trial_narrations_used', 0) or 0
            if trial_used >= 15:
                raise HTTPException(
                    status_code=403,
                    detail=json.dumps({"error": "TRIAL_LIMIT", "message": "You've used all 15 trial narrations. Upgrade to Premium!"})
                )
        
        # STEP 5: Update status to 'generating' for this language
        audio_status_by_lang[cache_key] = 'generating'
        try:
            supabase.table('stories').update({
                'audio_status': 'generating',
                'audio_status_by_lang': json.dumps(audio_status_by_lang),
                'audio_error': None
            }).eq('id', story_id).execute()
            logger.info(f"[NARRATION] Set status='generating' for key={cache_key}")
        except:
            # Fall back to legacy column only
            try:
                supabase.table('stories').update({'audio_status': 'generating'}).eq('id', story_id).execute()
            except:
                pass
        
        # ==================== BACKGROUND PROCESSING ====================
        # Return immediately and process narration in background to avoid proxy timeouts
        # For 8-15 minute stories, TTS can take 2-5 minutes which exceeds proxy limits
        
        # Prepare data for background task
        background_data = {
            'story_id': story_id,
            'user_id': user_id,
            'story': story,
            'story_language': story_language,
            'narration_lang': narration_lang,
            'cache_key': cache_key,
            'narrator_id_for_cache': narrator_id_for_cache,
            'voice_preference': request.voicePreference,
            'parent_voice_id_override': parent_voice_id_override,
            'user_plan': user_data.get('plan', 'free'),
            'is_tester': is_tester,
        }
        
        # Launch background task
        asyncio.create_task(process_narration_background(background_data))
        
        logger.info(f"[NARRATION] ✓ Background task launched for story {story_id}")
        logger.info("[NARRATION] Returning immediately to prevent proxy timeout")
        
        return NarrationResponse(
            status='generating',
            audioUrl=None,
            message="Narration is being generated. Please check back in a moment..."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = build_error_string(e, "Narration request failed")
        logger.error(f"[NARRATION] {error_msg}")
        update_story_error(error_msg, narration_language_code)
        raise HTTPException(status_code=500, detail="Failed to start narration generation")


# ==================== BACKGROUND NARRATION PROCESSOR ====================
# This runs in the background after the endpoint returns immediately
# Handles the actual TTS generation and upload to avoid proxy timeouts

async def process_narration_background(data: dict):
    """
    Process narration generation in the background.
    This allows the API endpoint to return immediately while TTS runs.
    """
    story_id = data['story_id']
    user_id = data['user_id']
    story = data['story']
    story_language = data['story_language']
    narration_lang = data['narration_lang']
    cache_key = data['cache_key']
    narrator_id_for_cache = data['narrator_id_for_cache']
    voice_preference = data.get('voice_preference')
    parent_voice_id_override = data.get('parent_voice_id_override')
    user_plan = data.get('user_plan', 'free')
    is_tester = data.get('is_tester', False)
    
    logger.info(f"[NARRATION-BG] Starting background processing for story {story_id}")
    
    def update_story_error_bg(error_msg: str, lang: str = None):
        """Update story with failed status and error message"""
        try:
            update_data = {'audio_status': 'failed', 'audio_error': error_msg}
            if lang:
                try:
                    current = supabase.table('stories').select('audio_status_by_lang').eq('id', story_id).execute()
                    if current.data and len(current.data) > 0:
                        status_by_lang = current.data[0].get('audio_status_by_lang') or {}
                        if isinstance(status_by_lang, str):
                            status_by_lang = json.loads(status_by_lang)
                        status_by_lang[cache_key] = 'failed'
                        update_data['audio_status_by_lang'] = json.dumps(status_by_lang)
                except:
                    pass
            supabase.table('stories').update(update_data).eq('id', story_id).execute()
            logger.info(f"[NARRATION-BG] Updated error status for story {story_id}")
        except Exception as db_err:
            logger.error(f"[NARRATION-BG] Failed to update story status: {str(db_err)}")
    
    try:
        # STEP 6: Get story text
        story_text = story.get('full_text', '')
        if not story_text:
            pages = story.get('pages', [])
            story_text = " ".join(pages) if pages else ""
        
        if not story_text:
            error_msg = "400 | Story has no text to narrate"
            update_story_error_bg(error_msg, narration_lang)
            return
        
        logger.info(f"[NARRATION-BG] Story text: {len(story_text)} chars in '{story_language}'")
        
        # STEP 7: TRANSLATE if narration language differs from story language
        text_for_tts = story_text
        if narration_lang != story_language:
            logger.info(f"[NARRATION-BG] TRANSLATING: {story_language} -> {narration_lang}")
        text_for_tts = await translate_text_for_narration(...)

        if not text_for_tts or not isinstance(text_for_tts, str):
            logger.warning("[NARRATION-BG] Invalid translated text, using original")
            text_for_tts = story_text
            logger.info(f"[NARRATION-BG] Using fallback text: {len(text_for_tts)} chars")
        else:
            logger.info(f"[NARRATION-BG] Translation complete: {len(text_for_tts)} chars")
        
        # Apply TTS text normalization for better pacing
        text_for_tts = normalize_text_for_tts(text_for_tts)
        
        # Clean up text for smoother narration (remove awkward commas, etc.)
        text_for_tts = clean_text_for_narration(text_for_tts)
        
        # Determine TTS provider
        tts_provider = "elevenlabs" if should_use_elevenlabs(voice_preference=voice_preference) else "openai"

        logger.info(f"[NARRATION-BG] provider: {tts_provider}")
        logger.info(f"[NARRATION-BG] elevenlabs_key_present: {elevenlabs_available()}")
        logger.info(f"[NARRATION-BG] USE_ELEVENLABS: {USE_ELEVENLABS}")
        logger.info(f"[NARRATION-BG] voice_preference: {voice_preference}")
        logger.info("-" * 60)
        logger.info("[NARRATION-BG] SENDING TO TTS:")
        logger.info(f"[NARRATION-BG]   story_id: {story_id}")
        logger.info(f"[NARRATION-BG]   provider: {tts_provider}")
        logger.info(f"[NARRATION-BG]   narration_language: {narration_lang}")
        logger.info(f"[NARRATION-BG]   tts_chars: {len(text_for_tts)}")
        logger.info("-" * 60)
        
        # STEP 8: Generate TTS
        try:
            audio_bytes, voice_id_used, char_count, estimated_cost = await generate_tts_audio(
                text=text_for_tts,
                language_code=narration_lang,
                voice_preset=voice_preference,
                story_id=story_id,
                provider=tts_provider,
                voice_id_override=parent_voice_id_override
            )
            logger.info(f"[NARRATION-BG] TTS SUCCESS: {char_count} chars, voice={voice_id_used}, cost=${estimated_cost:.4f}")
        except HTTPException as http_err:
            logger.error(f"[NARRATION-BG] TTS Error: {http_err.detail}")
            update_story_error_bg(http_err.detail, narration_lang)
            return
        except Exception as tts_error:
            error_msg = f"TTS generation failed: {str(tts_error)}"
            logger.error(f"[NARRATION-BG] {error_msg}")
            update_story_error_bg(error_msg, narration_lang)
            return
        
        # STEP 9: COMPRESS audio if needed
        original_size = len(audio_bytes)
        if original_size > 4 * 1024 * 1024:
            logger.info(f"[NARRATION-BG] Audio file large ({original_size/1024/1024:.1f}MB) - compressing...")
            audio_bytes = compress_audio_for_upload(audio_bytes, target_bitrate="48k")
            
            if len(audio_bytes) > 5 * 1024 * 1024:
                logger.info(f"[NARRATION-BG] Still large ({len(audio_bytes)/1024/1024:.1f}MB) - aggressive compression...")
                audio_bytes = compress_audio_for_upload(audio_bytes, target_bitrate="32k")
        
        # STEP 10: Upload to Supabase Storage
        narrator_id = voice_preference or DEFAULT_NARRATOR
        storage_path = f"{user_id}/{story_id}/{narrator_id}_{narration_lang}.mp3"
        logger.info(f"[NARRATION-BG] Uploading audio ({len(audio_bytes)/1024/1024:.1f}MB) to: {storage_path}")
        
        try:
            storage_result = supabase.storage.from_('story-audio').upload(
                storage_path,
                audio_bytes,
                {"content-type": "audio/mpeg", "upsert": "true"}
            )
            logger.info(f"[NARRATION-BG] Upload SUCCESS: {storage_path}")
        except Exception as storage_error:
            error_msg = f"Storage upload failed: {str(storage_error)}"
            logger.error(f"[NARRATION-BG] {error_msg}")
            update_story_error_bg(error_msg, narration_lang)
            return
        
        # STEP 11: Update database with completed audio path
        # Get current audio_path_by_lang and audio_status_by_lang
        try:
            current_story = supabase.table('stories').select('audio_path_by_lang, audio_status_by_lang').eq('id', story_id).execute()
            audio_path_by_lang = {}
            audio_status_by_lang = {}
            if current_story.data and len(current_story.data) > 0:
                audio_path_by_lang = current_story.data[0].get('audio_path_by_lang') or {}
                audio_status_by_lang = current_story.data[0].get('audio_status_by_lang') or {}
                if isinstance(audio_path_by_lang, str):
                    audio_path_by_lang = json.loads(audio_path_by_lang)
                if isinstance(audio_status_by_lang, str):
                    audio_status_by_lang = json.loads(audio_status_by_lang)
        except:
            audio_path_by_lang = {}
            audio_status_by_lang = {}
        
        audio_path_by_lang[cache_key] = storage_path
        audio_status_by_lang[cache_key] = 'ready'
        
        try:
            supabase.table('stories').update({
                'audio_url': storage_path,
                'audio_status': 'ready',
                'audio_path_by_lang': json.dumps(audio_path_by_lang),
                'audio_status_by_lang': json.dumps(audio_status_by_lang),
                'audio_error': None,
                'audio_language_code': narration_lang,
                'audio_voice_id': voice_id_used,
                'audio_chars': char_count
            }).eq('id', story_id).execute()
            logger.info("[NARRATION-BG] DB UPDATE SUCCESS")
        except Exception as update_err:
            try:
                supabase.table('stories').update({
                    'audio_url': storage_path,
                    'audio_status': 'ready'
                }).eq('id', story_id).execute()
            except:
                logger.error(f"[NARRATION-BG] Failed to update DB: {str(update_err)}")
        
        # STEP 12: Update trial usage if applicable
        if user_plan == 'trial':
            try:
                current_user = supabase.table('users_profile').select('trial_narrations_used').eq('id', user_id).execute()
                if current_user.data:
                    trial_used = current_user.data[0].get('trial_narrations_used', 0) or 0
                    supabase.table('users_profile').update({
                        'trial_narrations_used': trial_used + 1
                    }).eq('id', user_id).execute()
            except:
                pass
        
        # Increment daily narration counter for free users
        if user_plan == 'free':
            try:
                await increment_narration_usage(user_id)
            except:
                pass
        
        logger.info("=" * 60)
        logger.info("[NARRATION-BG] ✓ SUCCESS!")
        logger.info(f"[NARRATION-BG]   story_id: {story_id}")
        logger.info(f"[NARRATION-BG]   narration_language: {narration_lang}")
        logger.info(f"[NARRATION-BG]   audio_path: {storage_path}")
        logger.info(f"[NARRATION-BG]   chars_narrated: {char_count}")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"Background narration failed: {str(e)}"
        logger.error(f"[NARRATION-BG] UNEXPECTED ERROR: {error_msg}")
        update_story_error_bg(error_msg, narration_lang)

@api_router.get("/narration/status")
async def get_narration_status(story_id: str, user_id: str = Depends(get_current_user)):
    """
    Get the current narration status for a story.
    Works with or without the audio_status column.
    """
    try:
        story_result = supabase.table('stories').select('audio_url, audio_status').eq('id', story_id).eq('user_id', user_id).execute()
        
        if not story_result.data or len(story_result.data) == 0:
            raise HTTPException(status_code=404, detail="Story not found")
        
        story = story_result.data[0]
        
        # Determine status from audio_status column or audio_url presence
        status = story.get('audio_status', 'none')
        if status == 'none' and story.get('audio_url'):
            status = 'ready'
        
        return {
            "audioStatus": status,
            "hasAudio": bool(story.get('audio_url'))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting narration status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get narration status")


# ==================== PAGE-1-FIRST CHUNKED NARRATION ====================
# This system generates audio page-by-page, allowing playback to start after
# just Page 1 is ready (~10-15 seconds) instead of waiting for the entire story.
# Remaining pages generate in background while user listens to Page 1.

@api_router.post("/narration/request-chunked", response_model=ChunkedNarrationResponse)
async def request_chunked_narration(request: NarrationRequest, user_id: str = Depends(get_current_user)):
    """
    PAGE-1-FIRST Narration: Generate narration page-by-page for faster start.
    
    Flow:
    1. User taps "Generate Narration"
    2. Page 1 generates immediately (~10-15 sec)
    3. Returns as soon as Page 1 is ready
    4. Remaining pages generate in background
    5. Frontend polls /narration/page-status for progress
    """
    story_id = request.storyId
    narration_language_code = request.narrationLanguageCode
    
    # ==================== SUBSCRIPTION CHECK ====================
    subscription = await get_user_subscription(user_id)
    narration_access = check_feature_access(subscription, "narration")
    
    if not narration_access["allowed"]:
        logger.info(f"[CHUNKED] User {user_id} hit narration limit")
        raise HTTPException(
            status_code=403,
            detail={
                "error": "narration_limit_reached",
                "message": narration_access.get("upgrade_message", "Daily narration limit reached"),
                "upgrade_required": True,
            }
        )
    
    # Check narrator access (premium narrators)
    if request.voicePreference and request.voicePreference != "calm_storyteller":
        narrator_access = check_feature_access(subscription, "narrator", request.voicePreference)
        if not narrator_access["allowed"]:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "premium_narrator",
                    "message": narrator_access.get("reason", "Premium narrator required"),
                    "upgrade_required": True,
                }
            )
    
    # ==================== PARENT VOICE HANDLING ====================
    parent_voice_id_override = None
    if request.voicePreference == "parent_voice":
        try:
            user_result = supabase.table('users_profile').select('parent_voice_id, parent_voice_status').eq('id', user_id).execute()
            if user_result.data and len(user_result.data) > 0:
                parent_voice_id_override = user_result.data[0].get('parent_voice_id')
                parent_voice_status = user_result.data[0].get('parent_voice_status', 'none')
                
                if not parent_voice_id_override or parent_voice_status != 'ready':
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "parent_voice_not_ready",
                            "message": "Parent Voice is not set up yet. Please record your voice in Settings first.",
                            "setup_required": True,
                        }
                    )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[CHUNKED] Error fetching parent voice: {e}")
            raise HTTPException(status_code=500, detail="Failed to load Parent Voice")
    
    # NOTE: Language restriction removed - we now have language-specific narrators
    # The frontend should auto-select a compatible narrator for the language
    # (e.g., night_owl_spanish for Spanish, night_owl_german for German, etc.)
    # Parent Voice works with any language since the user records in their language
    
    logger.info("=" * 60)
    logger.info("[CHUNKED] PAGE-1-FIRST NARRATION REQUEST")
    logger.info(f"[CHUNKED] story_id: {story_id}")
    logger.info(f"[CHUNKED] narrator: {request.voicePreference or DEFAULT_NARRATOR}")
    logger.info(f"[CHUNKED] language: {narration_language_code}")
    logger.info("=" * 60)
    
    try:
        # Fetch story
        story_result = supabase.table('stories').select(
            'id, user_id, title, pages, language, chunked_audio_status'
        ).eq('id', story_id).eq('user_id', user_id).execute()
        
        if not story_result.data:
            raise HTTPException(status_code=404, detail="Story not found")
        
        story = story_result.data[0]
        pages = story.get('pages', [])
        total_pages = len(pages)
        story_language = story.get('language', 'en') or 'en'
        
        if total_pages == 0:
            raise HTTPException(status_code=400, detail="Story has no pages")
        
        # Resolve narration language
        narration_lang = (narration_language_code or '').strip().lower()[:2]
        if not narration_lang or narration_lang == 'sa':
            narration_lang = story_language
        
        # Build cache key for this narrator+language combo
        narrator_id = request.voicePreference or DEFAULT_NARRATOR
        cache_key = f"{narrator_id}_{narration_lang}"
        
        # Check for existing chunked audio status
        chunked_status = story.get('chunked_audio_status') or {}
        if isinstance(chunked_status, str):
            try:
                chunked_status = json.loads(chunked_status)
            except:
                chunked_status = {}
        
        narrator_status = chunked_status.get(cache_key, {})
        pages_ready = narrator_status.get('pages_ready', [])
        pages_generating = narrator_status.get('pages_generating', [])
        
        # Check if Page 1 is already ready (cache hit)
        if 1 in pages_ready:
            logger.info(f"[CHUNKED] CACHE HIT: Page 1 already ready for {cache_key}")
            
            # Get signed URL for page 1
            page1_path = f"{user_id}/{story_id}/chunked/{cache_key}/page_1.mp3"
            try:
                signed_url_result = supabase.storage.from_('story-audio').create_signed_url(page1_path, 3600)
                page1_url = signed_url_result.get('signedURL') if signed_url_result else None
            except:
                page1_url = None
            
            return ChunkedNarrationResponse(
                status='page_ready' if len(pages_ready) < total_pages else 'all_ready',
                currentPage=1,
                totalPages=total_pages,
                pageAudioUrl=page1_url,
                pagesReady=pages_ready,
                message=f"Page 1 ready. {len(pages_ready)}/{total_pages} pages complete."
            )
        
        # Check if already generating
        if 1 in pages_generating:
            logger.info(f"[CHUNKED] Page 1 already generating for {cache_key}")
            return ChunkedNarrationResponse(
                status='generating',
                currentPage=1,
                totalPages=total_pages,
                pagesReady=pages_ready,
                message="Page 1 is being generated..."
            )
        
        # ==================== TTS GENERATION BLOCK ====================
        # If TTS generation is blocked, return a helpful message
        if TTS_GENERATION_BLOCKED:
            logger.warning(f"[CHUNKED] TTS GENERATION BLOCKED - No cached audio for {cache_key}")
            raise HTTPException(
                status_code=503,
                detail="Narration generation is temporarily paused to preserve credits. Please use a story with existing cached narration, or wait until generation is re-enabled."
            )
        # ==============================================================
        
        # Get user plan info
        user_result = supabase.table('users_profile').select('plan, email').eq('id', user_id).execute()
        user_data = user_result.data[0] if user_result.data else {}
        user_plan = user_data.get('plan', 'free')
        user_email = user_data.get('email', '')
        is_tester = user_email and user_email.lower() in [e.lower() for e in TESTER_EMAILS]
        
        # Update status: Page 1 is now generating
        pages_generating = [1]
        narrator_status = {
            'pages_ready': [],
            'pages_generating': pages_generating,
            'pages_failed': [],
            'started_at': datetime.utcnow().isoformat()
        }
        chunked_status[cache_key] = narrator_status
        
        try:
            supabase.table('stories').update({
                'chunked_audio_status': json.dumps(chunked_status)
            }).eq('id', story_id).execute()
        except Exception as e:
            logger.warning(f"[CHUNKED] Could not update status: {e}")
        
        # Prepare background task data
        background_data = {
            'story_id': story_id,
            'user_id': user_id,
            'pages': pages,
            'story_language': story_language,
            'narration_lang': narration_lang,
            'cache_key': cache_key,
            'narrator_id': narrator_id,
            'voice_preference': request.voicePreference,
            'parent_voice_id_override': parent_voice_id_override,
            'user_plan': user_plan,
            'is_tester': is_tester,
        }
        
        # Launch background task for Page 1 FIRST, then remaining pages
        asyncio.create_task(process_chunked_narration_page1_first(background_data))
        
        logger.info(f"[CHUNKED] ✓ Background task launched for {total_pages} pages")
        
        return ChunkedNarrationResponse(
            status='generating',
            currentPage=1,
            totalPages=total_pages,
            pagesReady=[],
            message="Generating Page 1 narration... (~10-15 seconds)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHUNKED] Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start chunked narration")


async def process_chunked_narration_page1_first(data: dict):
    """
    PAGE-1-FIRST Background Processor:
    1. Generate Page 1 audio immediately
    2. Update DB so frontend can start playback
    3. Generate remaining pages in sequence
    """
    story_id = data['story_id']
    user_id = data['user_id']
    pages = data['pages']
    story_language = data['story_language']
    narration_lang = data['narration_lang']
    cache_key = data['cache_key']
    narrator_id = data['narrator_id']
    voice_preference = data.get('voice_preference')
    parent_voice_id_override = data.get('parent_voice_id_override')
    user_plan = data.get('user_plan', 'free')
    is_tester = data.get('is_tester', False)
    
    total_pages = len(pages)
    pages_ready = []
    pages_failed = []
    
    logger.info(f"[CHUNKED-BG] Starting Page-1-First generation for {total_pages} pages")
    
    def update_chunked_status(pages_ready: list, pages_generating: list, pages_failed: list):
        """Update the chunked audio status in DB"""
        try:
            current = supabase.table('stories').select('chunked_audio_status').eq('id', story_id).execute()
            chunked_status = {}
            if current.data and current.data[0].get('chunked_audio_status'):
                chunked_status = current.data[0].get('chunked_audio_status')
                if isinstance(chunked_status, str):
                    chunked_status = json.loads(chunked_status)
            
            chunked_status[cache_key] = {
                'pages_ready': pages_ready,
                'pages_generating': pages_generating,
                'pages_failed': pages_failed,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            supabase.table('stories').update({
                'chunked_audio_status': json.dumps(chunked_status)
            }).eq('id', story_id).execute()
        except Exception as e:
            logger.error(f"[CHUNKED-BG] Failed to update status: {e}")
    
    async def generate_page_audio(page_num: int, page_text: str) -> bool:
        """Generate audio for a single page. Returns True on success."""
        try:
            # Clean page text (remove narration markers)
            clean_text = page_text.replace('[NARRATION_START]', '').replace('[NARRATION_END]', '').strip()
            
            if not clean_text:
                logger.warning(f"[CHUNKED-BG] Page {page_num} has no text, skipping")
                return True  # Skip empty pages
            
            # Translate if needed
            text_for_tts = clean_text
            if narration_lang != story_language:
                text_for_tts = await translate_text_for_narration(clean_text, story_language, narration_lang)
            # SAFETY CHECK (CRITICAL FIX)
            if not text_for_tts or not isinstance(text_for_tts, str):
                logger.warning("[CHUNKED-BG] Invalid translated text, using original")
                text_for_tts = clean_text
    
            # Apply TTS text normalization for better pacing
            text_for_tts = normalize_text_for_tts(text_for_tts)
            
            # Clean up text for smoother narration (remove awkward commas, etc.)
            text_for_tts = clean_text_for_narration(text_for_tts)
            
            logger.info(f"[CHUNKED-BG] Generating Page {page_num}: {len(text_for_tts)} chars")
            
            # Generate TTS
            # Generate TTS
            tts_provider = "elevenlabs" if should_use_elevenlabs(voice_preference=voice_preference) else "openai"

            logger.info(f"[CHUNKED-BG] TTS provider selected: {tts_provider}")
            logger.info(f"[CHUNKED-BG] elevenlabs_key_present: {elevenlabs_available()}")
            logger.info(f"[CHUNKED-BG] USE_ELEVENLABS: {USE_ELEVENLABS}")

            audio_bytes, voice_id_used, char_count, estimated_cost = await generate_tts_audio(
            text=text_for_tts,
            language_code=narration_lang,
            voice_preset=voice_preference,
            story_id=f"{story_id}_page{page_num}",
            provider=tts_provider,
            voice_id_override=parent_voice_id_override
            )
            
            logger.info(f"[CHUNKED-BG] Page {page_num} TTS complete: {len(audio_bytes)} bytes, ${estimated_cost:.4f}")
            
            # Compress if needed (individual pages should be smaller, but just in case)
            if len(audio_bytes) > 2 * 1024 * 1024:
                audio_bytes = compress_audio_for_upload(audio_bytes, target_bitrate="48k")
            
            # Upload to storage
            storage_path = f"{user_id}/{story_id}/chunked/{cache_key}/page_{page_num}.mp3"
            
            supabase.storage.from_('story-audio').upload(
                storage_path,
                audio_bytes,
                {"content-type": "audio/mpeg", "upsert": "true"}
            )
            
            logger.info(f"[CHUNKED-BG] Page {page_num} uploaded: {storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"[CHUNKED-BG] Page {page_num} failed: {str(e)}")
            return False
    
    # ==================== GENERATE PAGE 1 FIRST ====================
    logger.info("[CHUNKED-BG] === PRIORITY: Generating Page 1 ===")
    update_chunked_status([], [1], [])
    
    page1_success = await generate_page_audio(1, pages[0])
    
    if page1_success:
        pages_ready.append(1)
        update_chunked_status(pages_ready, list(range(2, total_pages + 1)), [])
        logger.info("[CHUNKED-BG] ✓ Page 1 READY - Frontend can start playback!")
    else:
        pages_failed.append(1)
        update_chunked_status([], [], pages_failed)
        logger.error("[CHUNKED-BG] ✗ Page 1 FAILED - Cannot continue")
        return
    
    # ==================== GENERATE REMAINING PAGES ====================
    for page_num in range(2, total_pages + 1):
        logger.info(f"[CHUNKED-BG] Generating Page {page_num}/{total_pages}")
        
        # Update status: this page is now generating
        remaining = [p for p in range(page_num, total_pages + 1)]
        update_chunked_status(pages_ready, remaining, pages_failed)
        
        page_success = await generate_page_audio(page_num, pages[page_num - 1])
        
        if page_success:
            pages_ready.append(page_num)
        else:
            pages_failed.append(page_num)
        
        # Small delay to prevent rate limiting
        await asyncio.sleep(0.5)
    
    # ==================== FINAL STATUS UPDATE ====================
    update_chunked_status(pages_ready, [], pages_failed)
    
    # Update usage counters
    if user_plan == 'free' and not is_tester:
        try:
            await increment_narration_usage(user_id)
        except:
            pass
    
    logger.info("=" * 60)
    logger.info("[CHUNKED-BG] ✓ COMPLETE!")
    logger.info(f"[CHUNKED-BG]   Pages ready: {pages_ready}")
    logger.info(f"[CHUNKED-BG]   Pages failed: {pages_failed}")
    logger.info("=" * 60)


@api_router.get("/narration/page-status", response_model=PageStatusResponse)
async def get_page_narration_status(
    story_id: str, 
    narrator: str = None,
    lang: str = None,
    user_id: str = Depends(get_current_user)
):
    """
    Get the status of chunked page-by-page narration.
    Used by frontend to know which pages are ready for playback.
    """
    try:
        story_result = supabase.table('stories').select(
            'pages, chunked_audio_status'
        ).eq('id', story_id).eq('user_id', user_id).execute()
        
        if not story_result.data:
            raise HTTPException(status_code=404, detail="Story not found")
        
        story = story_result.data[0]
        total_pages = len(story.get('pages', []))
        
        # Build cache key - use the actual language requested
        # (Language restriction removed - we now have language-specific narrators)
        narrator_id = narrator or DEFAULT_NARRATOR
        narration_lang = (lang or 'en').strip().lower()[:2]
        
        cache_key = f"{narrator_id}_{narration_lang}"
        logger.debug(f"[PAGE-STATUS] Cache key: {cache_key} (narrator={narrator_id}, lang={lang}→{narration_lang})")
        
        # Get chunked status
        chunked_status = story.get('chunked_audio_status') or {}
        if isinstance(chunked_status, str):
            try:
                chunked_status = json.loads(chunked_status)
            except:
                chunked_status = {}
        
        narrator_status = chunked_status.get(cache_key, {})
        pages_ready = narrator_status.get('pages_ready', [])
        pages_generating = narrator_status.get('pages_generating', [])
        pages_failed = narrator_status.get('pages_failed', [])
        
        return PageStatusResponse(
            storyId=story_id,
            totalPages=total_pages,
            pagesReady=pages_ready,
            pagesGenerating=pages_generating,
            pagesFailed=pages_failed,
            allReady=len(pages_ready) == total_pages and len(pages_failed) == 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PAGE-STATUS] Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get page status")


@api_router.get("/narration/page-audio")
async def get_page_audio_url(
    story_id: str,
    page: int,
    narrator: str = None,
    lang: str = None,
    user_id: str = Depends(get_current_user)
):
    """
    Get signed URL for a specific page's audio.
    Used by frontend to load individual page audio files.
    """
    try:
        # Build cache key - use the actual language requested
        # (Language restriction removed - we now have language-specific narrators)
        narrator_id = narrator or DEFAULT_NARRATOR
        narration_lang = (lang or 'en').strip().lower()[:2]
        
        cache_key = f"{narrator_id}_{narration_lang}"
        logger.debug(f"[PAGE-AUDIO] Cache key: {cache_key} (narrator={narrator_id}, lang={lang}→{narration_lang})")
        
        # Build storage path
        storage_path = f"{user_id}/{story_id}/chunked/{cache_key}/page_{page}.mp3"
        
        # Create signed URL (valid for 1 hour)
        signed_url_result = supabase.storage.from_('story-audio').create_signed_url(storage_path, 3600)
        
        if signed_url_result and signed_url_result.get('signedURL'):
            return {
                "page": page,
                "audioUrl": signed_url_result.get('signedURL'),
                "expiresIn": 3600
            }
        else:
            raise HTTPException(status_code=404, detail=f"Audio for page {page} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PAGE-AUDIO] Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get page audio")



async def get_narration_signed_url(story_id: str, lang: str = None, narrator: str = None, user_id: str = Depends(get_current_user)):
    """
    Get a signed URL for audio playback.
    Language-aware: specify ?lang=es to get Spanish audio.
    Narrator-aware: specify ?narrator=wise_owl to get that narrator's audio.
    The URL expires in 1 hour for security.
    """
    try:
        # Use default narrator if not specified
        narrator_id = narrator or DEFAULT_NARRATOR
        logger.info(f"[SIGNED-URL] Request: story_id={story_id}, lang={lang}, narrator={narrator_id}, user_id={user_id}")
        
        # Get story with per-language audio paths
        story_result = supabase.table('stories').select(
            'audio_url, audio_path_by_lang, audio_status_by_lang, user_id, language'
        ).eq('id', story_id).execute()
        
        if not story_result.data or len(story_result.data) == 0:
            raise HTTPException(status_code=404, detail="Story not found")
        
        story = story_result.data[0]
        
        # Verify ownership
        if story.get('user_id') != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Determine which language to use
        requested_lang = (lang or story.get('language', 'en')).lower()[:2]
        
        # Try to get narrator+language-specific audio path (new format)
        audio_path_by_lang = story.get('audio_path_by_lang') or {}
        if isinstance(audio_path_by_lang, str):
            try:
                audio_path_by_lang = json.loads(audio_path_by_lang)
            except:
                audio_path_by_lang = {}
        
        # New cache key format: {narrator}_{lang}
        cache_key = f"{narrator_id}_{requested_lang}"
        storage_path = audio_path_by_lang.get(cache_key)
        
        logger.info(f"[SIGNED-URL] Looking for cache_key: '{cache_key}' in audio_path_by_lang")
        
        # Fall back to legacy language-only key (for old audio) ONLY if narrator matches default
        if not storage_path and narrator_id == DEFAULT_NARRATOR:
            storage_path = audio_path_by_lang.get(requested_lang)
            if storage_path:
                logger.info(f"[SIGNED-URL] Using legacy lang-only path for default narrator: {storage_path}")
        
        # Fall back to legacy audio_url ONLY if:
        # 1. No narrator-specific audio found
        # 2. Requested narrator is the default narrator
        # 3. The legacy audio_url contains the default narrator's identifier
        # This prevents returning wrong narrator's audio
        if not storage_path and narrator_id == DEFAULT_NARRATOR:
            legacy_url = story.get('audio_url')
            if legacy_url:
                # Check if legacy URL is for the default narrator
                if DEFAULT_NARRATOR in str(legacy_url) or requested_lang in str(legacy_url).split('/')[-1]:
                    storage_path = legacy_url
                    logger.info(f"[SIGNED-URL] Using legacy audio_url for default narrator: {storage_path}")
                else:
                    logger.info("[SIGNED-URL] Legacy audio_url exists but not for default narrator, skipping")
        
        if not storage_path:
            # If no audio for this narrator, return 404 to prompt regeneration
            logger.info(f"[SIGNED-URL] No audio found for narrator='{narrator_id}', lang='{requested_lang}'. Regeneration needed.")
            raise HTTPException(
                status_code=404, 
                detail=f"No audio available for narrator '{narrator_id}'. Please generate narration with this narrator first."
            )
        
        # Handle legacy full URLs
        if storage_path.startswith('http'):
            if '/story-audio/' in storage_path:
                storage_path = storage_path.split('/story-audio/')[-1].split('?')[0]
                logger.info(f"[SIGNED-URL] Extracted path from URL: {storage_path}")
            else:
                raise HTTPException(status_code=500, detail="Invalid audio URL format")
        
        logger.info(f"[SIGNED-URL] Creating signed URL for: {storage_path}")
        
        # Create signed URL
        signed_url_result = supabase.storage.from_('story-audio').create_signed_url(storage_path, 3600)
        
        # Handle both camelCase and snake_case responses
        signed_url = None
        if isinstance(signed_url_result, dict):
            signed_url = signed_url_result.get('signedURL') or signed_url_result.get('signed_url') or signed_url_result.get('signedUrl')
        
        if not signed_url:
            logger.error(f"[SIGNED-URL] Failed to create URL: {signed_url_result}")
            raise HTTPException(status_code=500, detail="Failed to generate audio URL")
        
        logger.info(f"[SIGNED-URL] SUCCESS for {requested_lang}: {signed_url[:80]}...")
        
        return {
            "signedUrl": signed_url,
            "expiresIn": 3600,
            "language": requested_lang
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SIGNED-URL] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get signed URL: {str(e)}")

@api_router.post("/narration/prewarm")
async def prewarm_narration(request: NarrationRequest, background_tasks, user_id: str = Depends(get_current_user)):
    """
    Background prewarm narration (fire-and-forget).
    This endpoint returns immediately and generates audio in the background.
    """
    
    async def generate_in_background(story_id: str, narration_lang: str, voice_pref: str, uid: str):
        """Background task for narration generation"""
        try:
            # Reuse the main request logic
            req = NarrationRequest(
                storyId=story_id,
                narrationLanguageCode=narration_lang,
                voicePreference=voice_pref
            )
            # Note: In a real implementation, we'd call the generation logic directly
            # For now, the endpoint itself handles the logic
            logger.info(f"Prewarm narration completed for story {story_id}")
        except Exception as e:
            logger.warning(f"Prewarm narration failed (non-critical): {str(e)}")
    
    # Return immediately, generation happens in background
    logger.info(f"Queueing prewarm narration for story {request.storyId}")
    
    return {"status": "queued", "message": "Narration generation queued in background"}

# ================== Sleep Mode ==================

class SleepModeRequest(BaseModel):
    childName: str
    age: int
    storyCount: int = 3  # Default 3 stories
    theme: Optional[str] = None  # Optional, random if not specified
    storyLanguageCode: str = "en"
    narrationLanguageCode: Optional[str] = None

@api_router.post("/sleep-mode/start")
async def start_sleep_mode(request: SleepModeRequest, user_id: str = Depends(get_current_user)):
    """Start Sleep Mode - queue multiple stories for auto-play (Premium feature)"""
    try:
        # Check if user is premium
        user_result = supabase.table('users_profile').select('plan').eq('id', user_id).execute()
        
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        plan = user_result.data[0].get('plan', 'free')
        if plan not in ['premium', 'trial']:
            raise HTTPException(
                status_code=403,
                detail=json.dumps({"error": "PAYWALL", "reason": "sleep_mode_premium"})
            )
        
        # Limit story count to prevent abuse
        story_count = min(request.storyCount, 5)
        
        # Available themes and morals for random selection
        themes = ['dragons', 'space', 'animals', 'princess', 'adventure']
        morals = ['kindness', 'bravery', 'sharing', 'patience']
        
        # Generate stories
        import random
        generated_stories = []
        
        for i in range(story_count):
            selected_theme = request.theme if request.theme else random.choice(themes)
            selected_moral = random.choice(morals)
            
            # Create story request
            story_request = GenerateStoryRequest(
                userId=user_id,
                childName=request.childName,
                age=request.age,
                theme=selected_theme,
                moral=selected_moral,
                calmLevel='very_calm',  # Sleep mode always uses very calm
                durationMin=5,  # 5-minute stories for sleep mode
                storyLanguageCode=request.storyLanguageCode,
                narrationLanguageCode=request.narrationLanguageCode
            )
            
            try:
                # Generate story using existing function
                story_data = await generate_story_with_openai(story_request)
                
                # Save to database
                full_text = " ".join(story_data["pages"])
                story_record = {
                    "user_id": user_id,
                    "title": story_data["title"],
                    "child_name": request.childName,
                    "age": request.age,
                    "theme": selected_theme,
                    "moral": selected_moral,
                    "calm_level": "very_calm",
                    "duration_min": 5,
                    "language": request.storyLanguageCode,
                    "pages": story_data["pages"],
                    "full_text": full_text,
                    "audio_url": None,
                    "is_favorite": False,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                result = supabase.table('stories').insert(story_record).execute()
                
                if result.data and len(result.data) > 0:
                    saved_story = result.data[0]
                    generated_stories.append({
                        "storyId": saved_story['id'],
                        "title": story_data["title"],
                        "theme": selected_theme,
                        "pages": story_data["pages"],
                        "pageCount": len(story_data["pages"])
                    })
                    logger.info(f"Sleep mode story {i+1}/{story_count} generated for user {user_id}")
                    
            except Exception as story_error:
                logger.error(f"Failed to generate sleep mode story {i+1}: {str(story_error)}")
                # Continue with next story even if one fails
        
        # Update streak (only once for the session)
        if generated_stories:
            await update_streak(user_id)
        
        return {
            "success": True,
            "storiesGenerated": len(generated_stories),
            "storiesRequested": story_count,
            "stories": generated_stories,
            "message": f"Generated {len(generated_stories)} stories for sleep mode"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting sleep mode: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start sleep mode: {str(e)}")

# ================== PARENT VOICE ENDPOINTS ==================

@api_router.get("/parent-voice/profile")
async def get_parent_voice_profile(user_id: str = Depends(get_current_user)):
    """Get the user's parent voice profile status"""
    try:
        result = supabase.table('users_profile').select('parent_voice_id, parent_voice_status, parent_voice_created_at').eq('id', user_id).execute()
        
        if not result.data or len(result.data) == 0:
            return {"status": "none"}
        
        profile = result.data[0]
        voice_id = profile.get('parent_voice_id')
        status = profile.get('parent_voice_status', 'none')
        
        if not voice_id and status != 'processing':
            return {"status": "none"}
        
        return {
            "status": status,
            "voice_id": voice_id,
            "created_at": profile.get('parent_voice_created_at')
        }
        
    except Exception as e:
        logger.error(f"Error fetching parent voice profile: {str(e)}")
        return {"status": "none"}

@api_router.post("/parent-voice/upload")
async def upload_parent_voice(
    request: Request,
    user_id: str = Depends(get_current_user)
):
    """Upload voice samples and create a voice clone using ElevenLabs"""
    import httpx
    
    try:
        # Check if user has premium access
        profile = supabase.table('users_profile').select('plan, email').eq('id', user_id).execute()
        user_email = profile.data[0].get('email', '') if profile.data else ''
        
        # Check if tester
        is_tester = user_email.lower() in [e.lower() for e in TESTER_EMAILS]
        user_plan = profile.data[0].get('plan', 'free') if profile.data else 'free'
        
        if user_plan != 'premium' and not is_tester:
            raise HTTPException(status_code=403, detail="Parent Voice is a Premium feature")
        
        # Get the form data
        form = await request.form()
        audio_files = []
        
        for key in form:
            if key.startswith('audio_'):
                file = form[key]
                content = await file.read()
                audio_files.append({
                    'name': file.filename,
                    'content': content,
                    'content_type': file.content_type
                })
        
        if len(audio_files) < 5:
            raise HTTPException(status_code=400, detail="Please record all 5 voice samples")
        
        logger.info(f"[PARENT-VOICE] Received {len(audio_files)} audio files from user {user_id}")
        
        # Update status to processing
        supabase.table('users_profile').update({
            'parent_voice_status': 'processing'
        }).eq('id', user_id).execute()
        
        # Get ElevenLabs API key
        eleven_labs_key = os.environ.get('ELEVENLABS_API_KEY')
        if not eleven_labs_key or eleven_labs_key == 'placeholder-elevenlabs-key':
            # Mock response for development
            logger.warning("[PARENT-VOICE] No ElevenLabs API key, returning mock success")
            
            # Simulate processing delay
            import asyncio
            await asyncio.sleep(2)
            
            # Update with mock voice ID
            supabase.table('users_profile').update({
                'parent_voice_id': f'mock_voice_{user_id[:8]}',
                'parent_voice_status': 'ready',
                'parent_voice_created_at': datetime.utcnow().isoformat()
            }).eq('id', user_id).execute()
            
            return {"status": "ready", "message": "Voice profile created (mock mode)"}
        
        # Combine audio files for ElevenLabs voice cloning
        # ElevenLabs accepts multiple files
        files_for_upload = []
        for i, audio in enumerate(audio_files):
            files_for_upload.append(
                ('files', (audio['name'], audio['content'], audio['content_type']))
            )
        
        # Create voice clone using ElevenLabs API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers={
                    "xi-api-key": eleven_labs_key
                },
                data={
                    "name": f"Parent Voice - {user_id[:8]}",
                    "description": "Parent voice for PillowTales bedtime stories",
                    # Apply bedtime narration style labels
                    "labels": '{"style": "bedtime", "pace": "slow", "tone": "calm"}'
                },
                files=files_for_upload
            )
            
            if response.status_code == 200:
                result = response.json()
                voice_id = result.get('voice_id')
                
                logger.info(f"[PARENT-VOICE] Voice clone created: {voice_id}")
                
                # Update user profile with voice ID
                supabase.table('users_profile').update({
                    'parent_voice_id': voice_id,
                    'parent_voice_status': 'ready',
                    'parent_voice_created_at': datetime.utcnow().isoformat()
                }).eq('id', user_id).execute()
                
                return {"status": "ready", "voice_id": voice_id}
            else:
                logger.error(f"[PARENT-VOICE] ElevenLabs error: {response.status_code} - {response.text}")
                
                supabase.table('users_profile').update({
                    'parent_voice_status': 'error'
                }).eq('id', user_id).execute()
                
                raise HTTPException(status_code=500, detail="Failed to create voice clone")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PARENT-VOICE] Upload error: {str(e)}")
        
        # Update status to error
        try:
            supabase.table('users_profile').update({
                'parent_voice_status': 'error'
            }).eq('id', user_id).execute()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Failed to process voice samples: {str(e)}")


class ParentVoiceAudioData(BaseModel):
    promptId: int
    base64: str
    mimeType: str

class ParentVoiceBase64Request(BaseModel):
    audioData: List[ParentVoiceAudioData]

@api_router.post("/parent-voice/upload-base64")
async def upload_parent_voice_base64(
    request_data: ParentVoiceBase64Request,
    user_id: str = Depends(get_current_user)
):
    """Upload voice samples as base64 and create a voice clone using ElevenLabs.
    This endpoint is more reliable for React Native clients than multipart form-data.
    """
    import httpx
    import base64
    
    try:
        logger.info(f"[PARENT-VOICE] Received base64 upload request from user {user_id}")
        logger.info(f"[PARENT-VOICE] Number of audio samples: {len(request_data.audioData)}")
        
        # Check if user has premium access
        profile = supabase.table('users_profile').select('plan, email').eq('id', user_id).execute()
        user_email = profile.data[0].get('email', '') if profile.data else ''
        
        # Check if tester
        is_tester = user_email.lower() in [e.lower() for e in TESTER_EMAILS]
        user_plan = profile.data[0].get('plan', 'free') if profile.data else 'free'
        
        if user_plan != 'premium' and not is_tester:
            logger.warning(f"[PARENT-VOICE] User {user_id} is not premium (plan: {user_plan}, tester: {is_tester})")
            raise HTTPException(status_code=403, detail="Parent Voice is a Premium feature")
        
        if len(request_data.audioData) < 5:
            raise HTTPException(status_code=400, detail="Please record all 5 voice samples")
        
        # Decode base64 audio files
        audio_files = []
        for audio in request_data.audioData:
            try:
                content = base64.b64decode(audio.base64)
                audio_files.append({
                    'name': f'prompt_{audio.promptId}.m4a',
                    'content': content,
                    'content_type': audio.mimeType or 'audio/mp4'
                })
                logger.info(f"[PARENT-VOICE] Decoded audio {audio.promptId}, size: {len(content)} bytes")
            except Exception as e:
                logger.error(f"[PARENT-VOICE] Failed to decode audio {audio.promptId}: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid audio data for sample {audio.promptId}")
        
        logger.info(f"[PARENT-VOICE] Successfully decoded {len(audio_files)} audio files")
        
        # Update status to processing
        try:
            supabase.table('users_profile').update({
                'parent_voice_status': 'processing'
            }).eq('id', user_id).execute()
        except Exception as e:
            logger.warning(f"[PARENT-VOICE] Could not update status to processing: {e}")
        
        # Get ElevenLabs API key
        eleven_labs_key = os.environ.get('ELEVENLABS_API_KEY')
        if not eleven_labs_key or eleven_labs_key == 'placeholder-elevenlabs-key':
            # Mock response for development
            logger.warning("[PARENT-VOICE] No ElevenLabs API key, returning mock success")
            
            # Simulate processing delay
            import asyncio
            await asyncio.sleep(2)
            
            # Update with mock voice ID
            try:
                supabase.table('users_profile').update({
                    'parent_voice_id': f'mock_voice_{user_id[:8]}',
                    'parent_voice_status': 'ready',
                    'parent_voice_created_at': datetime.utcnow().isoformat()
                }).eq('id', user_id).execute()
            except Exception as e:
                logger.warning(f"[PARENT-VOICE] Could not update mock status: {e}")
            
            return {"status": "ready", "message": "Voice profile created (mock mode)"}
        
        # Prepare files for ElevenLabs voice cloning
        files_for_upload = []
        for audio in audio_files:
            files_for_upload.append(
                ('files', (audio['name'], audio['content'], audio['content_type']))
            )
        
        logger.info(f"[PARENT-VOICE] Sending {len(files_for_upload)} files to ElevenLabs...")
        
        # Create voice clone using ElevenLabs API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers={
                    "xi-api-key": eleven_labs_key
                },
                data={
                    "name": f"Parent Voice - {user_id[:8]}",
                    "description": "Parent voice for PillowTales bedtime stories",
                    "labels": '{"style": "bedtime", "pace": "slow", "tone": "calm"}'
                },
                files=files_for_upload
            )
            
            logger.info(f"[PARENT-VOICE] ElevenLabs response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                voice_id = result.get('voice_id')
                
                logger.info(f"[PARENT-VOICE] Voice clone created successfully: {voice_id}")
                
                # Update user profile with voice ID
                try:
                    supabase.table('users_profile').update({
                        'parent_voice_id': voice_id,
                        'parent_voice_status': 'ready',
                        'parent_voice_created_at': datetime.utcnow().isoformat()
                    }).eq('id', user_id).execute()
                except Exception as e:
                    logger.warning(f"[PARENT-VOICE] Could not update voice status: {e}")
                
                return {"status": "ready", "voice_id": voice_id}
            else:
                logger.error(f"[PARENT-VOICE] ElevenLabs error: {response.status_code} - {response.text}")
                
                try:
                    supabase.table('users_profile').update({
                        'parent_voice_status': 'error'
                    }).eq('id', user_id).execute()
                except:
                    pass
                
                raise HTTPException(status_code=500, detail=f"Failed to create voice clone: {response.text}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PARENT-VOICE] Upload error: {str(e)}")
        
        # Update status to error
        try:
            supabase.table('users_profile').update({
                'parent_voice_status': 'error'
            }).eq('id', user_id).execute()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Failed to process voice samples: {str(e)}")



@api_router.get("/parent-voice/sample")
async def get_parent_voice_sample(user_id: str = Depends(get_current_user)):
    """Generate a short voice sample using the user's cloned voice"""
    import httpx
    import base64
    
    # Sample text in multiple languages
    SAMPLE_TEXTS = {
        'en': "This is how your PillowTales voice will sound. Sweet dreams.",
        'es': "Así es como sonará tu voz de PillowTales. Dulces sueños.",
        'de': "So wird deine PillowTales-Stimme klingen. Süße Träume.",
        'fr': "Voici comment sonnera votre voix PillowTales. Faites de beaux rêves.",
        'it': "Ecco come suonerà la tua voce PillowTales. Sogni d'oro.",
        'pt': "É assim que sua voz do PillowTales vai soar. Bons sonhos.",
        'nl': "Zo zal je PillowTales-stem klinken. Slaap lekker.",
    }
    
    try:
        # Get voice ID and user's preferred language
        result = supabase.table('users_profile').select('parent_voice_id, parent_voice_status, preferred_language').eq('id', user_id).execute()
        
        if not result.data or not result.data[0].get('parent_voice_id'):
            raise HTTPException(status_code=404, detail="No voice profile found")
        
        voice_id = result.data[0]['parent_voice_id']
        status = result.data[0].get('parent_voice_status')
        preferred_lang = result.data[0].get('preferred_language', 'en') or 'en'
        
        if status != 'ready':
            raise HTTPException(status_code=400, detail="Voice is not ready yet")
        
        # For mock voice, return a mock response
        if voice_id.startswith('mock_'):
            return {"audio_url": None, "message": "Mock voice - no sample available"}
        
        eleven_labs_key = os.environ.get('ELEVENLABS_API_KEY')
        if not eleven_labs_key or eleven_labs_key == 'placeholder-elevenlabs-key':
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        # Get sample text in user's preferred language
        sample_text = SAMPLE_TEXTS.get(preferred_lang, SAMPLE_TEXTS['en'])
        
        logger.info(f"[PARENT-VOICE] Generating sample for voice_id: {voice_id}, language: {preferred_lang}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": eleven_labs_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": sample_text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.75,
                        "similarity_boost": 0.75
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"[PARENT-VOICE] Sample generation failed: {response.text}")
                raise HTTPException(status_code=500, detail="Failed to generate voice sample")
            
            # Upload to story-audio bucket (which exists)
            audio_data = response.content
            sample_path = f"{user_id}/voice_samples/parent_voice_preview.mp3"
            
            logger.info(f"[PARENT-VOICE] Uploading sample to: {sample_path}")
            
            try:
                # Try to remove existing sample first
                supabase.storage.from_('story-audio').remove([sample_path])
            except Exception as e:
                logger.info(f"[PARENT-VOICE] No existing sample to remove: {e}")
            
            # Upload the new sample
            upload_result = supabase.storage.from_('story-audio').upload(
                sample_path,
                audio_data,
                {"content-type": "audio/mpeg"}
            )
            logger.info(f"[PARENT-VOICE] Upload result: {upload_result}")
            
            # Get signed URL (valid for 1 hour)
            signed_url_result = supabase.storage.from_('story-audio').create_signed_url(sample_path, 3600)
            signed_url = signed_url_result.get('signedURL')
            
            logger.info(f"[PARENT-VOICE] Sample ready, signed URL generated")
            
            return {"audio_url": signed_url}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PARENT-VOICE] Sample error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate voice sample")


@api_router.delete("/parent-voice/profile")
async def delete_parent_voice_profile(user_id: str = Depends(get_current_user)):
    """Delete the user's parent voice profile"""
    import httpx
    
    try:
        # Get current voice ID
        result = supabase.table('users_profile').select('parent_voice_id').eq('id', user_id).execute()
        
        if result.data and result.data[0].get('parent_voice_id'):
            voice_id = result.data[0]['parent_voice_id']
            
            # Delete from ElevenLabs if it's a real voice ID
            if not voice_id.startswith('mock_'):
                eleven_labs_key = os.environ.get('ELEVENLABS_API_KEY')
                if eleven_labs_key and eleven_labs_key != 'placeholder-elevenlabs-key':
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            await client.delete(
                                f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                                headers={"xi-api-key": eleven_labs_key}
                            )
                            logger.info(f"[PARENT-VOICE] Deleted voice {voice_id} from ElevenLabs")
                    except Exception as e:
                        logger.warning(f"[PARENT-VOICE] Failed to delete from ElevenLabs: {e}")
        
        # Clear profile data
        supabase.table('users_profile').update({
            'parent_voice_id': None,
            'parent_voice_status': None,
            'parent_voice_created_at': None
        }).eq('id', user_id).execute()
        
        return {"success": True, "message": "Voice profile deleted"}
        
    except Exception as e:
        logger.error(f"[PARENT-VOICE] Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete voice profile")

# ================== DOWNLOADS ENDPOINT ==================
@api_router.get("/downloads/{filename}")
async def download_file(filename: str):
    """Serve downloadable files (screenshots, assets, etc.)"""
    downloads_dir = ROOT_DIR / "downloads"
    file_path = downloads_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security: only allow specific file types
    allowed_extensions = ['.zip', '.png', '.jpg', '.pdf']
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=403, detail="File type not allowed")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Startup event - check and install required dependencies"""
    logger.info("=" * 60)
    logger.info("PillowTales API Starting Up...")
    logger.info("=" * 60)
    
    # Check for ffmpeg/ffprobe (required for audio processing)
    await check_and_install_ffmpeg()
    
    logger.info("PillowTales API Ready!")
    logger.info("=" * 60)

async def check_and_install_ffmpeg():
    """Check if ffmpeg is available, install if missing"""
    import shutil
    import subprocess
    
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')
    
    if ffmpeg_path and ffprobe_path:
        logger.info(f"[STARTUP] ✅ ffmpeg found at: {ffmpeg_path}")
        logger.info(f"[STARTUP] ✅ ffprobe found at: {ffprobe_path}")
        return True
    
    logger.warning("[STARTUP] ⚠️ ffmpeg/ffprobe NOT FOUND - attempting to install...")
    
    try:
        # Install ffmpeg using apt-get
        result = subprocess.run(
            ['sudo', 'apt-get', 'update'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        result = subprocess.run(
            ['sudo', 'apt-get', 'install', '-y', 'ffmpeg'],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            # Verify installation
            ffmpeg_path = shutil.which('ffmpeg')
            ffprobe_path = shutil.which('ffprobe')
            
            if ffmpeg_path and ffprobe_path:
                logger.info(f"[STARTUP] ✅ ffmpeg installed successfully at: {ffmpeg_path}")
                logger.info(f"[STARTUP] ✅ ffprobe installed successfully at: {ffprobe_path}")
                return True
            else:
                logger.error("[STARTUP] ❌ ffmpeg installation completed but binaries not found in PATH")
                return False
        else:
            logger.error(f"[STARTUP] ❌ ffmpeg installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("[STARTUP] ❌ ffmpeg installation timed out")
        return False
    except Exception as e:
        logger.error(f"[STARTUP] ❌ ffmpeg installation error: {str(e)}")
        return False

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down PillowTales API")
