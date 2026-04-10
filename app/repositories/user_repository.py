from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from supabase import Client


class UserRepository:
    def __init__(self, client: Client):
        self.client = client
        self._wallet_path = Path('/tmp/pillowtales_parent_voice_wallets.json')

    def _read_wallets(self) -> dict:
        try:
            if self._wallet_path.exists():
                return json.loads(self._wallet_path.read_text() or '{}')
        except Exception:
            pass
        return {}

    def _write_wallets(self, data: dict) -> None:
        self._wallet_path.write_text(json.dumps(data))

    def _wallet_defaults(self, profile: Optional[dict] = None) -> dict:
        profile = profile or {}
        credits = profile.get('parent_voice_story_credits')
        if credits is None:
            credits = profile.get('parent_voice_credits', 0)
        intro_used = profile.get('parent_voice_intro_used')
        if intro_used is None:
            intro_used = False
        return {
            'credits': int(credits or 0),
            'intro_used': bool(intro_used),
        }

    def get_profile(self, user_id: str) -> Optional[dict]:
        result = self.client.table('users_profile').select('*').eq('id', user_id).limit(1).execute()
        return result.data[0] if result.data else None

    def get_by_id(self, user_id: str) -> dict | None:
        result = self.client.table('users_profile').select('*').eq('id', user_id).limit(1).execute()
        return result.data[0] if result.data else None

    def create_profile(self, profile: dict) -> dict:
        result = self.client.table('users_profile').insert(profile).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail='Failed to create user profile')
        return result.data[0]

    def update_profile(self, user_id: str, values: dict) -> dict:
        result = self.client.table('users_profile').update(values).eq('id', user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail='User not found')
        return result.data[0]

    def get_parent_voice_wallet(self, user_id: str) -> dict:
        profile = self.get_profile(user_id) or {}
        wallets = self._read_wallets()
        wallet = wallets.get(user_id)
        if wallet is None:
            wallet = self._wallet_defaults(profile)
            wallets[user_id] = wallet
            self._write_wallets(wallets)
        return {
            'credits': int(wallet.get('credits', 0)),
            'intro_used': bool(wallet.get('intro_used', False)),
            'profile': profile,
        }

    def save_parent_voice_wallet(self, user_id: str, *, credits: int, intro_used: bool) -> dict:
        wallets = self._read_wallets()
        wallet = {
            'credits': max(0, int(credits)),
            'intro_used': bool(intro_used),
        }
        wallets[user_id] = wallet
        self._write_wallets(wallets)
        try:
            self.client.table('users_profile').update({
                'parent_voice_story_credits': wallet['credits'],
                'parent_voice_intro_used': wallet['intro_used'],
            }).eq('id', user_id).execute()
        except Exception:
            # Fallback is the local wallet store when DB columns are not present yet.
            pass
        return wallet
