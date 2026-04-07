from __future__ import annotations

from typing import Optional

from fastapi import HTTPException
from supabase import Client


class UserRepository:
    def __init__(self, client: Client):
        self.client = client

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
