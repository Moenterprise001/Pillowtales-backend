from __future__ import annotations

from typing import List, Optional

from fastapi import HTTPException
from supabase import Client


class StoryRepository:
    def __init__(self, client: Client):
        self.client = client

    def insert(self, record: dict) -> dict:
        result = self.client.table('stories').insert(record).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail='Failed to save story')
        return result.data[0]

    def list_for_user(self, user_id: str) -> List[dict]:
        result = self.client.table('stories').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        return result.data or []

    def get(self, story_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        query = self.client.table('stories').select('*').eq('id', story_id)
        if user_id:
            query = query.eq('user_id', user_id)
        result = query.limit(1).execute()
        return result.data[0] if result.data else None

    def update(self, story_id: str, user_id: str, values: dict) -> dict:
        result = self.client.table('stories').update(values).eq('id', story_id).eq('user_id', user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail='Story not found')
        return result.data[0]

    def delete(self, story_id: str, user_id: str) -> None:
        existing = self.get(story_id, user_id)
        if not existing:
            raise HTTPException(status_code=404, detail='Story not found')
        self.client.table('stories').delete().eq('id', story_id).eq('user_id', user_id).execute()

    def count_since(self, user_id: str, since_iso: str) -> int:
        result = self.client.table('stories').select('id', count='exact').eq('user_id', user_id).gte('created_at', since_iso).execute()
        return getattr(result, 'count', 0) or 0

    def count_all(self, user_id: str) -> int:
        result = self.client.table('stories').select('id', count='exact').eq('user_id', user_id).execute()
        return getattr(result, 'count', 0) or 0

    def count_narrations_since(self, user_id: str, since_iso: str) -> int:
        result = self.client.table('stories').select('id', count='exact').eq('user_id', user_id).not_.is_('audio_created_at', 'null').gte('audio_created_at', since_iso).execute()
        return getattr(result, 'count', 0) or 0

    def submit_feedback(self, record: dict) -> dict:
        result = (
            self.client.table('story_feedback')
            .upsert(record, on_conflict='story_id,user_id')
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=500, detail='Failed to save story feedback')
        return result.data[0]

    def get_feedback(self, story_id: str, user_id: str) -> Optional[dict]:
        result = (
            self.client.table('story_feedback')
            .select('*')
            .eq('story_id', story_id)
            .eq('user_id', user_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

