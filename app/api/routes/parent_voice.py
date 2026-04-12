from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_current_user, get_user_repo
from app.repositories.user_repository import UserRepository

router = APIRouter(prefix='/parent-voice', tags=['parent-voice'])

_META_PATH = Path('/tmp/pillowtales_parent_voice_profiles.json')


def _read_meta() -> dict:
    try:
        if _META_PATH.exists():
            return json.loads(_META_PATH.read_text() or '{}')
    except Exception:
        pass
    return {}


def _write_meta(data: dict) -> None:
    _META_PATH.write_text(json.dumps(data))


def _get_user_meta(user_id: str) -> dict:
    return _read_meta().get(user_id, {})


def _set_user_meta(user_id: str, values: dict) -> dict:
    data = _read_meta()
    current = data.get(user_id, {})
    current.update(values)
    data[user_id] = current
    _write_meta(data)
    return current


def _clear_user_meta(user_id: str) -> None:
    data = _read_meta()
    if user_id in data:
        del data[user_id]
        _write_meta(data)


@router.get('/profile')
async def get_parent_voice_profile(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo)) -> dict:
    profile = user_repo.get_profile(user_id) or {}
    meta = _get_user_meta(user_id)
    status = profile.get('parent_voice_status') or meta.get('status') or 'none'
    voice_id = profile.get('parent_voice_id') or meta.get('voice_id')
    if not voice_id:
        status = 'none'
    return {'status': status, 'voice_id': voice_id, 'created_at': meta.get('created_at')}


@router.post('/upload')
async def upload_parent_voice(
    user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
    audio_1: UploadFile | None = File(None),
    audio_2: UploadFile | None = File(None),
    audio_3: UploadFile | None = File(None),
    audio_4: UploadFile | None = File(None),
    audio_5: UploadFile | None = File(None),
) -> dict:
    files = [f for f in [audio_1, audio_2, audio_3, audio_4, audio_5] if f is not None]
    if len(files) < 5:
        raise HTTPException(status_code=400, detail='Please upload all 5 voice samples.')

    api_key = os.getenv('ELEVENLABS_API_KEY', '').strip()
    if not api_key:
        raise HTTPException(status_code=500, detail='ELEVENLABS_API_KEY is not configured.')

    user_repo.update_profile(user_id, {'parent_voice_status': 'processing'})
    _set_user_meta(user_id, {'status': 'processing'})

    multipart = []
    first_sample_path = None
    sb = user_repo.client
    for idx, upload in enumerate(files, start=1):
        payload = await upload.read()
        filename = upload.filename or f'prompt_{idx}.m4a'
        content_type = upload.content_type or 'audio/mp4'
        multipart.append(('files', (filename, payload, content_type)))
        sample_path = f'parent-voice-samples/{user_id}/{filename}'
        try:
            sb.storage.from_('story-audio').upload(sample_path, payload, {'content-type': content_type, 'upsert': 'true'})
            if first_sample_path is None:
                first_sample_path = sample_path
        except Exception:
            pass

    multipart.extend([
        ('name', (None, f'PillowTales Parent Voice {user_id[:8]}')),
        ('description', (None, 'Parent Voice profile created from PillowTales voice samples.')),
        ('labels', (None, json.dumps({'source': 'pillowtales', 'user_id': user_id}))),
    ])

    try:
        async with httpx.AsyncClient(timeout=180.0) as client_http:
            response = await client_http.post(
                'https://api.elevenlabs.io/v1/voices/add',
                headers={'xi-api-key': api_key},
                files=multipart,
            )
    except Exception:
        user_repo.update_profile(user_id, {'parent_voice_status': 'error'})
        _set_user_meta(user_id, {'status': 'error'})
        raise HTTPException(status_code=502, detail='Failed to connect to ElevenLabs while creating the voice profile.')

    if response.status_code not in (200, 201):
        user_repo.update_profile(user_id, {'parent_voice_status': 'error'})
        _set_user_meta(user_id, {'status': 'error', 'error': response.text[:500]})
        raise HTTPException(status_code=502, detail=f'ElevenLabs could not create the voice profile: {response.text[:200]}')

    data = response.json()
    voice_id = data.get('voice_id') or data.get('voice', {}).get('voice_id')
    if not voice_id:
        user_repo.update_profile(user_id, {'parent_voice_status': 'error'})
        raise HTTPException(status_code=502, detail='ElevenLabs did not return a voice ID.')

    created_at = datetime.now(timezone.utc).isoformat()
    user_repo.update_profile(user_id, {'parent_voice_id': voice_id, 'parent_voice_status': 'ready'})
    _set_user_meta(user_id, {'status': 'ready', 'voice_id': voice_id, 'created_at': created_at, 'sample_path': first_sample_path})

    return {'status': 'ready', 'message': 'Parent Voice profile created successfully.'}


@router.get('/sample')
async def get_parent_voice_sample(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo)) -> dict:
    profile = user_repo.get_profile(user_id) or {}
    meta = _get_user_meta(user_id)
    if not (profile.get('parent_voice_id') or meta.get('voice_id')):
        raise HTTPException(status_code=404, detail='Parent Voice profile not found.')
    sample_path = meta.get('sample_path')
    if not sample_path:
        raise HTTPException(status_code=404, detail='Voice sample is not available yet.')
    try:
        signed = user_repo.client.storage.from_('story-audio').create_signed_url(sample_path, 3600)
        audio_url = signed.get('signedURL') or signed.get('signedUrl')
    except Exception:
        audio_url = None
    if not audio_url:
        raise HTTPException(status_code=404, detail='Voice sample is not available yet.')
    return {'audio_url': audio_url}


@router.delete('/profile')
async def delete_parent_voice_profile(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo)) -> dict:
    profile = user_repo.get_profile(user_id) or {}
    meta = _get_user_meta(user_id)
    voice_id = profile.get('parent_voice_id') or meta.get('voice_id')
    api_key = os.getenv('ELEVENLABS_API_KEY', '').strip()

    if voice_id and api_key:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client_http:
                await client_http.delete(
                    f'https://api.elevenlabs.io/v1/voices/{voice_id}',
                    headers={'xi-api-key': api_key},
                )
        except Exception:
            pass

    sample_path = meta.get('sample_path')
    if sample_path:
        try:
            user_repo.client.storage.from_('story-audio').remove([sample_path])
        except Exception:
            pass

    user_repo.update_profile(user_id, {'parent_voice_id': None, 'parent_voice_status': 'none'})
    _clear_user_meta(user_id)
    return {'status': 'deleted'}
