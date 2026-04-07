from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import ROOT_DIR, settings

router = APIRouter(tags=['system'])


@router.get('/')
async def root() -> dict:
    return {'message': f'{settings.app_name} is running'}


@router.get('/health')
async def health() -> dict:
    return {
        'status': 'ok',
        'app': settings.app_name,
        'gemini_configured': bool(settings.gemini_api_key),
        'supabase_configured': bool(settings.supabase_url and settings.supabase_service_role_key),
    }


@router.get('/download/{filename}')
async def download_file(filename: str) -> FileResponse:
    file_path = ROOT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail='File not found')
    allowed_extensions = {'.zip', '.png', '.jpg', '.jpeg', '.pdf'}
    if file_path.suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=403, detail='File type not allowed')
    return FileResponse(path=str(file_path), filename=Path(filename).name, media_type='application/octet-stream')
