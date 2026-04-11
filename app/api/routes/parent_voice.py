from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.core.auth import get_current_user
import uuid

router = APIRouter(prefix="/api/parent-voice", tags=["parent-voice"])

@router.post("/create-profile")
async def create_parent_voice_profile(
    audio: UploadFile = File(...),
    user=Depends(get_current_user)
):
    try:
        voice_id = f"voice_{uuid.uuid4().hex[:8]}"

        # TODO: replace with ElevenLabs API
        # save voice_id to DB

        return {
            "status": "success",
            "voice_id": voice_id
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Failed to create voice profile")