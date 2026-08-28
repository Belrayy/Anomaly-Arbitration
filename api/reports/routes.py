from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from database import get_db
from database.models import User, Report

router = APIRouter(
    prefix="/reports",
    tags=["Reports"]
)