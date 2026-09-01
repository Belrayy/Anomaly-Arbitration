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


@router.get("")
def get_my_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    reports = (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .order_by(Report.created_at.desc())
        .all()
    )

    return [
        {
            "id": report.id,
            "filename": report.filename,
            "model": report.model,
            "algorithm": report.algorithm,
            "created_at": report.created_at,
        }
        for report in reports
    ]


@router.get("/{report_id}")
def download_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    report = (
        db.query(Report)
        .filter(
            Report.id == report_id,
            Report.user_id == current_user.id
        )
        .first()
    )

    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    file_path = Path(report.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file no longer exists.")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=report.filename
    )