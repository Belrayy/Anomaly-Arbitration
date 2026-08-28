from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"


def get_user_report_directory(user_id: int) -> Path:
    directory = REPORTS_DIR / str(user_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_report(
    user_id: int,
    filename: str,
    content: bytes
) -> Path:

    directory = get_user_report_directory(user_id)

    file_path = directory / filename

    file_path.write_bytes(content)

    return file_path