"""
app/routers/stats.py
=====================
GET /stats  — aggregate violation statistics for the dashboard.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.database import get_db, get_stats
from app.models.schemas import StatsResponse, ViolationTypeStats
from config import VIOLATION_DISPLAY_NAMES

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get(
    "",
    response_model=StatsResponse,
    summary="Get aggregate violation statistics",
    description="""
Returns dashboard-level statistics:
- Total images processed
- Total violations detected
- Total fines collected (INR)
- Breakdown by violation type
- Most common violation type
- Average violations per image
""",
)
def get_statistics(db: Session = Depends(get_db)):
    raw = get_stats(db)

    # Build per-type breakdown
    by_type_list = []
    for v_type, count in raw.get("by_type", {}).items():
        # Compute total fines for this type from count × fine amount
        from config import VIOLATION_FINES
        fine_per = VIOLATION_FINES.get(v_type, 0)
        by_type_list.append(ViolationTypeStats(
            violation_type = v_type,
            display_name   = VIOLATION_DISPLAY_NAMES.get(v_type, v_type),
            count          = count,
            total_fines    = count * fine_per,
        ))

    # Sort by count descending
    by_type_list.sort(key=lambda x: x.count, reverse=True)

    most_common = by_type_list[0].violation_type if by_type_list else None

    total_images     = raw.get("total_images", 0)
    total_violations = raw.get("total_violations", 0)
    avg = round(total_violations / total_images, 2) if total_images > 0 else 0.0

    return StatsResponse(
        total_images              = total_images,
        total_violations          = total_violations,
        total_fines_inr           = raw.get("total_fines", 0),
        by_type                   = by_type_list,
        most_common               = most_common,
        avg_violations_per_image  = avg,
    )
