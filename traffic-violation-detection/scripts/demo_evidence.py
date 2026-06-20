"""
scripts/demo_evidence.py
=========================
End-to-End Evidence Generation Demo — Phase 6

Runs the FULL pipeline on synthetic traffic scenes:
  preprocess → detect → check violations → OCR plates → generate evidence

Saves annotated images + metadata JSON to evidence/ folder.

HOW TO RUN:
    python scripts\\demo_evidence.py

OUTPUT:
    evidence/
        <today's date>/
            <evidence_id>/
                annotated.jpg
                metadata.json
    samples/evidence_demo/
        summary.txt
"""

import sys
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.evidence_generator import EvidenceGenerator
from core.violation_detector  import ViolationDetector
from core.ocr                 import LicensePlateOCR
from config                   import EVIDENCE_DIR


DEMO_OUT = PROJECT_ROOT / "samples" / "evidence_demo"
DEMO_OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# SYNTHETIC SCENE BUILDERS
# ══════════════════════════════════════════════════════════════

def draw_car(img, x1, y1, x2, y2, color=(30, 80, 200)):
    h = y2-y1; w = x2-x1
    cv2.rectangle(img, (x1, y1+h//3), (x2, y2), color, -1)
    cv2.rectangle(img, (x1+w//5, y1), (x2-w//5, y1+h//3+4), color, -1)
    cv2.circle(img, (x1+w//5, y2+4), w//8, (15,15,15), -1)
    cv2.circle(img, (x2-w//5, y2+4), w//8, (15,15,15), -1)
    cv2.rectangle(img, (x1+4, y1+h//3-18), (x1+w//4+4, y1+h//3+6),
                  (200,220,255), -1)


def draw_motorcycle(img, x1, y1, x2, y2, color=(60,60,180)):
    h=y2-y1; w=x2-x1; cx=(x1+x2)//2
    cv2.ellipse(img,(cx,(y1+y2)//2),(w//2,h//3),0,0,360,color,-1)
    cv2.circle(img,(x1+w//5,y2+2),w//6,(15,15,15),-1)
    cv2.circle(img,(x2-w//5,y2+2),w//6,(15,15,15),-1)


def draw_person(img, x1, y1, x2, y2,
                skin=(100,150,200), has_helmet=False):
    h=y2-y1; cx=(x1+x2)//2
    hcolor=(30,30,30) if has_helmet else skin
    cv2.circle(img,(cx,y1+h//7),h//7,hcolor,-1)
    cv2.rectangle(img,(cx-h//12,y1+h//4),(cx+h//12,y1+2*h//3),skin,-1)


def draw_traffic_light(img, cx, cy, state="red"):
    cv2.rectangle(img,(cx-18,cy-45),(cx+18,cy+45),(20,20,20),-1)
    rc = (0,0,220) if state=="red"    else (15,15,15)
    yc = (0,200,230) if state=="yellow" else (15,15,15)
    gc = (0,180,0)  if state=="green"  else (15,15,15)
    cv2.circle(img,(cx,cy-25),12,rc,-1)
    cv2.circle(img,(cx,cy),   12,yc,-1)
    cv2.circle(img,(cx,cy+25),12,gc,-1)


def draw_plate(img, x1, y1, x2, y2, text="MH12AB1234"):
    cv2.rectangle(img,(x1,y1),(x2,y2),(245,245,240),-1)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)
    ph=y2-y1; scale=ph/52.0
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
    tx=x1+max(2,(x2-x1-tw)//2); ty=y2-max(4,(y2-y1-th)//2)
    cv2.putText(img,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,scale,(10,10,10),2)


def base_scene(h=480, w=640):
    img=np.zeros((h,w,3),dtype=np.uint8)
    img[:h//2]=(170,155,110); img[h//2:]=(72,73,80)
    for x in range(0,w,90):
        cv2.rectangle(img,(x,h//2+12),(x+50,h//2+20),(200,200,200),-1)
    return img


def scene_stop_line_violation():
    img=base_scene()
    cv2.line(img,(0,390),(640,390),(255,255,255),6)
    cv2.putText(img,"STOP",(275,386),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    draw_car(img,130,310,410,490,color=(30,70,180))
    draw_plate(img,210,430,380,462,"MH12AB1234")
    return img, "Stop Line Violation + Plate MH12AB1234"


def scene_triple_riding():
    img=base_scene()
    draw_motorcycle(img,220,300,400,460)
    for i,dx in enumerate([-35,0,35]):
        draw_person(img,240+dx,255,310+dx,430,has_helmet=False)
    draw_plate(img,250,430,370,460,"KA03EF9012")
    return img, "Triple Riding + Helmet Violation"


def scene_red_light():
    img=base_scene()
    draw_traffic_light(img,320,110,state="red")
    draw_car(img, 80,280,310,450,color=(30,70,180))
    draw_plate(img,130,400,280,432,"DL01CD5678")
    draw_car(img,360,290,580,460,color=(160,40,40))
    draw_plate(img,390,400,550,432,"TN04GH3456")
    return img, "Red Light Violation — Multiple vehicles"


def scene_illegal_parking():
    img=base_scene()
    # Normal car on road
    draw_car(img,150,300,420,450,color=(40,100,200))
    draw_plate(img,210,400,370,432,"UP14IJ7890")
    # Parked car at right edge, wide, near bottom
    draw_car(img,510,380,632,460,color=(80,80,80))
    draw_plate(img,520,418,626,442,"WB20KL2345")
    return img, "Illegal Parking at Road Edge"


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════

def run_scene(scene_fn, vd: ViolationDetector,
              ocr: LicensePlateOCR, gen: EvidenceGenerator,
              scene_name: str) -> dict:
    """Run full pipeline on one scene, return summary dict."""
    image, title = scene_fn()
    print(f"\n  Scene: {title}")

    # Violation detection (includes preprocessing + vehicle detection)
    viol_result = vd.detect_all(image)
    det_result  = viol_result.detection_result

    # License plate OCR
    vehicle_bboxes = [v.bbox for v in det_result.vehicles] \
                     if det_result else []
    plates = ocr.extract(image, vehicle_bboxes=vehicle_bboxes)

    # Assign plate numbers to violations
    plate_no = plates[0].plate_text_clean if plates else None
    for v in viol_result.violations:
        if v.plate_number is None:
            v.plate_number = plate_no

    # Generate evidence
    package = gen.generate(
        image            = image,
        image_path       = f"demo_{scene_name}.jpg",
        detection_result = det_result,
        violation_result = viol_result,
        plate_results    = plates,
    )

    # Copy annotated image to demo output folder for easy viewing
    if package.annotated_path:
        import shutil
        dest = DEMO_OUT / f"{scene_name}_evidence.jpg"
        shutil.copy2(package.annotated_path, dest)
        print(f"  Evidence saved: {dest.name}")

    print(f"  Violations : {len(viol_result.violations)}")
    for v in viol_result.violations:
        print(f"    [{v.violation_type}]  conf={v.confidence:.0%}"
              f"  fine=Rs.{v.fine_amount}")
    print(f"  Plates     : {[p.plate_text_clean for p in plates]}")
    print(f"  Evidence ID: {package.evidence_id}")
    print(f"  DB rows    : {package.db_record_ids}")
    print(f"  Time       : {viol_result.processing_ms:.0f}ms")

    return {
        "title":        title,
        "evidence_id":  package.evidence_id,
        "violations":   len(viol_result.violations),
        "total_fine":   viol_result.total_fines,
        "plates":       [p.plate_text_clean for p in plates],
        "annotated":    package.annotated_path,
        "metadata":     package.metadata_path,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n"+"="*58)
    print("  EVIDENCE GENERATION DEMO — Phase 6")
    print("="*58)

    print("\n[Step 1/4] Loading models ...")
    vd  = ViolationDetector(); vd.load_models()
    ocr = LicensePlateOCR();   ocr.load_models()
    gen = EvidenceGenerator(save_to_db=True)
    print("  All models ready\n")

    scenes = [
        (scene_stop_line_violation, "stop_line"),
        (scene_triple_riding,       "triple_riding"),
        (scene_red_light,           "red_light"),
        (scene_illegal_parking,     "illegal_parking"),
    ]

    print(f"[Step 2/4] Running {len(scenes)} scene(s) ...\n"+"-"*58)
    results = []
    for fn, name in scenes:
        try:
            r = run_scene(fn, vd, ocr, gen, name)
            results.append(r)
        except Exception as e:
            print(f"  ERROR in scene {name}: {e}")

    # DB stats
    print("\n[Step 3/4] Checking database ...")
    try:
        from core.database import SessionLocal, init_db, get_stats
        init_db()
        db = SessionLocal()
        stats = get_stats(db)
        db.close()
        print(f"  DB total violations : {stats['total_violations']}")
        print(f"  DB total images     : {stats['total_images']}")
        print(f"  DB total fines      : Rs.{stats['total_fines']}")
        print(f"  By type: {stats['by_type']}")
    except Exception as e:
        print(f"  WARN  DB check failed: {e}")

    # Write summary
    print("\n[Step 4/4] Writing summary ...")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "scenes":       results,
        "evidence_dir": str(EVIDENCE_DIR),
    }
    summary_path = DEMO_OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str),
                             encoding="utf-8")

    total_v = sum(r["violations"] for r in results)
    total_f = sum(r["total_fine"]  for r in results)

    print("\n"+"="*58)
    print("  EVIDENCE GENERATION DEMO COMPLETE")
    print("="*58)
    print(f"""
  {len(results)} scene(s) processed
  Total violations : {total_v}
  Total fines      : Rs.{total_f}
  Evidence folder  : {EVIDENCE_DIR}
  Demo output      : {DEMO_OUT}

  HOW TO VIEW:
  1. Open samples/evidence_demo/ — annotated JPEGs
  2. Open evidence/<date>/<id>/metadata.json — full JSON record
  3. Run: python core/database.py  — test DB queries

  NEXT STEP: Phase 7 — Backend API
      (tell Kiro: "start phase 7")
""")


if __name__ == "__main__":
    main()
