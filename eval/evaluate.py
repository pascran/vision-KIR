#!/usr/bin/env python3
"""Phase 3: 검출 평가 — test split mAP@50/50-95, per-class P/R/AP, confusion → eval/results_<RUN>.md."""
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged" / "data.yaml"
RUN = os.environ.get("RUN", "detect_v1")
WEIGHTS = ROOT / "runs" / RUN / "weights" / "best.pt"


def main() -> None:
    model = YOLO(str(WEIGHTS))
    names = model.names
    m = model.val(data=str(DATA), split="test", project=str(ROOT / "runs"),
                  name=f"{RUN}_test", exist_ok=True, verbose=False)
    lines = [f"# 검출 평가 — {RUN} (test split)\n",
             f"- **mAP@50: {m.box.map50:.4f}**",
             f"- mAP@50-95: {m.box.map:.4f}",
             f"- mean P: {m.box.mp:.4f} / mean R: {m.box.mr:.4f}\n",
             "## per-class"]
    for i, ci in enumerate(m.box.ap_class_index):
        cid = int(ci)
        cname = names[cid] if isinstance(names, (list, dict)) else str(cid)
        # p/r/ap50/ap 모두 ap_class_index 위치 i로 정렬됨 — maps[cid](클래스ID 인덱싱)는 sparse 클래스에서 어긋남
        lines.append(f"- **{cname}**: P={m.box.p[i]:.3f} R={m.box.r[i]:.3f} "
                     f"AP50={m.box.ap50[i]:.3f} AP50-95={m.box.ap[i]:.3f}")
    out = ROOT / "eval" / f"results_{RUN}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[done] → {out}  (confusion matrix: runs/{RUN}_test/)")


if __name__ == "__main__":
    main()
