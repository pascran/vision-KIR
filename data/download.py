#!/usr/bin/env python3
"""Phase 1-a: Roboflow에서 로고/신발 객체검출 데이터셋 다운로드 (YOLO 포맷).

검증된 슬러그/버전 (2026-06 확인):
  - 로고: abner-colcq/fake-brand-logo-detetction v2 → 클래스 fake_logo/real_logo (821장, ★val/test split 없음)
  - 신발: ggdgdsgdg/original-or-fake-shoes v5 → 클래스 Authentic/Fake Nike AF1·AJ1 4종 (★grayscale 강제, test=1장)

API 키는 .env(ROBOFLOW_API_KEY)에서 로드 — 하드코딩 금지.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    load_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY 가 .env 에 없습니다.")

    from roboflow import Roboflow

    out = Path(__file__).resolve().parent / "datasets"
    out.mkdir(exist_ok=True)
    rf = Roboflow(api_key=api_key)

    print("[download] fake-brand-logo-detetction v2 (로고) ...")
    (rf.workspace("abner-colcq")
       .project("fake-brand-logo-detetction")
       .version(2)
       .download("yolov11", location=str(out / "logo"), overwrite=True))

    print("[download] original-or-fake-shoes v5 (신발) ...")
    (rf.workspace("ggdgdsgdg")
       .project("original-or-fake-shoes")
       .version(5)
       .download("yolov11", location=str(out / "shoes"), overwrite=True))

    print(f"[done] → {out}")
    for sub in ("logo", "shoes"):
        yaml = out / sub / "data.yaml"
        print(f"  {sub}: data.yaml={'있음' if yaml.exists() else '없음'}")


if __name__ == "__main__":
    main()
