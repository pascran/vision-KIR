"""순수 함수 단위 테스트 (GPU/데이터/torch 불필요 — CI에서 실행)."""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import dedup_split  # noqa: E402
import merge  # noqa: E402


def test_authenticity_of():
    assert merge.authenticity_of("fake air force") == "fake"
    assert merge.authenticity_of("ori air force") == "real"
    assert merge.authenticity_of("ori jordan 1") == "real"
    assert merge.authenticity_of("Fake brand logo") == "fake"
    assert merge.authenticity_of("Authentic Nike") == "real"
    assert merge.authenticity_of("sneaker") == "unknown"


def test_source_key():
    assert dedup_split.source_key("logo__train__x_jpg.rf.deadbeef.jpg") == "logo__train__x_jpg"
    assert dedup_split.source_key("plain.jpg") == "plain.jpg"


def test_ahash(tmp_path):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    Image.new("RGB", (32, 32), (123, 200, 50)).save(a)
    Image.new("RGB", (32, 32), (123, 200, 50)).save(b)
    assert dedup_split.ahash(a) == dedup_split.ahash(b)  # 동일 이미지 → 동일 해시
    half = Image.new("RGB", (32, 32), (0, 0, 0))
    for i in range(16):
        for j in range(32):
            half.putpixel((i, j), (255, 255, 255))
    half.save(c)
    assert dedup_split.ahash(c) != dedup_split.ahash(a)  # 다른 패턴 → 다른 해시
