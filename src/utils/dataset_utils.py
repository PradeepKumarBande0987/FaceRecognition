import hashlib
from pathlib import Path
import sys
import logging
import tarfile
import urllib
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def download_file(
    url: str,
    dest_path: str,
    expected_md5: Optional[str] = None,
) -> bool:
    """
    Download file with progress bar and optional MD5 verification.

    Args:
        url         : download URL
        dest_path   : local file path to save
        expected_md5: optional MD5 hash for verification

    Returns:
        True if successful
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded and verified
    if dest.exists():
        if expected_md5 and verify_md5(str(dest), expected_md5):
            logger.info(f"  ✅ Already downloaded: {dest.name}")
            return True
        elif not expected_md5:
            logger.info(f"  ✅ Already exists: {dest.name}")
            return True
        else:
            logger.warning(f"  ⚠️  MD5 mismatch, re-downloading: {dest.name}")

    logger.info(f"  📥 Downloading {dest.name}...")
    logger.info(f"     URL: {url}")

    try:
        urllib.request.urlretrieve(
            url,
            str(dest),
            DownloadProgressBar(dest.name),
        )
    except Exception as e:
        logger.error(f"  ❌ Download failed: {e}")
        return False

    # Verify MD5
    if expected_md5:
        if verify_md5(str(dest), expected_md5):
            logger.info(f"  ✅ MD5 verified: {dest.name}")
        else:
            logger.error(f"  ❌ MD5 mismatch: {dest.name}")
            return False

    return True

# ── Download Functions ────────────────────────────────────────────────────────

def verify_md5(filepath: str, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def extract_archive(archive_path: str, dest_dir: str) -> bool:
    """Extract a .tar.gz or .tgz archive into a destination directory."""
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        return True
    except Exception as exc:
        logger.error(f"  ❌ Extraction failed: {exc}")
        return False


# ── Progress Bar ──────────────────────────────────────────────────────────────

class DownloadProgressBar:
    """Simple CLI download progress bar."""

    def __init__(self, filename: str):
        self.filename = filename
        self.last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(int(downloaded / total_size * 100), 100)
            if pct != self.last_pct:
                bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
                sys.stdout.write(
                    f"\r  [{bar}] {pct:3d}%  "
                    f"{downloaded/1e6:.1f}/{total_size/1e6:.1f} MB"
                )
                sys.stdout.flush()
                self.last_pct = pct
        if downloaded >= total_size:
            print()