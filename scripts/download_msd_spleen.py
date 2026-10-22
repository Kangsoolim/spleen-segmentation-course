#!/usr/bin/env python3
"""Download and MD5-verify the official MSD Task09 Spleen archive."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
EXPECTED_MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"
BLOCK_SIZE = 1024 * 1024


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download_with_resume(url: str, archive: Path) -> None:
    part = archive.with_name(archive.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "MSD-Task09-Spleen-course-project/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接下载服务器：{exc}\n"
            f"请检查网络，或在浏览器中下载官方文件：{url}"
        ) from exc

    with response:
        status = getattr(response, "status", response.getcode())
        if offset and status == 206:
            mode = "ab"
            downloaded = offset
            content_range = response.headers.get("Content-Range", "")
            total = None
            if "/" in content_range:
                try:
                    total = int(content_range.rsplit("/", 1)[1])
                except ValueError:
                    total = None
        else:
            # The server may ignore Range. Restart rather than append a full response.
            mode = "wb"
            downloaded = 0
            offset = 0
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None

        last_report = 0.0
        start = time.time()
        with part.open(mode) as handle:
            while True:
                block = response.read(BLOCK_SIZE)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                now = time.time()
                if now - last_report >= 1.0:
                    if total:
                        pct = 100.0 * downloaded / total
                        print(
                            f"\r下载进度：{pct:6.2f}% "
                            f"({downloaded / (1024**3):.2f}/{total / (1024**3):.2f} GiB)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r已下载 {downloaded / (1024**3):.2f} GiB",
                            end="",
                            flush=True,
                        )
                    last_report = now
        print()
        elapsed = max(time.time() - start, 1e-6)
        print(f"本次下载用时 {elapsed / 60:.1f} 分钟。")
    part.replace(archive)


def safe_extract(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            posix = PurePosixPath(member.name)
            # Backslashes are rejected too, because they can act as path
            # separators on Windows even though PurePosixPath treats them as text.
            if posix.is_absolute() or ".." in posix.parts or "\\" in member.name:
                raise RuntimeError(f"归档包含不安全路径，已停止解压：{member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"归档包含链接条目，已停止解压：{member.name}")
            destination = (root / Path(*posix.parts)).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"归档路径越界，已停止解压：{member.name}")
        tar.extractall(path=root, members=members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="归档与解压目录（默认：data/raw）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="MD5 不正确时删除已有归档并重新下载；不会自动删除已解压目录",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "Task09_Spleen.tar"
    extracted = output_dir / "Task09_Spleen"

    if archive.exists():
        actual = md5_file(archive)
        if actual == EXPECTED_MD5:
            print(f"已存在归档且 MD5 正确：{actual}")
        elif args.force:
            print(f"现有归档 MD5 不匹配（{actual}），删除后重新下载。")
            archive.unlink()
        else:
            raise RuntimeError(
                f"现有归档 MD5 不匹配：{actual}\n"
                "请使用 --force 重新下载，或手动移走损坏的归档。"
            )

    if not archive.exists():
        print(f"开始下载：{URL}")
        download_with_resume(URL, archive)
        actual = md5_file(archive)
        if actual != EXPECTED_MD5:
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 校验失败：实际 {actual}，预期 {EXPECTED_MD5}。"
                "已删除校验失败的归档，请检查网络后重试。"
            )
        print(f"MD5 校验通过：{actual}")

    required = [
        extracted / "dataset.json",
        extracted / "imagesTr",
        extracted / "labelsTr",
        extracted / "imagesTs",
    ]
    if all(path.exists() for path in required):
        print(f"数据目录结构已存在，跳过重复解压：{extracted}")
    else:
        print(f"开始解压到：{output_dir}")
        safe_extract(archive, output_dir)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                "解压后未找到预期目录/文件：\n  " + "\n  ".join(missing)
            )
        print("解压完成，数据目录结构检查通过。")

    print(f"数据根目录：{extracted}")
    print("下一步运行：")
    print(
        "python scripts/audit_msd_spleen.py "
        "--data-root data/raw/Task09_Spleen --output-dir data --seed 42"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
