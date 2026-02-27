import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm


def get_box_filename(url: str) -> str:
    with requests.Session() as session:
        with session.get(url, allow_redirects=True, stream=True) as response:
            response.raise_for_status()
            content_disposition: str = response.headers.get("Content-Disposition", "")
            filename_match: Optional[re.Match[str]] = re.search(
                r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?',
                content_disposition,
            )
            return filename_match.group(1) if filename_match else response.url.rsplit("/", 1)[-1]


def is_valid_zip(zip_path: str | Path) -> bool:
    zip_path = Path(zip_path)
    if not zip_path.exists() or not zip_path.is_file():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            bad_member = zip_file.testzip()
            return bad_member is None
    except (zipfile.BadZipFile, OSError):
        return False


def is_zip_already_extracted(zip_path: str | Path, extract_dir: str | Path | None = None) -> bool:
    zip_path = Path(zip_path)
    extract_path = Path(extract_dir) if extract_dir is not None else zip_path.parent

    if not zip_path.exists():
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            members = [
                m for m in zip_file.infolist()
                if m.filename and not m.filename.startswith("__MACOSX/")
            ]
            if not members:
                return False

            for member in members:
                target_path = extract_path / member.filename
                if member.is_dir():
                    if not target_path.is_dir():
                        return False
                else:
                    if not target_path.is_file():
                        return False
            return True
    except (zipfile.BadZipFile, OSError):
        return False


def download_from_box(url: str, show_progress: bool = True, save_dir: str | Path = ".") -> Path:
    with requests.Session() as session:
        with session.get(url, allow_redirects=True, stream=True) as response:
            response.raise_for_status()

            content_disposition: str = response.headers.get("Content-Disposition", "")
            filename_match: Optional[re.Match[str]] = re.search(
                r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?',
                content_disposition,
            )
            filename: str = filename_match.group(1) if filename_match else response.url.rsplit("/", 1)[-1]

            total_size: Optional[int] = None
            content_length: str = response.headers.get("Content-Length", "")
            if content_length.isdigit():
                total_size = int(content_length)

            if total_size is None:
                try:
                    head_response: requests.Response = session.head(response.url, allow_redirects=True)
                    head_length: str = head_response.headers.get("Content-Length", "")
                    if head_length.isdigit():
                        total_size = int(head_length)
                except requests.RequestException:
                    pass

            output_dir: Path = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path: Path = output_dir / filename

            chunk_size: int = 1024 * 64

            if show_progress:
                with open(output_path, "wb") as output_file, tqdm(
                    total=total_size,
                    desc=filename,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            output_file.write(chunk)
                            progress_bar.update(len(chunk))
            else:
                with open(output_path, "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            output_file.write(chunk)

            return output_path


def unzip_file(zip_path: str | Path, extract_dir: str | Path | None = None, show_progress: bool = True) -> Path:
    zip_path = Path(zip_path)
    extract_path = Path(extract_dir) if extract_dir is not None else zip_path.parent
    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        members = zip_file.infolist()
        if show_progress:
            for member in tqdm(members, desc=f"Extracting {zip_path.name}", unit="file", dynamic_ncols=True):
                zip_file.extract(member, extract_path)
        else:
            zip_file.extractall(extract_path)

    return extract_path


def remove_macosx_artifacts(root_dir: str | Path) -> None:
    root = Path(root_dir)
    for macosx_dir in root.rglob("__MACOSX"):
        if macosx_dir.is_dir():
            shutil.rmtree(macosx_dir, ignore_errors=True)


def merge_dir_into(src_dir: str | Path, dst_dir: str | Path) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name

        if item.is_dir():
            if target.exists() and target.is_dir():
                merge_dir_into(item, target)
                try:
                    item.rmdir()
                except OSError:
                    pass
            else:
                shutil.move(str(item), str(target))
        else:
            if target.exists():
                item.unlink()
            else:
                shutil.move(str(item), str(target))


def flatten_single_nested_dir(root_dir: str | Path) -> None:
    root = Path(root_dir)
    if not root.exists():
        return

    entries = [p for p in root.iterdir() if p.name != "__MACOSX"]
    dirs = [p for p in entries if p.is_dir()]

    if len(dirs) == 1:
        nested = dirs[0]
        merge_dir_into(nested, root)
        try:
            nested.rmdir()
        except OSError:
            pass


def flatten_named_subdir(root_dir: str | Path, expected_name: str) -> None:
    root = Path(root_dir)
    if not root.exists():
        return

    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and p.name != "__MACOSX" and p.name.lower() == expected_name.lower()
    ]

    if len(candidates) == 1:
        nested = candidates[0]
        merge_dir_into(nested, root)
        try:
            nested.rmdir()
        except OSError:
            pass


def cleanup_extraction_artifacts(root_dir: str | Path, benchmark_name: str) -> None:
    remove_macosx_artifacts(root_dir)
    flatten_named_subdir(root_dir, benchmark_name)
    flatten_single_nested_dir(root_dir)
    remove_macosx_artifacts(root_dir)


if __name__ == "__main__":
    data_dir = Path("data")
    download_links = {
        "hocap": ["https://utdallas.box.com/shared/static/nb4aj2s67vl6pclk8gv7xle5a3frnpft.zip", "https://utdallas.box.com/shared/static/bbp5nrqw3bmk8dvpsbvmf16yr9i96z8l.txt", "https://utdallas.box.com/shared/static/aeq4pgda7v5h1t9js58px7n47ze2pc04.txt"],
        "hoi4d": ["https://utdallas.box.com/shared/static/ukjsyr41n8ewg3d0b3wykbk0vs87ie2k.zip", "https://utdallas.box.com/shared/static/m0bp0ay7ai7rrl8mzk0t7664gevkmt9l.txt", "https://utdallas.box.com/shared/static/lfko7bllk7mk5z3uygd2yfyrqtgnb9w1.txt"],
        "hot3d": ["https://utdallas.box.com/shared/static/h2cu3hqnpfm12ryxgf6pw15p92xffzgp.zip", "https://utdallas.box.com/shared/static/dfqrr2puxda4dhx3sss9p699pdxgbkkh.txt", "https://utdallas.box.com/shared/static/qcbuws69jbnwc1timoejrh7pbtjodnkg.txt"],
        "holoassist": ["https://utdallas.box.com/shared/static/5z9lzdi291d2o2pwswhkw1oq4d40pfle.zip", "https://utdallas.box.com/shared/static/jenkpoapzyacywefvefthsbgn8m7l06x.txt", "https://utdallas.box.com/shared/static/y810mcxbiru5i0ewipsnhib7b4uyzxti.txt"],
        "captaincook4d": ["https://utdallas.box.com/shared/static/ekjre07vnripjbkclcahb2h646n9tqju.zip", "https://utdallas.box.com/shared/static/97m6rsrocvsah832rlad5m739revgz0n.zip", "https://utdallas.box.com/shared/static/gt17or1yxvna8pev2fpu3encaf4c78ho.txt", "https://utdallas.box.com/shared/static/h5it87qxzj2aoeclxksjrq4yi5x9czn2.txt"],
    }

    data_dir.mkdir(parents=True, exist_ok=True)

    for benchmark_name, links in download_links.items():
        benchmark_root = data_dir / benchmark_name
        benchmark_root.mkdir(parents=True, exist_ok=True)

        for link in sorted(links):
            filename = get_box_filename(link)
            target_path = benchmark_root / filename
            is_zipped = filename.lower().endswith(".zip")

            if is_zipped:
                should_download_zip = True

                if target_path.exists():
                    if not is_valid_zip(target_path):
                        print(f"Removing incomplete/corrupt zip: {target_path}")
                        target_path.unlink()
                    else:
                        should_download_zip = False

                        if is_zip_already_extracted(target_path, extract_dir=benchmark_root):
                            print(f"Already extracted; removing zip: {target_path.name}")
                            cleanup_extraction_artifacts(benchmark_root, benchmark_name)
                            target_path.unlink()
                        else:
                            print(f"Zip already downloaded, extracting: {target_path.name}")
                            unzip_file(target_path, extract_dir=benchmark_root, show_progress=True)
                            cleanup_extraction_artifacts(benchmark_root, benchmark_name)
                            target_path.unlink()

                if should_download_zip:
                    download_path = download_from_box(link, show_progress=True, save_dir=benchmark_root)
                    unzip_file(download_path, extract_dir=benchmark_root, show_progress=True)
                    cleanup_extraction_artifacts(benchmark_root, benchmark_name)
                    if download_path.exists():
                        download_path.unlink()

            else:
                txt_target = benchmark_root / filename
                if txt_target.exists():
                    print(f"Skipping existing file: {txt_target}")
                else:
                    download_from_box(link, show_progress=False, save_dir=benchmark_root)
