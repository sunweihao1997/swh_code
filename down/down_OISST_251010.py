#!/usr/bin/env python3
import os, re, time, threading, xml.etree.ElementTree as ET
from urllib.parse import urljoin
from urllib.request import Request, urlopen

# ---------- 可配 ----------
OUT_DIR      = "/home/sun/wd_14/OISST/avhrr_thredds"  # 本地保存目录
MAX_WORKERS  = 6       # 并发数
RETRY        = 5       # 单文件重试
SLEEP        = 0.1     # 目录抓取间隔
YEAR_FILTER  = None    # 例: r'^(200[0-9]|201[0-9]|202[0-4])\d{2}$' 只下2000–2024
# -------------------------

# THREDDS 目录与下载前缀（NCEI固定套路）
CATALOG_BASE = "https://www.ncei.noaa.gov/thredds/catalog/OisstBase/NetCDF/V2.1/AVHRR/"
CATALOG_URL  = urljoin(CATALOG_BASE, "catalog.xml")
FILES_BASE   = "https://www.ncei.noaa.gov/thredds/fileServer/"

ACCEPT_EXT = (".nc", ".nc.gz")

UA = {"User-Agent":"oisst-bulk-downloader"}

def fetch(url, timeout=90):
    req = Request(url, headers=UA)
    with urlopen(req, timeout=timeout) as r:
        return r.read()

def parse_catalog(xml_bytes, base_url):
    """
    解析 THREDDS catalog.xml，返回：
      - 子目录（catalogRef -> href）
      - 文件数据集（dataset -> urlPath）
    """
    ns = {
        "cat": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0",
        "xlink": "http://www.w3.org/1999/xlink",
    }
    root = ET.fromstring(xml_bytes)

    # 子目录（catalogRef）
    subcats = []
    for cref in root.findall(".//cat:catalogRef", ns):
        href = cref.get("{http://www.w3.org/1999/xlink}href")
        if href:
            # href 通常是相对路径，如 "198109/catalog.xml"
            subcats.append(urljoin(base_url, href))

    # 数据集（dataset，含具体文件）
    files = []
    for ds in root.findall(".//cat:dataset", ns):
        urlPath = ds.get("urlPath")
        if urlPath and urlPath.lower().endswith(ACCEPT_EXT):
            files.append(urlPath)

    return subcats, files

def ensure_parent(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def download_file(url, out_path):
    for k in range(1, RETRY+1):
        try:
            ensure_parent(out_path)
            existing = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            req = Request(url, headers=UA)
            if existing:
                req.add_header("Range", f"bytes={existing}-")
            with urlopen(req, timeout=180) as r:
                status = getattr(r, "status", 200)
                mode = "ab" if existing and status == 206 else "wb"
                if existing and status == 200:
                    mode = "wb"  # 服务器不支持 Range，就重下
                with open(out_path, mode) as f:
                    while True:
                        chunk = r.read(1<<20)  # 1MB
                        if not chunk: break
                        f.write(chunk)
            return True
        except Exception as e:
            if k < RETRY:
                time.sleep(2*k)
            else:
                print(f"[FAIL] {url} -> {e}")
                return False

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 先抓根目录的 catalog.xml，得到子目录（每个月的子 catalog）
    try:
        root_xml = fetch(CATALOG_URL)
    except Exception as e:
        print(f"[ERROR] 获取根目录失败: {e}")
        return

    subcats, root_files = parse_catalog(root_xml, CATALOG_BASE)

    # 仅保留 YYYYMM/ 子目录（如果 YEAR_FILTER 指定，则应用）
    month_re = re.compile(r"/(\d{6})/catalog\.xml$")
    months = []
    for sc in subcats:
        m = month_re.search(sc)
        if not m:
            continue
        tag = m.group(1)  # YYYYMM
        if YEAR_FILTER and not re.match(YEAR_FILTER, tag):
            continue
        months.append(sc)
    months = sorted(set(months))
    print(f"将处理 {len(months)} 个月份目录")

    # 2) 收集所有文件（从每个月的 catalog.xml 里拿 dataset urlPath）
    all_urlpaths = []
    for cat in months:
        try:
            xml = fetch(cat)
            _, files = parse_catalog(xml, cat.rsplit('/', 1)[0] + '/')
            all_urlpaths.extend(files)
            time.sleep(SLEEP)
        except Exception as e:
            print(f"[WARN] 列目录失败 {cat}: {e}")

    # 3) 构造下载 URL 并下载
    #    THREDDS 的 fileServer 完整 URL = FILES_BASE + urlPath
    total = len(all_urlpaths)
    print(f"将下载 {total} 个文件…")
    idx = 0
    lock = threading.Lock()

    def worker():
        nonlocal idx
        while True:
            with lock:
                if idx >= total:
                    return
                urlPath = all_urlpaths[idx]
                idx += 1
                cur = idx
            url = urljoin(FILES_BASE, urlPath)
            # 本地保持从 AVHRR/ 之后的层级
            rel = urlPath.split("/AVHRR/", 1)[-1]
            out_path = os.path.join(OUT_DIR, rel)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"[{cur}/{total}] skip {rel}")
                continue
            ok = download_file(url, out_path)
            print(f"[{cur}/{total}] {'ok' if ok else 'err'} {rel}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(MAX_WORKERS)]
    for t in threads: t.start()
    for t in threads: t.join()
    print("完成。")

if __name__ == "__main__":
    main()
