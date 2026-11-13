"""Minimal BigTIFF reader over HTTP Range for AntScan full-resolution stacks (8-bit, uncompressed).
Parses IFD0 and IFD1 to learn the per-page layout; if pages are laid out at a fixed stride, any page/row range can be
fetched directly without walking all IFDs.
Usage: python -I bigtiff_range.py probe <url>
       python -I bigtiff_range.py crop <url> <z0> <z1> <y0> <y1> <x0> <x1> <zstep> <out.npy>
"""
import sys, struct, time, json, urllib.request
import numpy as np

UA = {"User-Agent": "dsc-biology-research/0.1 (academic, non-commercial)"}
STATS = {"requests": 0, "bytes": 0}


def get(url, start, end):
    for k in range(6):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                assert r.status == 206, r.status
                b = r.read()
            STATS["requests"] += 1; STATS["bytes"] += len(b)
            return b
        except Exception:
            if k == 5: raise
            time.sleep(1 + 2 * k)


def parse_ifd(url, off, big=True):
    cw, ew, ow = (8, 20, 8) if big else (2, 12, 4)
    head = get(url, off, off + cw - 1)
    n = struct.unpack("<Q" if big else "<H", head)[0]
    body = get(url, off + cw, off + cw + n * ew + ow - 1)
    tags = {}
    for i in range(n):
        e = body[i * ew:(i + 1) * ew]
        if big:
            tag, typ, cnt = struct.unpack("<HHQ", e[:12]); val = e[12:20]
        else:
            tag, typ, cnt = struct.unpack("<HHI", e[:8]); val = e[8:12]
        size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 16: 8, 12: 8, 11: 4}.get(typ, 1)
        fmt = {1: "B", 3: "H", 4: "I", 5: "II", 16: "Q", 12: "d", 11: "f"}.get(typ)
        nbytes = cnt * size
        if nbytes <= ow:
            raw = val[:nbytes]
        else:
            ptr = struct.unpack("<Q" if big else "<I", val)[0]
            raw = get(url, ptr, ptr + nbytes - 1) if nbytes < (1 << 22) else b""
        vals = struct.unpack("<" + fmt * cnt, raw) if (fmt and len(raw) == nbytes) else (raw,)
        tags[tag] = vals
    nxt = struct.unpack("<Q" if big else "<I", body[n * ew:n * ew + ow])[0]
    return tags, nxt


def layout(url):
    h = get(url, 0, 15)
    big = h[:4] == b"II+\x00"
    assert big or h[:4] == b"II*\x00", h[:4]
    ifd0 = struct.unpack("<Q", h[8:16])[0] if big else struct.unpack("<I", h[4:8])[0]
    t0, ifd1 = parse_ifd(url, ifd0, big)
    t1, ifd2 = parse_ifd(url, ifd1, big)
    W, H = t0[256][0], t0[257][0]; bps = t0[258][0]; comp = t0[259][0]
    rps = t0.get(278, (H,))[0]; so0, so1 = t0[273], t1[273]; bc0 = t0[279]
    total = get(url, 0, 0) and None
    info = {"big": big, "W": W, "H": H, "bits": bps, "compression": comp, "rows_per_strip": rps, "strips": len(so0),
            "ifd0": ifd0, "ifd1": ifd1, "ifd2": ifd2, "ifd_stride": ifd1 - ifd0, "ifd_stride2": ifd2 - ifd1,
            "data0": so0[0], "data1": so1[0], "data_stride": so1[0] - so0[0], "page_bytes": sum(bc0),
            "contiguous_page": all(so0[i + 1] - so0[i] == bc0[i] for i in range(len(so0) - 1))}
    return info


def n_pages(url, info):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Range"].split("/")[-1])
    return size, (size - info["data0"]) // info["data_stride"] if info["data_stride"] > 0 else None


def data_offset(info, k):
    """Page k data offset. AntScan files: data(k+1) = align2(data(k)+page_bytes) + ifd_len, with a constant IFD length,
    so from page 1 on the stride is constant (= ifd_stride). Verified against IFD1/IFD2 in layout()."""
    if k == 0:
        return info["data0"]
    return info["data1"] + (k - 1) * info["ifd_stride"]


def verify_offset(url, info, k):
    """Parse page k's IFD at its predicted position and compare its strip offset with data_offset()."""
    ifd = data_offset(info, k) + info["page_bytes"]; ifd += ifd % 2
    tags, _ = parse_ifd(url, ifd, info["big"])
    return tags[273][0] == data_offset(info, k)


def crop(url, info, z0, z1, y0, y1, x0, x1, zstep=1):
    W, H = info["W"], info["H"]; out = []
    for z in range(z0, z1, zstep):
        start = data_offset(info, z) + y0 * W
        b = get(url, start, start + (y1 - y0) * W - 1)
        a = np.frombuffer(b, np.uint8).reshape(y1 - y0, W)[:, x0:x1]
        out.append(a.copy())
    return np.stack(out)


if __name__ == "__main__":
    cmd, url = sys.argv[1], sys.argv[2]
    t = time.time(); info = layout(url)
    size, n = n_pages(url, info); info["file_bytes"] = size; info["n_pages_est"] = n
    if cmd == "probe":
        print(json.dumps(info)); print("stats", STATS, "sec", round(time.time() - t, 1))
    elif cmd == "crop":
        z0, z1, y0, y1, x0, x1, zs = map(int, sys.argv[3:10])
        ok = verify_offset(url, info, z1 - 1) and verify_offset(url, info, info["n_pages_est"] - 1)
        print("offset_check", ok)
        assert ok, "page layout not regular; refusing to crop"
        v = crop(url, info, z0, z1, y0, y1, x0, x1, zs); np.save(sys.argv[10], v)
        print("crop", v.shape, "stats", STATS, "sec", round(time.time() - t, 1))
