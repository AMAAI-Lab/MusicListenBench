"""Fetches only the drum stems of MUSDB18-HQ (Zenodo record 3338373, 22.7 GB zip) with HTTP range requests.
Reads the zip's central directory remotely, then the first SECONDS of every */drums.wav member (the rhythm task needs
6 s windows; the first 90 s of a song are plenty). Writes real_data/musdb18hq/drums/<split>__<track>.wav (44.1 kHz,
stereo, 16-bit). Usage: python -I -m musiclistenbench.other_sets.fetch_musdb_drums
MUSDB18-HQ is licensed for research use; do not redistribute the stems."""
import io, os, struct, sys, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf

URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
OUT = os.path.join(os.environ.get("REAL_DATA_DIR", "real_data"), "musdb18hq", "drums")
SECONDS = 90; SR = 44100; NEED = 44 + SECONDS * SR * 4 + 4096

class HTTPFile(io.RawIOBase):
    def __init__(self, url, block=4 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def _get(self, a, b):
        for k in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b - 1}"})
                return urllib.request.urlopen(req, timeout=120).read()
            except Exception as e:
                err = e
        raise err
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        if n > self.block:                      # large read: fetch directly
            d = self._get(self.pos, self.pos + n); self.pos += len(d); return d
        bi = self.pos // self.block
        if bi not in self.cache:
            self.cache = {bi: self._get(bi * self.block, min(self.size, (bi + 1) * self.block))}
        blk = self.cache[bi]; o = self.pos - bi * self.block; d = blk[o:o + n]
        if len(d) < n:                          # crosses into the next block
            self.pos += len(d); return d + self.read(n - len(d))
        self.pos += len(d); return d
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def fetch(name):
    split, track = name.split("/")[-3], name.split("/")[-2]
    out = f"{OUT}/{split}__{track.replace('/', '_')}.wav"
    if os.path.exists(out): return out, "exists"
    z = zipfile.ZipFile(HTTPFile(URL))
    with z.open(name) as f:
        raw = f.read(NEED)
    # parse the RIFF header up to the data chunk
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", name
    p = 12; fmt = None
    while p < len(raw) - 8:
        cid, sz = raw[p:p + 4], struct.unpack("<I", raw[p + 4:p + 8])[0]
        if cid == b"fmt ": fmt = struct.unpack("<HHIIHH", raw[p + 8:p + 24])
        if cid == b"data": p += 8; break
        p += 8 + sz + (sz & 1)
    tag, ch, sr, _, _, bits = fmt
    assert tag == 1 and bits == 16 and sr == SR, (name, fmt)
    pcm = np.frombuffer(raw[p:p + (len(raw) - p) // (2 * ch) * 2 * ch], "<i2").reshape(-1, ch)
    sf.write(out, pcm[:SECONDS * SR], SR, subtype="PCM_16")
    return out, f"{len(pcm) / SR:.1f}s"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = [i.filename for i in zipfile.ZipFile(HTTPFile(URL)).infolist() if i.filename.endswith("/drums.wav")]
    print(len(names), "drum stems", flush=True)
    with ThreadPoolExecutor(12) as ex:
        for out, msg in ex.map(fetch, names):
            print(os.path.basename(out), msg, flush=True)
    print("DONE", flush=True)
