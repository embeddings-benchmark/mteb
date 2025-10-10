from pathlib import Path

import requests
from tqdm.auto import tqdm


# https://stackoverflow.com/a/62113293
def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        Path(fname).open("wb") as file,
        tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
