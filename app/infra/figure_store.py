"""
FigureStore (disk only)
-----------------------
<STATIC_DIR>/<file_id>/<uuid>.png 에 저장 후
'/static/fig/<file_id>/<uuid>.png' URL 반환
"""
import os, uuid
from typing import List

STATIC_DIR = os.getenv("FIG_STATIC_DIR", "./static/fig")
URL_PREFIX = "/static/fig"


class FigureStore:
    def save_many(self, file_id: str, images: List[bytes]) -> List[str]:
        dir_path = os.path.join(STATIC_DIR, file_id)
        os.makedirs(dir_path, exist_ok=True)

        urls = []
        for img in images:
            fname = f"{uuid.uuid4().hex}.png"
            with open(os.path.join(dir_path, fname), "wb") as fp:
                fp.write(img)
            urls.append(f"{URL_PREFIX}/{file_id}/{fname}")
        return urls

