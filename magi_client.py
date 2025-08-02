"""
MAGI-1 Video Client.

CLI examples:
    python magi_client.py --prompt "Sunset over ocean"
    python magi_client.py --method direct --image docs/example.jpg
"""
import argparse, base64, os, sys, time
from pathlib import Path
from typing import Optional, Dict, Any

import requests

class MagiVideoClient:
    def __init__(self, base_url="http://localhost:8002"):
        self.base = base_url.rstrip("/")
        self.sess = requests.Session()

    # ------------ helpers ------------
    def _data_uri(self, path: str) -> str:
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                "png":"image/png","gif":"image/gif"}.get(Path(path).suffix[1:].lower(),"image/jpeg")
        b64 = base64.b64encode(open(path,"rb").read()).decode()
        return f"data:{mime};base64,{b64}"

    # ------------ public API ------------
    def ping(self) -> Dict[str,Any]:
        return self.sess.get(f"{self.base}/ping").json()

    def _openai(self, prompt, image_uri=None, model="magi-video-001"):
        content = [{"type":"text","text":prompt}]
        if image_uri: content.append({"type":"image_url","image_url":{"url":image_uri}})
        req = {"model":model,"messages":[{"role":"user","content":content}]}
        r = self.sess.post(f"{self.base}/v1/chat/completions", json=req, timeout=600)
        r.raise_for_status()
        data = r.json()
        url = data["choices"][0]["message"]["content"]
        return {"success":True,"full_url": self.base+url if url.startswith("/") else url,
                "response_id": data["id"], "model": data["model"]}

    def _direct(self, prompt, image_uri=None, model_size=None, gpus=None):
        payload = {"prompt": prompt}
        if image_uri: payload["image_url"] = image_uri
        if model_size: payload["model_size"] = model_size
        if gpus: payload["gpus"] = gpus
        r = self.sess.post(f"{self.base}/generate", json=payload, timeout=600)
        r.raise_for_status(); data = r.json()
        return {"success":True,
                "full_download_url": self.base+data["download_url"],
                **data}

    # public convenience wrappers
    def generate(self, prompt, image_path=None, image_url=None,
                 method="openai", **extra):
        img_uri = None
        if image_path: img_uri = self._data_uri(image_path)
        elif image_url: img_uri = image_url
        return (self._openai if method=="openai" else self._direct)(
            prompt, image_uri=img_uri, **extra)

    def download(self, url, dest):
        url = self.base+url if url.startswith("/") else url
        with self.sess.get(url, stream=True) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest,"wb") as f:
                for chunk in r.iter_content(8192): f.write(chunk)
        return dest

def main():
    p = argparse.ArgumentParser(description="MAGI-1 Video Client")
    p.add_argument("--base-url", default="http://localhost:8002")
    p.add_argument("--prompt", default="A kitten playing piano")
    p.add_argument("--image")
    p.add_argument("--image-url")
    p.add_argument("--output", default="/tmp/magi_video.mp4")
    p.add_argument("--method", choices=["openai","direct"], default="openai")
    p.add_argument("--model-size")
    p.add_argument("--gpus", type=int)
    args = p.parse_args()

    cli = MagiVideoClient(args.base_url)
    print("Ping:", cli.ping())

    t0 = time.time()
    res = cli.generate(args.prompt,
                       image_path=args.image,
                       image_url=args.image_url,
                       method=args.method,
                       model_size=args.model_size,
                       gpus=args.gpus)
    print("Generation took", time.time()-t0,"s")
    url = res["full_url"] if args.method=="openai" else res["full_download_url"]
    print("Video URL:", url)
    cli.download(url, args.output)
    print("Saved to:", args.output)

if __name__ == "__main__":
    main()
