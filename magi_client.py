"""
MAGI-1 Video Client.

CLI examples:
    python3 magi_client.py --prompt "Sunset over ocean"
    python3 magi_client.py --method direct --image docs/example.jpg
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                "png":"image/png","gif":"image/gif"}.get(Path(path).suffix[1:].lower(),"image/jpeg")
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            raise Exception(f"Failed to read image file {path}: {e}")

    # ------------ public API ------------
    def ping(self) -> Dict[str,Any]:
        try:
            response = self.sess.get(f"{self.base}/ping")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to ping server at {self.base}: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error during ping: {e}")

    def _openai(self, prompt, image_uri=None, model="magi-video-001"):
        content = [{"type":"text","text":prompt}]
        if image_uri: content.append({"type":"image_url","image_url":{"url":image_uri}})
        req = {"model":model,"messages":[{"role":"user","content":content}]}
        try:
            r = self.sess.post(f"{self.base}/v1/chat/completions", json=req, timeout=600)
            r.raise_for_status()
            data = r.json()
            if "choices" not in data or not data["choices"]:
                raise Exception("No choices returned from API")
            url = data["choices"][0]["message"]["content"]
            return {"success":True,"full_url": self.base+url if url.startswith("/") else url,
                    "response_id": data["id"], "model": data["model"]}
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected API response format: missing {e}")
        except Exception as e:
            raise Exception(f"Error in OpenAI API call: {e}")

    def _direct(self, prompt, image_uri=None, model_size=None, gpus=None):
        payload = {"prompt": prompt}
        if image_uri: payload["image_url"] = image_uri
        if model_size: payload["model_size"] = model_size
        if gpus: payload["gpus"] = gpus
        try:
            r = self.sess.post(f"{self.base}/generate", json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            if "download_url" not in data:
                raise Exception("No download_url in response")
            return {"success":True,
                    "full_download_url": self.base+data["download_url"],
                    **data}
        except requests.exceptions.RequestException as e:
            raise Exception(f"Direct API request failed: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected API response format: missing {e}")
        except Exception as e:
            raise Exception(f"Error in direct API call: {e}")

    # public convenience wrappers
    def generate(self, prompt, image_path=None, image_url=None,
                 method="openai", **extra):
        img_uri = None
        if image_path: img_uri = self._data_uri(image_path)
        elif image_url: img_uri = image_url
        
        if method == "openai":
            # Only pass parameters that _openai accepts
            openai_kwargs = {k: v for k, v in extra.items() if k in ['model']}
            return self._openai(prompt, image_uri=img_uri, **openai_kwargs)
        else:
            # _direct can accept all extra parameters
            return self._direct(prompt, image_uri=img_uri, **extra)

    def download(self, url, dest):
        url = self.base+url if url.startswith("/") else url
        try:
            with self.sess.get(url, stream=True) as r:
                r.raise_for_status()
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest,"wb") as f:
                    for chunk in r.iter_content(8192): 
                        f.write(chunk)
            return dest
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download from {url}: {e}")
        except OSError as e:
            raise Exception(f"Failed to write to {dest}: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error during download: {e}")

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

    try:
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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
