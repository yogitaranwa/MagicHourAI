"""Magic Hour video generation pipeline with reference support and polling."""

import os
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from magic_hour import Client
from ..config import MAGIC_HOUR_API_KEY, OUTPUT_DIR, VERTICAL_FRAMING_INSTRUCTION
from ..models import Screenplay, Shot, Character
from .journal import JobJournal
from .asset_manager import AssetManager


class MagicHourGenerator:
    """Manages video shot generation using Magic Hour API with character reference support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        journal: Optional[JobJournal] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or MAGIC_HOUR_API_KEY
        self.mock_mode = mock_mode or not bool(self.api_key)
        self.client = Client(token=self.api_key) if not self.mock_mode else None
        self.journal = journal or JobJournal()
        self.asset_manager = AssetManager(magic_hour_client=self.client)
        self.clips_dir = OUTPUT_DIR / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def build_shot_manifest(
        self,
        screenplay: Screenplay,
        model: str = "ltx-2.5",
        resolution: str = "480p",
        audio: bool = True,
    ) -> List[Dict[str, Any]]:
        """Construct Magic Hour API payloads for all shots in the screenplay."""
        # Ensure reference images are uploaded
        self.asset_manager.prepare_cast_references(screenplay.cast)

        cast_bible = "; ".join(f"{c.name}: {c.look}" for c in screenplay.cast)

        # Build reference items for text-to-video models supporting references
        reference_items = []
        for c in screenplay.cast:
            if c.api_asset_path:
                reference_items.append({"name": c.name, "file_path": c.api_asset_path})

        manifest = []
        for s in screenplay.shots:
            dialogue_str = (
                f"{s.speaker} says: \"{s.dialogue}\""
                if (s.speaker and s.dialogue)
                else "No spoken dialogue."
            )

            prompt = (
                f"{screenplay.genre} microdrama. {screenplay.vibe}. "
                f"Cast: {cast_bible}. {s.visual}. Action: {s.action}. "
                f"Camera: {s.camera}. Framing: {VERTICAL_FRAMING_INSTRUCTION} "
                f"Continuity: {s.continuity}. {dialogue_str}. "
                f"Sound: {s.audio}. No watermarks, subtitles or logos."
            )

            payload: Dict[str, Any] = {
                "name": f"{screenplay.title.replace(' ', '_')}_{s.id}",
                "style": {"prompt": prompt},
                "end_seconds": float(s.duration),
                "resolution": resolution,
                "model": model,
                "audio": audio,
            }

            if s.mode == "t2v":
                payload["aspect_ratio"] = screenplay.aspect_ratio
                if reference_items:
                    payload["references"] = reference_items
                endpoint = "text_to_video"
            else:
                # Image-to-video mode
                payload["assets"] = {"image_file_path": s.image_path}
                endpoint = "image_to_video"

            item = {
                "shot_id": s.id,
                "mode": s.mode,
                "endpoint": endpoint,
                "payload": payload,
            }
            manifest.append(item)

        return manifest

    def run_shot(self, manifest_item: Dict[str, Any], timeout: int = 1200) -> Path:
        """Submit and poll a single shot, returning the downloaded clip Path."""
        shot_id = manifest_item["shot_id"]
        payload = manifest_item["payload"]
        endpoint = manifest_item["endpoint"]
        tag = self.journal.compute_tag(payload)
        out_clip = self.clips_dir / f"{shot_id}_{tag[:10]}.mp4"

        # Check existing clip on disk
        if out_clip.exists():
            return out_clip

        if self.mock_mode:
            print(f"[MOCK] Simulated generation for {shot_id}")
            out_clip.write_text(f"Mock video for {shot_id}", encoding="utf-8")
            return out_clip

        job_record = self.journal.get_job(tag)

        # 1. Submit if not already submitted
        if not job_record or job_record.get("state") not in ("submitted", "complete"):
            self.journal.set_job(tag, {"state": "submitting", "shot_id": shot_id})
            try:
                api_target = getattr(self.client.v1, endpoint)
                response = api_target.create(**payload)
                job_id = response.id
                credits_charged = getattr(response, "credits_charged", None)
                job_record = {
                    "state": "submitted",
                    "id": job_id,
                    "shot_id": shot_id,
                    "credits_charged": credits_charged,
                }
                self.journal.set_job(tag, job_record)
            except Exception as e:
                self.journal.set_job(tag, {"state": "error", "error": str(e), "shot_id": shot_id})
                raise RuntimeError(f"Failed to submit {shot_id} to Magic Hour: {e}") from e

        # 2. Poll until complete
        job_id = job_record["id"]
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            status_res = self.client.v1.video_projects.get(id=job_id)
            state = status_res.status

            if state == "complete":
                downloads = status_res.downloads or []
                if not downloads:
                    raise RuntimeError(f"Shot {shot_id} marked complete but returned no download URLs.")
                download_url = downloads[0].url
                self._download_file(download_url, out_clip)
                job_record.update({
                    "state": "complete",
                    "url": download_url,
                    "final_credits": getattr(status_res, "credits_charged", None),
                })
                self.journal.set_job(tag, job_record)
                return out_clip

            if state in ("error", "canceled"):
                job_record["state"] = state
                self.journal.set_job(tag, job_record)
                raise RuntimeError(f"Shot {shot_id} processing {state} on Magic Hour.")

            time.sleep(4)

        raise TimeoutError(f"Shot {shot_id} timed out after {timeout} seconds.")

    def _download_file(self, url: str, target_path: Path) -> None:
        """Stream download to temporary file and atomically rename."""
        temp_path = target_path.with_suffix(".part")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        temp_path.replace(target_path)
