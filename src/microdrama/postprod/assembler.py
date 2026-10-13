"""Episode video assembly and FFmpeg post-production pipeline."""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from ..config import OUTPUT_DIR
from ..models import Shot, Screenplay
from .subtitle import SubtitleGenerator


def get_ffmpeg_binary() -> Optional[str]:
    """Discover FFmpeg executable from imageio-ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except ImportError:
        pass
    
    system_exe = shutil.which("ffmpeg")
    if system_exe:
        return system_exe
    
    return None


class EpisodeAssembler:
    """Normalizes and stitches video clips into a master 9:16 microdrama episode."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_bin = get_ffmpeg_binary()

    def run_ffmpeg(self, args: List[str]) -> None:
        """Run FFmpeg command."""
        if not self.ffmpeg_bin:
            raise RuntimeError(
                "FFmpeg executable not found. Install imageio-ffmpeg or install ffmpeg in system PATH."
            )
        cmd = [self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y"] + [str(a) for a in args]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {res.returncode}: {res.stderr}")

    @staticmethod
    def get_dimensions(aspect_ratio: str = "9:16") -> Tuple[int, int]:
        """Return (width, height) for target aspect ratio."""
        if aspect_ratio == "16:9":
            return (1280, 720)
        return (720, 1280)  # Standard high-quality 9:16 vertical mobile

    def assemble(
        self,
        screenplay: Screenplay,
        clip_paths: List[Path],
        out_filename: str = "episode_master.mp4",
        burn_subtitles: bool = False,
    ) -> Path:
        """Normalize, stitch, and attach subtitles to complete episode."""
        if len(clip_paths) != len(screenplay.shots):
            raise ValueError(f"Shot count mismatch: {len(screenplay.shots)} shots planned, {len(clip_paths)} clips provided.")

        for p in clip_paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"Missing shot clip: {p}")

        w, h = self.get_dimensions(screenplay.aspect_ratio)
        normalized_clips: List[Path] = []
        norm_dir = self.output_dir / "normalized_clips"
        norm_dir.mkdir(parents=True, exist_ok=True)

        for i, (shot, clip_path) in enumerate(zip(screenplay.shots, clip_paths)):
            norm_clip = norm_dir / f"norm_{shot.id}.mp4"
            
            # Normalize scale, aspect padding, 24fps, stereo audio
            vf_filter = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
            )
            
            args = [
                "-i", clip_path,
                "-vf", vf_filter,
                "-r", "24",
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-ar", "48000",
                "-ac", "2",
                norm_clip,
            ]
            self.run_ffmpeg(args)
            normalized_clips.append(norm_clip)

        # Concatenate using concat demuxer
        concat_list = self.output_dir / "concat_list.txt"
        concat_list.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in normalized_clips),
            encoding="utf-8",
        )

        master_out = self.output_dir / out_filename
        self.run_ffmpeg([
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            master_out,
        ])

        # Generate companion SRT file
        srt_path = master_out.with_suffix(".srt")
        SubtitleGenerator.generate_srt(screenplay, srt_path)

        return master_out
