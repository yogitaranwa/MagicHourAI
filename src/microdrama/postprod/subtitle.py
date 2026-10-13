"""Microdrama subtitle generator and SRT formatting."""

from pathlib import Path
from typing import List, Optional
from ..models import Screenplay


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into standard SRT timestamp HH:MM:SS,mmm."""
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


class SubtitleGenerator:
    """Generates SRT and subtitle tracks for microdrama episodes."""

    @staticmethod
    def generate_srt(screenplay: Screenplay, out_file: Optional[Path] = None) -> str:
        """Create SRT file matching the timing of the shots."""
        subtitles: List[str] = []
        current_time = 0.0
        sub_index = 1

        for shot in screenplay.shots:
            shot_duration = float(shot.duration)
            if shot.dialogue and shot.dialogue.strip():
                speaker_tag = f"{shot.speaker}: " if shot.speaker else ""
                text = f"{speaker_tag}{shot.dialogue.strip()}"
                
                # Dialogue typically spans the first 60-80% of the shot duration
                start_str = format_srt_timestamp(current_time)
                end_str = format_srt_timestamp(current_time + min(shot_duration, max(2.5, len(text.split()) * 0.4)))

                block = f"{sub_index}\n{start_str} --> {end_str}\n{text}\n"
                subtitles.append(block)
                sub_index += 1

            current_time += shot_duration

        srt_content = "\n".join(subtitles)
        if out_file:
            out_file = Path(out_file)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(srt_content, encoding="utf-8")

        return srt_content
