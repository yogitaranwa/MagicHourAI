"""High-level Orchestrator for the AI Micro Drama Generator."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from .models import Character, Screenplay, QualityScore
from .templates import get_template, list_templates
from .narrative import Screenwriter, MicrodramaJudge
from .client import MagicHourGenerator, AssetManager
from .postprod import StoryboardGenerator, SubtitleGenerator, EpisodeAssembler
from .config import OUTPUT_DIR, DEFAULT_MODEL, DEFAULT_RESOLUTION


class MicrodramaEngine:
    """End-to-end MicroDrama Generation Pipeline."""

    def __init__(
        self,
        magic_hour_api_key: Optional[str] = None,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-3.5-flash",
        llm_api_key: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.mock_mode = mock_mode
        self.screenwriter = Screenwriter(
            provider="mock" if mock_mode else llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        )
        self.generator = MagicHourGenerator(
            api_key=magic_hour_api_key,
            mock_mode=mock_mode,
        )
        self.asset_manager = AssetManager(magic_hour_client=self.generator.client)
        self.storyboard_gen = StoryboardGenerator()
        self.assembler = EpisodeAssembler()

    def generate_episode(
        self,
        story_prompt: str,
        template_id: Optional[str] = None,
        reference_images: Optional[Dict[str, str]] = None,
        specific_scenes: Optional[List[str]] = None,
        shot_count: int = 8,
        seconds_per_shot: int = 8,
        model: str = DEFAULT_MODEL,
        resolution: str = DEFAULT_RESOLUTION,
        render_video: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete microdrama generation pipeline."""
        print("[1/5] Initializing template & character assets...")
        template = get_template(template_id) if template_id else None
        
        # Build cast with reference images attached
        cast: List[Character] = []
        if template:
            cast = [c.model_copy() for c in template.default_cast]
        else:
            cast = [
                Character(
                    name="Protagonist",
                    role="Protagonist",
                    look="Striking face, sharp intense eyes, stylish outfit",
                    voice="Urgent, sharp",
                )
            ]

        # Attach user reference images to matching character names or protagonist
        if reference_images:
            for char_name, img_path in reference_images.items():
                matched = False
                for c in cast:
                    if c.name.lower() == char_name.lower():
                        c.reference_image_path = str(img_path)
                        matched = True
                        break
                if not matched and cast:
                    # Attach to primary character if only one provided
                    cast[0].reference_image_path = str(img_path)

        # Upload reference images to Magic Hour
        cast = self.asset_manager.prepare_cast_references(cast)

        print("[2/5] Writing high-stakes microdrama screenplay...")
        screenplay = self.screenwriter.generate_screenplay(
            story_idea=story_prompt,
            cast=cast,
            template=template,
            shot_count=shot_count,
            seconds_per_shot=seconds_per_shot,
            aspect_ratio="9:16",
            specific_scenes=specific_scenes,
        )

        print("[3/5] Evaluating script against DramaBench microdrama metrics...")
        score = MicrodramaJudge.evaluate(screenplay)
        print(f"   Quality Score: Hook={score.hook_score}/5 | Cliffhanger={score.cliffhanger_score}/5 | Dialogue={score.dialogue_score}/5")

        print("[4/5] Generating zero-cost visual storyboard cards...")
        card_paths = self.storyboard_gen.render_full_storyboard(screenplay)
        srt_path = OUTPUT_DIR / f"{screenplay.title.replace(' ', '_')}.srt"
        SubtitleGenerator.generate_srt(screenplay, srt_path)
        print(f"   Saved {len(card_paths)} storyboard cards and subtitle file: {srt_path.name}")

        result: Dict[str, Any] = {
            "screenplay": screenplay,
            "quality_score": score,
            "storyboard_cards": card_paths,
            "subtitles_srt": srt_path,
            "clips": [],
            "master_mp4": None,
        }

        if not render_video:
            print("[INFO] Storyboard ready. Video generation skipped per user request.")
            return result

        print("[5/5] Generating video clips via Magic Hour API...")
        manifest = self.generator.build_shot_manifest(
            screenplay=screenplay,
            model=model,
            resolution=resolution,
            audio=True,
        )

        clip_paths = []
        for i, item in enumerate(manifest, 1):
            shot_id = item["shot_id"]
            print(f"   Rendering shot {i}/{len(manifest)} ({shot_id})...")
            clip_path = self.generator.run_shot(item)
            clip_paths.append(clip_path)

        result["clips"] = clip_paths

        # Post-production assembly if clips are real video files
        if not self.mock_mode and self.assembler.ffmpeg_bin:
            print("[POST-PROD] Assembling vertical 9:16 master episode...")
            master_mp4 = self.assembler.assemble(
                screenplay=screenplay,
                clip_paths=clip_paths,
                out_filename=f"{screenplay.title.replace(' ', '_')}_master.mp4",
            )
            result["master_mp4"] = master_mp4
            print(f"[SUCCESS] Episode complete! Master output: {master_mp4}")
        else:
            print("[INFO] Clips saved. Run FFmpeg assembly to stitch final master MP4.")

        return result
