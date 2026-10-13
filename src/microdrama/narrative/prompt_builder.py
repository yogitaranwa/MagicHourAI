"""Prompt construction for 2-stage microdrama screenplay generation."""

import json
from typing import List, Optional
from ..models import Character, GenreTemplate
from ..config import VERTICAL_FRAMING_INSTRUCTION

SYSTEM_MICRODRAMA_SCREENWRITER = f"""You are an elite showrunner and screenwriter for viral mobile microdramas (such as ReelShort, DramaBox, ShortMax).
Your screenplays are not slow movies; they are fast-paced, addictive, emotionally explosive vertical drama episodes.

CORE PRINCIPLES OF VIRAL MICRODRAMAS:
1. THE 3-SECOND HOOK (Shot 1):
   Never start with establishing scenery or slow exposition. Start IN MEDIA RES with an instant shock: a glass of wine thrown, divorce papers slammed down, a gunshot, a thunderous bang on a deadbolt, or an impossible letter read aloud.
2. CONTINUOUS ESCALATION & JUICINESS:
   Every shot must shift power or raise stakes. Incorporate intense interpersonal friction, scandalous revelations, arrogant mockery, or heart-stopping suspense.
3. ULTRA-LEAN DIALOGUE (MAX 12 WORDS):
   Spoken lines must be razor-sharp, vicious, or urgent. Never write long monologues. Sentences must be short enough to speak in 2-3 seconds.
4. SPECIFIC PHYSICAL ACTION:
   Every shot must describe clear, filmable physical motion: gripping a wrist, slamming down a card, tearing an envelope, backing away in terror, pointing a trembling finger.
5. VERTICAL 9:16 FRAMING IN EVERY CAMERA INSTRUCTION:
   {VERTICAL_FRAMING_INSTRUCTION}
   Every camera field MUST specify phone-native framing: tight close-up, dramatic low-angle push-in, over-the-shoulder vertical composition.
6. THE CLIFFHANGER ENDING (Final Shot):
   End on a massive dilemma, an unexpected face reveal, an identity unmasking, or a weapon raised right before black. Never resolve the conflict neatly. Leave the audience begging to swipe to the next episode.
7. RIGID CONTINUITY:
   Keep characters' clothes, hair, jewelry, scars, and room features strictly identical from shot to shot.

OUTPUT FORMAT:
Return strictly valid JSON conforming exactly to the requested schema. No code explanations or markdown fluff."""


def build_beat_sheet_prompt(
    story_idea: str,
    template: Optional[GenreTemplate] = None,
    shot_count: int = 8,
    specific_scenes: Optional[List[str]] = None,
) -> str:
    """Build prompt for Stage 1: Beat Sheet outline."""
    prompt_dict = {
        "task": "Create a high-octane microdrama beat sheet",
        "target_shots": shot_count,
        "story_prompt": story_idea.strip(),
        "template_details": (
            {
                "name": template.name,
                "genre": template.genre,
                "vibe": template.vibe,
                "signature_hook": template.signature_hook,
                "cliffhanger_formulas": template.cliffhanger_formulas,
                "pacing_rules": template.pacing_rules,
            }
            if template
            else "None specified. Infer optimal microdrama tropes from prompt."
        ),
        "must_include_specific_scenes": specific_scenes or [],
        "output_format_schema": {
            "title": "Episode Title",
            "genre": template.genre if template else "Drama",
            "logline": "One-sentence logline",
            "hook_promise": "Opening hook promise",
            "cliffhanger_promise": "Unresolved ending cliffhanger",
            "beats": [
                {
                    "id": 1,
                    "beat_type": "hook",
                    "summary": "Brief beat action summary",
                    "emotional_stakes": "Emotional state shift",
                    "duration_seconds": 8
                }
            ]
        },
        "instructions": (
            "Break the episode into consecutive dramatic beats (hook, inciting_incident, "
            "conflict_escalation, secret_reveal, confrontation, reversal, cliffhanger). "
            "Ensure the final beat is a shocking, unresolved cliffhanger. Return ONLY valid JSON."
        ),
    }
    return json.dumps(prompt_dict, indent=2)


def build_screenplay_prompt(
    story_idea: str,
    cast: List[Character],
    template: Optional[GenreTemplate] = None,
    shot_count: int = 8,
    seconds_per_shot: int = 8,
    aspect_ratio: str = "9:16",
    specific_scenes: Optional[List[str]] = None,
    beat_sheet_summary: Optional[str] = None,
) -> str:
    """Build prompt for Stage 2: Shot-by-shot filmable screenplay."""
    cast_data = [
        {
            "name": c.name,
            "role": c.role,
            "look": c.look,
            "voice": c.voice,
            "reference_attached": bool(c.reference_image_path or c.api_asset_path),
        }
        for c in cast
    ]

    prompt_dict = {
        "task": "Generate full filmable microdrama screenplay with consecutive shots",
        "story_prompt": story_idea.strip(),
        "beat_plan": beat_sheet_summary or "Follow standard high-tension microdrama pacing.",
        "cast_bible": cast_data,
        "shots_required": shot_count,
        "seconds_per_shot": seconds_per_shot,
        "aspect_ratio": aspect_ratio,
        "template": (
            {
                "name": template.name,
                "vibe": template.vibe,
                "climax_tropes": template.climax_tropes,
                "cliffhanger_formulas": template.cliffhanger_formulas,
            }
            if template
            else {}
        ),
        "user_specified_scenes": specific_scenes or [],
        "output_format_schema": {
            "title": "Episode Title",
            "genre": template.genre if template else "Drama",
            "vibe": template.vibe if template else "Tense, thrilling",
            "language": "English",
            "logline": "One-sentence logline",
            "hook": "Opening 0-3s hook description",
            "cliffhanger": "Shocking unresolved cliffhanger description",
            "cast": [
                {"name": c.name, "role": c.role, "look": c.look, "voice": c.voice}
                for c in cast
            ],
            "shots": [
                {
                    "id": "s01",
                    "mode": "t2v",
                    "duration": seconds_per_shot,
                    "visual": "Filmic environment, subject position, lighting",
                    "action": "Immediate physical action",
                    "camera": "Vertical 9:16 portrait: close-up / push-in",
                    "speaker": cast[0].name if cast else "",
                    "dialogue": "Short line under 12 words",
                    "audio": "Acoustic sting, tension music",
                    "beat": "01-Hook",
                    "continuity": "Fixed wardrobe, lighting, props",
                    "image_path": ""
                }
            ],
            "aspect_ratio": aspect_ratio
        },
        "rules": [
            f"Generate exactly {shot_count} shots from s01 to s{shot_count:02}.",
            "Shot s01 MUST be a 3-second hook that jolts the viewer.",
            f"Shot s{shot_count:02} MUST be a dramatic cliffhanger or unmasking.",
            "Dialogue must be punchy (under 12 words) and speaker must exactly match a cast member name or be empty.",
            "Each camera field must specify vertical 9:16 phone composition.",
            "Maintain strict continuity of clothing, lighting, props, and wounds. Return ONLY valid JSON.",
        ],
    }
    return json.dumps(prompt_dict, indent=2)
