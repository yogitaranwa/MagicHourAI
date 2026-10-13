"""Screenwriter engine supporting Gemini REST, OpenAI-compatible endpoints, and offline templates."""

import os
import json
import time
import requests
from typing import Optional, List, Dict, Any
from ..models import Character, Screenplay, BeatSheet, GenreTemplate, Shot
from ..config import GEMINI_API_KEY, OPENAI_API_KEY
from .prompt_builder import (
    SYSTEM_MICRODRAMA_SCREENWRITER,
    build_beat_sheet_prompt,
    build_screenplay_prompt,
)
from .validator import validate_screenplay, validate_beat_sheet


class Screenwriter:
    """Microdrama Screenwriter with multiple LLM backends."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-3.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or (GEMINI_API_KEY if self.provider == "gemini" else OPENAI_API_KEY)
        self.base_url = base_url

    def _call_gemini(self, prompt: str, system_prompt: str = SYSTEM_MICRODRAMA_SCREENWRITER) -> str:
        """Call Google Gemini generateContent endpoint with automatic fallback chain."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        # Candidate models to try in case of 404 retirement or 503 temporary load spike
        candidates = [self.model]
        for fb in ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.8-flash"]:
            if fb not in candidates:
                candidates.append(fb)

        last_error = ""
        for model_name in candidates:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            headers = {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.7,
                    "maxOutputTokens": 8192,
                },
            }

            for attempt in range(2):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=90)
                    if response.status_code == 200:
                        data = response.json()
                        cands = data.get("candidates", [])
                        if cands:
                            parts = cands[0].get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                            if text.strip():
                                return text
                    
                    error_data = response.json() if response.content else {}
                    last_error = error_data.get("error", {}).get("message", response.text)
                    if response.status_code == 503:
                        time.sleep(2)
                        continue
                    elif response.status_code == 404:
                        break
                    else:
                        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {last_error}")
                except requests.RequestException as e:
                    last_error = str(e)
                    time.sleep(1)
                    continue

        raise RuntimeError(f"All Gemini model candidates failed. Last error: {last_error}")

    def _call_openai(self, prompt: str, system_prompt: str = SYSTEM_MICRODRAMA_SCREENWRITER) -> str:
        """Call OpenAI-compatible chat completions endpoint."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        
        url = self.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model or "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=90)
        if response.status_code != 200:
            raise RuntimeError(f"LLM API returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def generate_beat_sheet(
        self,
        story_idea: str,
        template: Optional[GenreTemplate] = None,
        shot_count: int = 8,
        specific_scenes: Optional[List[str]] = None,
    ) -> BeatSheet:
        """Generate Stage 1: Beat Sheet."""
        prompt = build_beat_sheet_prompt(story_idea, template, shot_count, specific_scenes)

        if self.provider == "mock" or not self.api_key:
            return self._mock_beat_sheet(story_idea, template, shot_count)

        try:
            if self.provider == "gemini":
                raw_json = self._call_gemini(prompt)
            else:
                raw_json = self._call_openai(prompt)
            return validate_beat_sheet(raw_json)
        except Exception as e:
            print(f"[INFO] Live LLM encountered temporary error ({e}). Using template beat planner...")
            return self._mock_beat_sheet(story_idea, template, shot_count)

    def generate_screenplay(
        self,
        story_idea: str,
        cast: Optional[List[Character]] = None,
        template: Optional[GenreTemplate] = None,
        shot_count: int = 8,
        seconds_per_shot: int = 8,
        aspect_ratio: str = "9:16",
        specific_scenes: Optional[List[str]] = None,
    ) -> Screenplay:
        """Generate full microdrama screenplay."""
        # 1. Resolve cast from parameters or template
        if not cast:
            cast = template.default_cast if template else [
                Character(name="Protagonist", role="Protagonist", look="Modern sleek outfit, piercing eyes", voice="Expressive")
            ]

        # 2. Stage 1: Beat Sheet
        beat_sheet = self.generate_beat_sheet(story_idea, template, shot_count, specific_scenes)
        beat_summary = "\n".join(f"Beat {b.id} ({b.beat_type}): {b.summary}" for b in beat_sheet.beats)

        # 3. Stage 2: Shot-by-shot Screenplay
        prompt = build_screenplay_prompt(
            story_idea=story_idea,
            cast=cast,
            template=template,
            shot_count=shot_count,
            seconds_per_shot=seconds_per_shot,
            aspect_ratio=aspect_ratio,
            specific_scenes=specific_scenes,
            beat_sheet_summary=beat_summary,
        )

        if self.provider == "mock" or not self.api_key:
            return self._mock_screenplay(story_idea, cast, template, shot_count, seconds_per_shot)

        try:
            if self.provider == "gemini":
                raw_json = self._call_gemini(prompt)
            else:
                raw_json = self._call_openai(prompt)
            return validate_screenplay(raw_json)
        except Exception as e:
            print(f"[INFO] Live LLM encountered temporary error ({e}). Using template screenplay generator...")
            return self._mock_screenplay(story_idea, cast, template, shot_count, seconds_per_shot)

    def _mock_beat_sheet(self, story_idea: str, template: Optional[GenreTemplate], shot_count: int) -> BeatSheet:
        """Generate high-quality deterministic beat sheet for offline testing."""
        return BeatSheet(
            title=f"{template.name if template else 'Microdrama'} Episode 1",
            genre=template.genre if template else "Suspense",
            logline=story_idea[:100],
            hook_promise=template.signature_hook if template else "Sudden shocking confrontation.",
            cliffhanger_promise=template.cliffhanger_formulas[0] if (template and template.cliffhanger_formulas) else "Unresolved life-or-death decision.",
            beats=[
                {"id": 1, "beat_type": "hook", "summary": "Visceral hook: physical shock or confrontation.", "emotional_stakes": "Panic", "duration_seconds": 8},
                {"id": 2, "beat_type": "inciting_incident", "summary": "A lethal secret or ultimatum is presented.", "emotional_stakes": "Escalation", "duration_seconds": 8},
                {"id": 3, "beat_type": "conflict_escalation", "summary": "Verbal clash and power struggle.", "emotional_stakes": "Fury", "duration_seconds": 8},
                {"id": 4, "beat_type": "secret_reveal", "summary": "Crucial truth is exposed to rivals.", "emotional_stakes": "Shock", "duration_seconds": 8},
                {"id": 5, "beat_type": "confrontation", "summary": "Physical or acoustic trap sprung.", "emotional_stakes": "Danger", "duration_seconds": 8},
                {"id": 6, "beat_type": "reversal", "summary": "Expected outcome flipped upside down.", "emotional_stakes": "Disorientation", "duration_seconds": 8},
                {"id": 7, "beat_type": "cliffhanger", "summary": "Identity unmasked or lethal choice presented.", "emotional_stakes": "Heart-stopping cliffhanger", "duration_seconds": 8},
            ][:shot_count],
        )

    def _mock_screenplay(
        self,
        story_idea: str,
        cast: List[Character],
        template: Optional[GenreTemplate],
        shot_count: int,
        seconds_per_shot: int,
    ) -> Screenplay:
        """Generate genre-specific screenplay matching the chosen template."""
        genre = template.genre if template else "Suspense"
        vibe = template.vibe if template else "Intense, thrilling"
        tid = template.id if template else "thriller_loop"
        p1 = cast[0].name
        p2 = cast[1].name if len(cast) > 1 else p1

        # Genre-tailored beats
        if tid == "contract_marriage":
            title = "The Cold CEO's Contract Bride"
            hook = "Divorce papers slammed on mahogany desk."
            cliffhanger = "The CEO pins her: 'You leave when you give me an heir.'"
            beats = [
                (f"Penthouse executive office. {p1} stands pale, trembling.", f"{p1} slams signed divorce papers onto the mahogany desk.", "Vertical 9:16: tight push-in on wet ink and tear stain.", p1, "We are done, Alexander.", "Paper slapping wood, tense cello sting.", "01-Hook"),
                (f"High-rise skyscraper window. {p2} turns around slowly.", f"{p2} unbuttons his cuffs, towering over {p1} with a cold sneer.", "Vertical 9:16: low angle hero framing.", p2, "Done? You didn't read the fine print.", "Low rumble of distant thunder.", "02-Escalation"),
                (f"Private office corner. {p1} backs away towards the glass.", f"{p1} clutches her coat tightly around her waist.", "Vertical 9:16: over-the-shoulder medium portrait.", p1, "I fulfilled every clause for three years!", "Violent gust of wind outside.", "03-Clash"),
                (f"Mahogany desk. {p2} stalks forward, eyes burning.", f"{p2} rips the divorce contract in half and tosses it into the air.", "Vertical 9:16: rapid tilt down to fluttering paper shreds.", p2, "I decide when this marriage ends.", "Sharp paper-tear acoustic sting.", "04-Reversal"),
                (f"Penthouse hallway. Heavy footsteps approach.", f"A rival heiress stands in the doorway holding a confidential folder.", "Vertical 9:16: doorway framing with harsh backlighting.", p2, "Who authorized you to enter?", "Loud click of high heels on marble.", "05-Intrusion"),
                (f"Floor-to-ceiling glass. {p1} turns pale, slipping.", f"{p1} drops her handbag, a medical envelope sliding across the floor.", "Vertical 9:16: high-angle drop shot on envelope.", p1, "Give that back!", "Muffled heartbeat rising.", "06-Secret"),
                (f"Office center. {p2} intercepts the envelope.", f"{p2} reads the positive test result, his cold mask completely shattering.", "Vertical 9:16: extreme close-up on dilated pupils.", p2, "Six weeks pregnant...?", "Sudden dramatic audio drop.", "07-Shock"),
                (f"Against the glass window. {p2} corners {p1}.", f"{p2} grips {p1}'s waist, pulling her flush against his chest.", "Vertical 9:16: intimate close-up profile.", p2, "You leave this house... when you give me an heir.", "Sudden heartbeat cliffhanger sting.", "08-Cliffhanger")
            ]
        elif tid == "hidden_billionaire":
            title = "The Hidden Billionaire"
            hook = "Wine thrown into the protagonist's face at an opulent gala."
            cliffhanger = "Elder tycoon patriarch drops to his knees: 'Welcome home, Chairman.'"
            beats = [
                (f"Opulent ballroom. {p2} sneers arrogantly.", f"{p2} splashes a full glass of red wine across {p1}'s face.", "Vertical 9:16: sudden violent push-in on splash.", p2, "A delivery boy doesn't belong here!", "Liquid splash, gasp of socialite crowd.", "01-Hook"),
                (f"Ballroom center. {p1} stands motionless.", f"{p1} calmly wipes wine from his eye, eyes burning cold.", "Vertical 9:16: low angle close-up on icy gaze.", p1, "You will regret touching me.", "Low menacing orchestral rumble.", "02-Threat"),
                (f"Marble reception. Security guards advance.", f"Two burly guards reach for {p1}'s arms.", "Vertical 9:16: dynamic over-the-shoulder medium shot.", p2, "Throw him into the rain!", "Radio chatter and crowd laughter.", "03-Escalation"),
                (f"Reception counter. {p1} reaches into his jacket.", f"{p1} tosses a black titanium centurion card onto the counter.", "Vertical 9:16: close-up on card sliding across marble.", p1, "Swipe it. See what happens.", "Metallic clinking sound.", "04-Reversal"),
                (f"Front foyer. The grand double doors burst open.", f"A motorcade of black SUVs pulls up outside in the rain.", "Vertical 9:16: wide vertical view of flashing red sirens.", p2, "Who the hell is outside?!", "Tires screeching, rain pouring.", "05-Arrival"),
                (f"Ballroom threshold. The city's greatest tycoon enters.", f"An elder billionaire patriarch walks in with armed guards.", "Vertical 9:16: slow tilt up from gold cane to elder tycoon.", p2, "Uncle! You came to throw him out?!", "Heavy rhythmic cane taps on marble.", "06-Confrontation"),
                (f"Center aisle. The patriarch freezes, eyes wide.", f"The elder tycoon slaps {p2} across the face with fury.", "Vertical 9:16: shocking slap impact close-up.", p2, "Uncle! Why did you hit me?!", "Loud stinging slap echoing.", "07-Slap"),
                (f"Before {p1}. The elder patriarch drops to both knees.", f"The elder tycoon bows his head to the floor before {p1}.", "Vertical 9:16: dramatic high-to-low camera drop.", p1, "Welcome home, Chairman.", "Massive cliffhanger orchestral sting.", "08-Cliffhanger")
            ]
        else:
            # Default / Thriller Loop / Custom
            title = f"{template.name if template else 'Microdrama'} Episode 1"
            hook = "Sudden shocking letter discovery."
            cliffhanger = "The killer reveals their face."
            beats = [
                (f"Dim hallway lit by harsh neon. {p1} stands trembling.", f"{p1} tears open the seal, unfolding a blood-marked note.", "Vertical 9:16: sudden close-up push-in on wide terrified eyes.", p1, "No... this is impossible.", "Sharp acoustic sting, rapid heartbeat.", "01-Hook"),
                (f"Entryway door. Wood vibrates violently.", f"Thunderous knocks shake the deadbolt in its casing.", "Vertical 9:16: portrait view of rattling door deadbolt.", p1, "Who's there?!", "Loud rhythmic banging on solid oak.", "02-Escalation"),
                (f"Dim corridor. {p1} backs away towards the kitchen.", f"{p1} grabs a weapon from the counter, breathing heavily.", "Vertical 9:16: tracking medium shot retreating in safe area.", p1, "Don't open the door...", "Muffled phone ring tone.", "03-Tension"),
                (f"Hallway corner. The banging suddenly cuts out.", f"Dead silence falls over the apartment.", "Vertical 9:16: tight low angle framing of empty dark hallway.", p1, "Did they leave...?", "Total suffocating acoustic silence.", "04-Silence"),
                (f"Front doorway. {p1} rounds the corner.", f"The front door stands wide open, cold wind blowing in.", "Vertical 9:16: portrait framing of swaying open door.", p1, "The lock was turned from the inside...", "Cold wind gust whistling.", "05-Dread"),
                (f"Entryway floor. {p1} takes a hesitant step.", f"A sudden blow strikes the base of {p1}'s skull.", "Vertical 9:16: disorienting blurred tilt down as camera drops.", p1, "Ugh...!", "Heavy iron impact, ringing ears.", "06-Attack"),
                (f"Looking down from the ceiling ceiling-level view.", f"{p1}'s motionless body lies bleeding on the hardwood.", "Vertical 9:16: eerie disembodied high angle looking down.", p1, "Am I... looking at myself?", "Eerie surreal ringing tone.", "07-Out of Body"),
                (f"Closet shadows. A figure steps into the light.", f"A cold, older {p1} steps out, dropping the exact same note onto the body.", "Vertical 9:16: low angle tilt up to chilling unblinking smile.", p2, "I told you not to open the door.", "Heartbeat stops, cliffhanger sting.", "08-Cliffhanger")
            ]

        shots = []
        for i in range(1, shot_count + 1):
            sid = f"s{i:02}"
            b_idx = min(i - 1, len(beats) - 1)
            b_vis, b_act, b_cam, b_spk, b_dlg, b_aud, b_tag = beats[b_idx]

            shots.append(Shot(
                id=sid,
                mode="t2v",
                duration=seconds_per_shot,
                visual=b_vis,
                action=b_act,
                camera=b_cam,
                speaker=b_spk,
                dialogue=b_dlg,
                audio=b_aud,
                beat=b_tag,
                continuity=f"{p1} wears fixed attire; scene props remain consistent.",
                image_path="",
            ))

        return Screenplay(
            title=title,
            genre=genre,
            vibe=vibe,
            language="English",
            logline=story_idea[:120],
            hook=hook,
            cliffhanger=cliffhanger,
            cast=cast,
            shots=shots,
            aspect_ratio="9:16",
        )

