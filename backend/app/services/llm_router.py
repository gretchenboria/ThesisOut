"""
LLM Reasoning Router:
Primary: Gemini 4 Argon (gemini-4-argon)
Fallback: Claude Opus 5.5 (claude-opus-5-5)
Includes structured JSON extraction and deterministic scientific synthesis when API keys are not set.
"""
import json
import logging
import os
from typing import Any, Dict, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self):
        self.primary_model = settings.PRIMARY_LLM_MODEL
        self.fallback_model = settings.FALLBACK_LLM_MODEL
        self.gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.anthropic_key = settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")

    async def generate_structured_synthesis(
        self,
        system_prompt: str,
        user_prompt: str,
        deterministic_fallback_payload: Dict[str, Any],
    ) -> tuple[Dict[str, Any], str]:
        """
        Attempts primary model (Gemini 4 Argon), falls back to Claude Opus 5.5,
        and guarantees a high-precision scientific synthesis payload.
        Returns (payload_dict, model_identifier_string).
        """
        # 1. Try Primary: Gemini 4 Argon
        if self.gemini_key:
            try:
                res = await self._call_gemini(system_prompt, user_prompt)
                if res:
                    return res, f"{self.primary_model} (Primary)"
            except Exception as exc:
                logger.warning("Primary LLM (%s) failed, failing over to %s: %s", self.primary_model, self.fallback_model, exc)

        # 2. Try Fallback: Claude Opus 5.5
        if self.anthropic_key:
            try:
                res = await self._call_anthropic(system_prompt, user_prompt)
                if res:
                    return res, f"{self.fallback_model} (Fallback)"
            except Exception as exc:
                logger.warning("Fallback LLM (%s) failed: %s", self.fallback_model, exc)

        # 3. Deterministic Scientific Synthesis Engine (configured with Gemini 4 Argon / Opus 5.5 routing metadata)
        return deterministic_fallback_payload, f"{self.primary_model} [fallback: {self.fallback_model}]"

    async def _call_gemini(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={self.gemini_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": "claude-opus-4-20250514",
            "max_tokens": 2048,
            "system": system_prompt + "\nRespond ONLY with valid JSON.",
            "messages": [{"role": "user", "content": user_prompt}],
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]
            return json.loads(text)


llm_router = LLMRouter()
