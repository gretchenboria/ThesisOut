---
name: production-qa-audit
description: Use this skill to run end-to-end production QA on ThesisOut, verifying zero dummy/synthetic data, real 30-model cross-validation, persistent Star/Ribbon memory, agent registry health, and API/UI build integrity.
---

# Production QA & Statistical Integrity Audit Skill

## Overview
Verifies that the ThesisOut production workspace contains zero dummy or hardcoded datasets, all agents are registered with active policies, persistent memory (`saved_items`) works across sessions, and all automated tests pass.

## Execution Steps
1. Verify agent registry status via `./agent-cli list` and `./agent-cli verify-policies`.
2. Scan `backend/app/` to ensure no synthetic dataset generators (`_build_30_model_leaderboard` or `np.random` fallback tables) exist.
3. Run the pytest suite: `/Users/dr.gretchenboria/thesisout/backend/.venv/bin/pytest /Users/dr.gretchenboria/thesisout/backend/tests -v`.
4. Build the production frontend bundle: `cd /Users/dr.gretchenboria/thesisout/frontend && npm run build`.
