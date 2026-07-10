"""Versioned prompt templates.

Prompts are data, not code -- they live here so they can be reviewed and bumped
independently. Each template carries a ``*_VERSION`` string that gets logged with
the extraction so runs are traceable.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Resume extraction                                                           #
# --------------------------------------------------------------------------- #

RESUME_EXTRACTION_VERSION = "resume-extract/v1"

RESUME_EXTRACTION_SYSTEM = """\
You are a precise resume parser. Extract a structured profile from the resume \
text. Respond with STRICT JSON only -- no prose, no markdown fences. The JSON \
must match this schema exactly:

{
  "skills": [string, ...],          // concrete, canonical skill names
  "years_experience": number|null,  // total professional years, null if unclear
  "education": [string, ...],       // degrees / institutions
  "credentials": [string, ...],     // certifications, courses
  "raw_summary": string             // 1-3 sentence summary of the candidate
}
"""

RESUME_EXTRACTION_USER = """\
Resume text:
---
{resume_text}
---
Return the JSON profile now.
"""


# --------------------------------------------------------------------------- #
# Requirement extraction (per posting)                                        #
# --------------------------------------------------------------------------- #

REQUIREMENT_EXTRACTION_VERSION = "requirement-extract/v1"

REQUIREMENT_EXTRACTION_SYSTEM = """\
You extract hiring requirements from a single job posting. Respond with STRICT \
JSON only -- no prose, no markdown fences. The JSON must match this schema:

{
  "requirements": [
    {
      "skill": string,                                  // one canonical skill
      "category": "technical"|"domain"|"soft"|"credential",
      "entry_level_expected": boolean                   // expected of an entry-level hire?
    },
    ...
  ]
}

Rules:
- One entry per distinct skill. Do not duplicate.
- "technical": tools/languages/frameworks. "domain": fields of knowledge \
(e.g. machine learning, statistics). "soft": interpersonal. "credential": \
degrees/certifications.
- Mark entry_level_expected=false for skills framed as a bonus/plus/desirable.
"""

REQUIREMENT_EXTRACTION_USER = """\
Job title: {title}
Company: {company}

Job description:
---
{description}
---
Return the JSON requirements now.
"""
