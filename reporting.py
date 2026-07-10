"""Presentation helpers for the ``GapReport``.

Pure formatting -- kept out of the graph so the notebook can render the same
report three ways (emoji summary, DataFrame, Markdown) without re-running it.
"""

from __future__ import annotations

from models import GapReport, GapSeverity

_SEVERITY_EMOJI = {
    GapSeverity.critical: "🔴",
    GapSeverity.important: "🟡",
    GapSeverity.nice: "🟢",
}
_SEVERITY_LABEL = {
    GapSeverity.critical: "critical",
    GapSeverity.important: "important",
    GapSeverity.nice: "nice-to-have",
}
_SEVERITY_ORDER = [GapSeverity.critical, GapSeverity.important, GapSeverity.nice]


def render_emoji_summary(report: GapReport) -> str:
    """Human-readable summary grouping gaps by severity with emoji markers."""
    lines = [
        f"Gap report for {report.target_role} ({report.level})",
        f"Postings analyzed: {report.postings_analyzed}  |  "
        f"Matched skills: {len(report.matched_skills)}  |  Gaps: {len(report.gaps)}",
        "",
    ]
    for severity in _SEVERITY_ORDER:
        group = [g for g in report.gaps if g.severity is severity]
        emoji = _SEVERITY_EMOJI[severity]
        lines.append(f"{emoji} {_SEVERITY_LABEL[severity].upper()} ({len(group)})")
        if not group:
            lines.append("   (none)")
        for gap in group:
            freq = f"{gap.requirement.frequency:.0%}"
            evidence = f"  [weak: {gap.resume_evidence}]" if gap.resume_evidence else ""
            lines.append(f"   - {gap.requirement.skill} ({freq} of postings){evidence}")
        lines.append("")
    return "\n".join(lines).rstrip()


def gaps_dataframe(report: GapReport):
    """Return a pandas DataFrame of gaps: [skill, category, frequency, severity, remediation]."""
    import pandas as pd  # lazy: only needed for this view

    rows = [
        {
            "skill": g.requirement.skill,
            "category": g.requirement.category,
            "frequency": g.requirement.frequency,
            "severity": _SEVERITY_LABEL[g.severity],
            "remediation": g.remediation,
        }
        for g in report.gaps
    ]
    df = pd.DataFrame(rows, columns=["skill", "category", "frequency", "severity", "remediation"])
    return df


def render_markdown(report: GapReport) -> str:
    """Readable Markdown summary for IPython.display.Markdown."""
    ts = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Skill Gap Report - {report.target_role} ({report.level})",
        "",
        f"*Generated {ts} · {report.postings_analyzed} postings analyzed*",
        "",
        f"**✅ Matched skills ({len(report.matched_skills)}):** "
        + (", ".join(report.matched_skills) if report.matched_skills else "_none_"),
        "",
        "## Prioritized gaps",
        "",
    ]
    for severity in _SEVERITY_ORDER:
        group = [g for g in report.gaps if g.severity is severity]
        if not group:
            continue
        lines.append(f"### {_SEVERITY_EMOJI[severity]} {_SEVERITY_LABEL[severity].title()}")
        lines.append("")
        for gap in group:
            freq = f"{gap.requirement.frequency:.0%}"
            lines.append(
                f"- **{gap.requirement.skill}** "
                f"({gap.requirement.category}, {freq} of postings) - {gap.remediation}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()
