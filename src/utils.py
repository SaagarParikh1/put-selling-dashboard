import re

import pandas as pd


def generate_signal_summary(row) -> str:
    symbol = row.get("symbol", "This stock")
    label = row.get("label", "Unrated")
    confidence = row.get("confidence", "N/A")
    score = row.get("score", "N/A")
    reasons = row.get("reasons", "")

    quality_score = row.get("quality_score")
    entry_score = row.get("entry_score")
    risk_score = row.get("risk_score")
    rs_20 = row.get("rs_20")
    liquidity_ok = row.get("liquidity_ok")

    quality_text = ""
    if quality_score is not None:
        if quality_score >= 7:
            quality_text = "the underlying trend and regime quality look strong"
        elif quality_score >= 4:
            quality_text = "the underlying quality looks reasonably constructive"
        elif quality_score >= 1:
            quality_text = "the underlying quality is mixed"
        else:
            quality_text = "the underlying trend quality looks weak"

    entry_text = ""
    if entry_score is not None:
        if entry_score >= 5:
            entry_text = "entry timing is favorable near support"
        elif entry_score >= 2:
            entry_text = "entry timing is decent but not ideal"
        elif entry_score >= 0:
            entry_text = "entry timing is mixed"
        else:
            entry_text = "entry timing is weak and support is less reliable"

    risk_text = ""
    if risk_score is not None:
        if risk_score >= 0:
            risk_text = "downside risk currently looks relatively controlled"
        elif risk_score >= -3:
            risk_text = "some downside risk is present but not extreme"
        elif risk_score >= -6:
            risk_text = "downside risk is becoming more meaningful"
        else:
            risk_text = "breakdown risk looks elevated"

    rs_text = ""
    if rs_20 is not None:
        if rs_20 >= 5:
            rs_text = "relative strength versus SPY is clearly positive"
        elif rs_20 >= 1:
            rs_text = "relative strength versus SPY is modestly positive"
        elif rs_20 > -1:
            rs_text = "relative strength versus SPY is roughly neutral"
        elif rs_20 > -5:
            rs_text = "relative strength versus SPY is mildly negative"
        else:
            rs_text = "relative strength versus SPY is materially weak"

    liquidity_text = ""
    if liquidity_ok is True:
        liquidity_text = "liquidity looks acceptable for screening purposes"
    elif liquidity_ok is False:
        liquidity_text = "liquidity looks thinner, which can make put-selling execution less attractive"

    components = [x for x in [quality_text, entry_text, risk_text] if x]

    if components:
        if len(components) == 1:
            framework_sentence = components[0].capitalize() + "."
        elif len(components) == 2:
            framework_sentence = f"{components[0].capitalize()}, and {components[1]}."
        else:
            framework_sentence = (
                f"{components[0].capitalize()}, "
                + ", ".join(components[1:-1])
                + f", and {components[-1]}."
            )
    else:
        framework_sentence = (
            "The setup is being judged mainly on underlying quality, entry timing, and downside risk."
        )

    context_parts = [x for x in [rs_text, liquidity_text] if x]
    context_sentence = ""
    if context_parts:
        if len(context_parts) == 1:
            context_sentence = context_parts[0].capitalize() + "."
        elif len(context_parts) == 2:
            context_sentence = f"{context_parts[0].capitalize()}, and {context_parts[1]}."
        else:
            context_sentence = (
                f"{context_parts[0].capitalize()}, "
                + ", ".join(context_parts[1:-1])
                + f", and {context_parts[-1]}."
            )

    return (
        f"{symbol} is currently rated {label} with a confidence level of {confidence}%. "
        f"Composite score: {score}. "
        f"{framework_sentence} "
        f"{context_sentence} "
        f"Key factors: {reasons}."
    )


KEYWORD_PATTERNS = [
    r"200-day moving average",
    r"50-day moving average",
    r"20-day moving average",
    r"EMA 21",
    r"EMA 9",
    r"RSI",
    r"ADX",
    r"MACD",
    r"Chaikin Money Flow",
    r"SPY",
    r"primary support",
    r"secondary support",
    r"recommended entry",
    r"market regime",
    r"lower Bollinger Band",
    r"50-day average",
    r"200-day average",
    r"confirmed bounce",
    r"early bounce",
    r"entry zone",
    r"support test",
]


def emphasize_signal_terms(text: str) -> str:
    if not text:
        return ""

    for pattern in KEYWORD_PATTERNS:
        text = re.sub(
            pattern,
            lambda match: f"**{match.group(0)}**",
            text,
            flags=re.IGNORECASE,
        )
    return text


def colorize_signal_reason(text: str) -> str:
    if not text:
        return ""

    colored = text

    for pattern in KEYWORD_PATTERNS:
        colored = re.sub(
            pattern,
            lambda match: f"<span style='font-weight:800; color:#f8fafc;'>{match.group(0)}</span>",
            colored,
            flags=re.IGNORECASE,
        )

    replacements = [
        (r"\bexceptional\b", "#22c55e"),
        (r"\bstrong\b", "#22c55e"),
        (r"\bsolid\b", "#84cc16"),
        (r"\bconstructive\b", "#22c55e"),
        (r"\bpositive\b", "#22c55e"),
        (r"\bsupportive\b", "#22c55e"),
        (r"\bbullish\b", "#22c55e"),
        (r"\bhealthy\b", "#22c55e"),
        (r"\bcontrolled\b", "#22c55e"),
        (r"\bimproving\b", "#22c55e"),
        (r"\bneutral\b", "#facc15"),
        (r"\bmixed\b", "#facc15"),
        (r"\bwait\b", "#facc15"),
        (r"\bwatch\b", "#60a5fa"),
        (r"\bstalk\b", "#60a5fa"),
        (r"\bstabilization\b", "#60a5fa"),
        (r"\bcandidate\b", "#84cc16"),
        (r"\bcaution\b", "#fb923c"),
        (r"\bpressure\b", "#fb923c"),
        (r"\belevated\b", "#fb923c"),
        (r"\brisk\b", "#ef4444"),
        (r"\bweak\b", "#ef4444"),
        (r"\bbearish\b", "#ef4444"),
        (r"\bnegative\b", "#ef4444"),
        (r"\bbelow\b", "#ef4444"),
        (r"\bbroken\b", "#ef4444"),
    ]

    for pattern, color in replacements:
        colored = re.sub(
            pattern,
            lambda match: f"<span style='color:{color}; font-weight:700;'>{match.group(0)}</span>",
            colored,
            flags=re.IGNORECASE,
        )

    return colored


def normalize_boolish(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "ok", "1"}:
            return True
        if lowered in {"false", "no", "thin", "0"}:
            return False
        return None

    if isinstance(value, (bool, int, float)):
        return bool(value)

    try:
        return bool(value)
    except Exception:
        return None


def build_signal_takeaways(row) -> list[str]:
    label = row.get("label", "Unrated")
    quality_score = row.get("quality_score")
    entry_score = row.get("entry_score")
    risk_score = row.get("risk_score")
    market_regime = row.get("market_regime")
    rs_20 = row.get("rs_20")
    liquidity_ok = row.get("liquidity_ok")
    bounce_signal = row.get("bounce_signal")

    if label == "High Probability Put Sell":
        takeaways = ["Current signal: **High Probability Put Sell**. This is one of the cleaner trade-ready windows in the model."]
    elif label == "Put Sell Candidate":
        takeaways = ["Current signal: **Put Sell Candidate**. The setup is close enough to support to consider, but it is still one step below the highest-conviction tier."]
    elif label == "Stalk / Watchlist":
        takeaways = ["Current signal: **Stalk / Watchlist**. The stock may be worth owning, but the put-selling timing still needs to improve."]
    else:
        takeaways = [f"Current signal: **{label}**."]

    if quality_score is not None:
        if quality_score >= 8:
            takeaways.append("Underlying quality is strong enough that assignment would be more acceptable if the trade went against you.")
        elif quality_score >= 4:
            takeaways.append("Underlying quality is decent, but not strong enough on its own to justify forcing exposure without a better entry.")
        else:
            takeaways.append("Underlying quality is weak, so this is not the kind of assignment profile most put sellers should welcome.")

    if entry_score is not None:
        if entry_score >= 5:
            takeaways.append("Entry timing is favorable because price is sitting near a more attractive support-based area and looks tradeable now.")
        elif entry_score >= 2:
            takeaways.append("Entry timing is acceptable, but this still looks selective rather than automatic.")
        elif label == "Stalk / Watchlist":
            takeaways.append("Entry timing is not ready yet, which is why this is better treated as a stalk list name than a live put sale.")
        else:
            takeaways.append("Entry timing is not compelling yet, so patience is probably better than forcing a put sale before support proves itself.")

    if risk_score is not None:
        if risk_score >= 0:
            takeaways.append("Downside risk looks relatively controlled right now.")
        elif risk_score >= -3:
            takeaways.append("Some downside risk is present, but it is not fully broken.")
        else:
            takeaways.append("Downside risk is elevated, which makes fresh put exposure less attractive.")

    if bounce_signal:
        takeaways.append(f"Support behavior: **{bounce_signal}**.")

    context = []
    if market_regime:
        context.append(f"broader **market regime** is {market_regime.lower()}")
    if rs_20 is not None:
        if rs_20 >= 1:
            context.append("relative strength versus **SPY** is positive")
        elif rs_20 <= -1:
            context.append("relative strength versus **SPY** is weak")
    if liquidity_ok is True:
        context.append("screen liquidity looks acceptable")
    elif liquidity_ok is False:
        context.append("liquidity looks thin")

    if context:
        takeaways.append("Context: " + "; ".join(context) + ".")

    return takeaways


def build_action_suggestion(row) -> tuple[str, str]:
    label = row.get("label", "")
    entry_status = (row.get("entry_status") or "").lower()
    quality_score = row.get("quality_score")
    entry_score = row.get("entry_score")
    risk_score = row.get("risk_score")
    liquidity_ok = normalize_boolish(row.get("liquidity_ok"))
    bounce_signal = (row.get("bounce_signal") or "").lower()

    if liquidity_ok is False:
        return (
            "Action Suggestion: Pass unless you are comfortable with thin execution.",
            "Even if the chart looks decent, thin liquidity can make put pricing and exits less reliable. For systematic put selling, that is a real drawback.",
        )

    if label == "High Probability Put Sell":
        if "confirmed bounce" in bounce_signal:
            return (
                "Action Suggestion: This is one of the cleaner put-selling moments.",
                "The stock has pulled into support and is showing a more convincing bounce, which is usually the kind of timing put sellers want to see when they are willing to own the stock on assignment.",
            )
        if "early bounce" in bounce_signal:
            return (
                "Action Suggestion: A starter-quality bounce is forming.",
                "Support is beginning to hold and price is trying to turn up. This is better than catching a falling knife, but it is still earlier-stage confirmation.",
            )
        if "in entry zone" in entry_status:
            return (
                "Action Suggestion: Consider a disciplined put sale now.",
                "The setup is constructive and price is already close to the preferred support area. If you sell a put here, keep strike selection disciplined around support and only use a strike where you would accept assignment.",
            )
        if "watch for stabilization" in entry_status:
            return (
                "Action Suggestion: Wait for confirmation near support.",
                "The setup is close, but price has not fully settled yet. Let support prove itself before leaning in.",
            )
        return (
            "Action Suggestion: Conditions are favorable, but execution discipline still matters.",
            "This is one of the cleaner put-selling windows in the model, but strike placement and assignment comfort still matter more than the label itself.",
        )

    if label == "Put Sell Candidate":
        if "confirmed bounce" in bounce_signal or "early bounce" in bounce_signal:
            return (
                "Action Suggestion: Consider a disciplined put sale if the strike still respects support.",
                "The stock is in a more usable part of the support map, but this is still a notch below the cleanest setups. Keep assignment comfort and downside room front and center.",
            )
        if "in entry zone" in entry_status or "watch for stabilization" in entry_status:
            return (
                "Action Suggestion: Keep stalking this for a cleaner trigger.",
                "The underlying is acceptable and price is close enough to the entry area to matter, but the bounce still needs a bit more proof.",
            )
        return (
            "Action Suggestion: Keep it on the watchlist, but wait for a cleaner support-based entry.",
            "The underlying may still be acceptable, but the current location is not ideal enough to force a put sale before price pulls back or firms up near support.",
        )

    if label == "Stalk / Watchlist":
        return (
            "Action Suggestion: Stalk this name, but do not force a put sale yet.",
            "This looks more like a stock worth monitoring than a stock that is ready right now. Wait for a better pullback into support or clearer bounce confirmation before treating it like a live put-selling setup.",
        )

    if label == "Neutral / Wait":
        return (
            "Action Suggestion: Wait and reassess after price improves or support firms up.",
            "This is not broken, but the quality, entry, and risk profile are not aligned enough yet for a strong put-selling setup.",
        )

    if risk_score is not None and risk_score <= -5:
        return (
            "Action Suggestion: Avoid fresh put exposure here.",
            "Risk is elevated enough that a support failure or continued downside would make assignment less attractive.",
        )

    return (
        "Action Suggestion: Stay defensive.",
        "The setup is weak enough that patience is better than forcing a trade.",
    )


def build_confidence_explanation(row) -> str:
    confidence = row.get("confidence")
    label = row.get("label", "signal")

    if confidence is None:
        return "Confidence shows how consistently the scoring model supports the assigned label."

    if confidence >= 80:
        strength = "very high"
    elif confidence >= 68:
        strength = "solid"
    elif confidence >= 55:
        strength = "moderate"
    else:
        strength = "low"

    return (
        f"Confidence is the model's conviction in the **{label}** label, based on how well the quality, entry, risk, trend, support, and flow signals agree with each other. "
        f"This reading is **{strength}** at {confidence}%."
    )


def build_table_setup_note(row) -> str:
    support_basis = row.get("support_basis") or ""
    reasons = row.get("reasons") or ""
    bounce_signal = row.get("bounce_signal") or ""
    entry_status = (row.get("entry_status") or "").strip()

    basis_parts = [part.strip() for part in support_basis.split(",") if part.strip()]
    reason_parts = [part.strip() for part in reasons.split("|") if part.strip()]

    if bounce_signal == "Confirmed bounce":
        return "Support held and bounce confirmation is in place."
    if bounce_signal == "Early bounce":
        return "Support is starting to hold, but the bounce is still early."
    if bounce_signal == "At support, no bounce yet":
        return "At support, but still waiting for a bounce confirmation."
    if bounce_signal == "Near support":
        return "Close to support; watch for stabilization before selling a put."
    if bounce_signal == "No bounce setup" and entry_status == "Wait for pullback":
        return "Underlying looks constructive, but price is still above the better support-based entry."
    if bounce_signal == "Broken below support":
        return "Support failed, which makes fresh put selling less attractive."

    if bounce_signal and bounce_signal != "N/A":
        return bounce_signal

    if basis_parts:
        top_basis = " / ".join(basis_parts[:2])
        return f"Support from {top_basis}."

    if reason_parts:
        return reason_parts[0]

    return "Setup note unavailable."


def group_signal_reasons(reasons: list[str]) -> dict[str, list[str]]:
    grouped = {"Tailwinds": [], "Watch Items": [], "Risks": []}

    for reason in reasons:
        lowered = reason.lower()

        if any(
            cue in lowered
            for cue in (
                "below",
                "weak",
                "bear",
                "risk",
                "pressure",
                "distribution",
                "underperform",
                "negative",
                "broken",
                "caution",
                "less attractive",
            )
        ):
            grouped["Risks"].append(reason)
        elif any(
            cue in lowered
            for cue in (
                "wait",
                "neutral",
                "modest",
                "less ideal",
                "not ideal",
                "stabilization",
                "somewhat",
                "mixed",
            )
        ):
            grouped["Watch Items"].append(reason)
        else:
            grouped["Tailwinds"].append(reason)

    return grouped


def build_avoid_reason(row) -> str:
    reasons = row.get("reasons") or ""
    parts = [part.strip() for part in reasons.split("|") if part.strip()]

    negative_cues = (
        "below",
        "weak",
        "bear",
        "risk",
        "pressure",
        "distribution",
        "underperform",
        "negative",
        "broken",
        "stretched",
        "failing",
        "caution",
    )

    selected = [part for part in parts if any(cue in part.lower() for cue in negative_cues)]
    chosen = selected[:2] if selected else parts[:2]

    if not chosen:
        return "Risk profile is unfavorable for a fresh put-selling entry."

    return " | ".join(chosen)
