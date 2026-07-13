"""Provenance Guard: Flask backend for AI-content-detection with appeals.

Routes:
  POST /submit          - classify a piece of text and log the decision
  POST /appeal           - contest a classification, flip status to under_review
  GET  /log               - recent audit log entries
  GET  /content/<id>     - stored state for one piece of content
"""

import uuid

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from scoring import attribution_for, combine_signals, label_for
from signals import llm_signal, stylometric_signal
from storage import append_log, get_content, init_db, mark_under_review, save_submission, recent_log

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")

init_db()

MIN_TEXT_LENGTH = 80


def _get_json_fields(*fields):
    """Parse the request body as JSON and require each field to be a non-empty string.

    Returns (values_dict, None) on success, or (None, (error_response, status)) on failure.
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return None, ({"error": "request body must be JSON"}, 400)

    values = {}
    for field in fields:
        value = body.get(field)
        if not isinstance(value, str) or not value.strip():
            return None, ({"error": f"'{field}' is required and must be a non-empty string"}, 400)
        values[field] = value
    return values, None


@app.route("/")
def index():
    """Serve the demo UI: submit content, view the verdict, appeal, inspect the log."""
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    """Classify submitted text as likely AI, likely human, or uncertain.

    Body: {"text": str, "creator_id": str}
    Runs both detection signals, combines them into a confidence score,
    persists the decision, and logs a classification event.
    """
    values, error = _get_json_fields("text", "creator_id")
    if error:
        body, status = error
        return jsonify(body), status

    text = values["text"]
    creator_id = values["creator_id"]

    if len(text.strip()) < MIN_TEXT_LENGTH:
        return jsonify({"error": "text too short for reliable analysis (minimum 80 characters)"}), 400

    content_id = str(uuid.uuid4())

    llm_result = llm_signal(text)
    stylo_result = stylometric_signal(text)

    llm_score = llm_result["score"] if llm_result["ok"] else None
    stylo_score = stylo_result["score"]

    confidence = combine_signals(llm_score, stylo_score)
    attribution = attribution_for(confidence)
    label = label_for(attribution)

    save_submission(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "text": text,
            "attribution": attribution,
            "confidence": confidence,
            "llm_score": llm_score,
            "stylo_score": stylo_score,
            "stylo_metrics": stylo_result["metrics"],
            "llm_rationale": llm_result["rationale"],
            "label": label,
            "status": "classified",
        }
    )

    append_log(
        event="classification",
        content_id=content_id,
        payload={
            "creator_id": creator_id,
            "attribution": attribution,
            "confidence": confidence,
            "llm_score": llm_score,
            "llm_ok": llm_result["ok"],
            "stylo_score": stylo_score,
            "stylo_metrics": stylo_result["metrics"],
            "label": label,
            "status": "classified",
            "degraded": llm_score is None,
        },
    )

    return (
        jsonify(
            {
                "content_id": content_id,
                "attribution": attribution,
                "confidence": confidence,
                "signals": {
                    "llm": {
                        "score": llm_score,
                        "ok": llm_result["ok"],
                        "rationale": llm_result["rationale"],
                    },
                    "stylometric": {
                        "score": stylo_score,
                        "metrics": stylo_result["metrics"],
                    },
                },
                "label": label,
                "status": "classified",
            }
        ),
        200,
    )


@app.route("/appeal", methods=["POST"])
@limiter.limit("5 per minute;20 per day")
def appeal():
    """Let a content creator contest an automated classification.

    Body: {"content_id": str, "creator_id": str, "creator_reasoning": str}
    Validates ownership, prevents duplicate open appeals, flips status to
    under_review, and logs an appeal event with a snapshot of the original
    decision alongside the creator's reasoning.
    """
    values, error = _get_json_fields("content_id", "creator_id", "creator_reasoning")
    if error:
        body, status = error
        return jsonify(body), status

    content_id = values["content_id"]
    creator_id = values["creator_id"]
    creator_reasoning = values["creator_reasoning"]

    content = get_content(content_id)
    if content is None:
        return jsonify({"error": "unknown content_id"}), 404

    if content["creator_id"] != creator_id:
        return jsonify({"error": "creator_id does not match the original submitter"}), 403

    if content["status"] == "under_review":
        return jsonify({"error": "an appeal is already open for this content"}), 409

    mark_under_review(content_id, creator_reasoning)

    append_log(
        event="appeal",
        content_id=content_id,
        payload={
            "creator_id": creator_id,
            "creator_reasoning": creator_reasoning,
            "original_decision": {
                "attribution": content["attribution"],
                "confidence": content["confidence"],
                "llm_score": content["llm_score"],
                "stylo_score": content["stylo_score"],
                "label": content["label"],
            },
            "status": "under_review",
        },
    )

    return (
        jsonify(
            {
                "content_id": content_id,
                "status": "under_review",
                "message": (
                    "Appeal received. Your content is now marked as under review and a "
                    "human reviewer will see your reasoning alongside the original "
                    "automated decision."
                ),
            }
        ),
        200,
    )


@app.route("/log", methods=["GET"])
def log():
    """Return the most recent audit log entries, newest first.

    Query param: limit (default 20, max 100).
    """
    limit = request.args.get("limit", default=20, type=int) or 20
    limit = max(1, min(limit, 100))
    return jsonify({"entries": recent_log(limit)})


@app.route("/content/<content_id>", methods=["GET"])
def content(content_id):
    """Return the full stored record for a single piece of content, including its text."""
    record = get_content(content_id)
    if record is None:
        return jsonify({"error": "unknown content_id"}), 404
    return jsonify(record)


@app.errorhandler(429)
def rate_limit_exceeded(e):
    """Return a JSON body for rate-limit errors instead of Flask-Limiter's default HTML."""
    return jsonify({"error": "rate limit exceeded", "detail": str(e.description)}), 429


if __name__ == "__main__":
    # Port 5000 collides with macOS AirPlay Receiver on some systems; use 5001 instead.
    app.run(debug=True, port=5001)
