"""
data/format_qa.py — Convert raw forum posts into USER/ASSISTANT dialogue.

Forum threads are naturally conversational: post 0 is the question/opener,
subsequent posts are replies.  We pair them as alternating USER/ASSISTANT turns
so the GPT learns to generate an ASSISTANT response when prompted.

Usage:
    python data/format_qa.py

Output:
    data/corpus_qa/*.txt  (one file per source thread, ready for prepare.py)
"""

import os
import re
import glob

INPUT_DIR  = "data/corpus_clean"   # PII-stripped corpus (run data/strip_pii.py first)
OUTPUT_DIR = "data/corpus_qa"
MIN_LENGTH = 60   # characters — skip posts that are too short to be useful


def clean_post(text):
    """Strip forum boilerplate that would confuse training."""
    # Attachment lines: "Attachments: foo.jpeg 45 KB"
    text = re.sub(r"Attachments:.*", "", text, flags=re.IGNORECASE)
    # "Sign In or Register to view this photo"
    text = re.sub(r"Sign In or Register to view this photo", "", text, flags=re.IGNORECASE)
    # "Reply Was this post helpful or interesting? Yes No Thank"
    text = re.sub(r"Reply Was this post helpful[^\n]*", "", text, flags=re.IGNORECASE)
    # "Edited N time(s). Last edit at ..."
    text = re.sub(r"Edited \d+ time\(s\)[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Last edit at[^\n]*", "", text, flags=re.IGNORECASE)
    # "In reply to # 12345 by username" quoted-reply headers
    text = re.sub(r"In reply to #\s*\d+[^\n]*", "", text, flags=re.IGNORECASE)
    # Stray image filenames: "ABC123.jpeg" or "DSCN5231.JPG"
    text = re.sub(r"\b[A-Z0-9_-]{6,}\.(jpeg|jpg|png|gif)\b", "", text, flags=re.IGNORECASE)
    # File sizes: "45 KB", "1.2 MB"
    text = re.sub(r"\b\d+(\.\d+)?\s*(KB|MB)\b", "", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_thread(posts):
    """Pair posts as USER/ASSISTANT dialogue, skipping short posts."""
    lines = []
    role_idx = 0  # 0 = USER, 1 = ASSISTANT, alternating
    for post in posts:
        post = clean_post(post)
        if len(post) < MIN_LENGTH:
            continue
        role = "USER" if role_idx % 2 == 0 else "ASSISTANT"
        lines.append(f"{role}: {post}")
        role_idx += 1
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    source_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not source_files:
        print(f"No .txt files found in {INPUT_DIR}/")
        return

    total_pairs = 0
    skipped = 0
    for filepath in source_files:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        raw_posts = [p.strip() for p in content.split("\n\n") if p.strip()]
        formatted = format_thread(raw_posts)

        # Need at least one USER and one ASSISTANT turn to be useful
        if formatted.count("USER:") < 1 or formatted.count("ASSISTANT:") < 1:
            skipped += 1
            continue

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(formatted + "\n\n")

        pairs = min(formatted.count("USER:"), formatted.count("ASSISTANT:"))
        total_pairs += pairs

    print(f"✅ {len(source_files) - skipped} threads → {total_pairs} Q&A pairs in {OUTPUT_DIR}/")
    if skipped:
        print(f"   {skipped} threads skipped (too short / no dialogue)")


if __name__ == "__main__":
    main()
