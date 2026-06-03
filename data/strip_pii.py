"""
data/strip_pii.py — Macie-style PII redaction for the forum corpus.

Reads each .txt in data/corpus/, redacts personally-identifiable information,
writes to data/corpus_clean/.  Subsequent steps (format_qa.py, prepare.py)
should read from corpus_clean/.

Categories covered (regex-based, no ML dependency):
  - Email addresses
  - Phone numbers (US, UK common formats)
  - URLs (sometimes contain usernames or personal blog/profile URLs)
  - Street addresses (number + Street/Road/Avenue/etc.)
  - US ZIP codes / UK postcodes
  - License plates / VINs (best-effort)
  - "Regards, NAME" / "Cheers, NAME" / "-NAME" signoffs
  - "Hi NAME," / "Dear NAME," / "Hello NAME," openers
  - "In reply to # NNNN by USERNAME" (forum quote headers)
  - "Sent from my iPhone" style mobile sigs

This is NOT a security-grade redaction — it's reasonable forum hygiene.
"""

import os
import re
import glob

INPUT_DIR  = "data/corpus"
OUTPUT_DIR = "data/corpus_clean"

# Common first names we'll redact when they appear in name-y contexts. Not
# exhaustive — just catches the long tail of forum signoffs we observed
# (Eric, Jeff, Brian, Terry, Ken, Bob, John, etc.).
COMMON_FIRST_NAMES = {
    "aaron", "alan", "alex", "andrew", "andy", "angie", "anthony", "art",
    "barry", "ben", "bill", "billy", "bob", "bobby", "brad", "brian", "bruce",
    "carl", "charlie", "chris", "colin", "craig", "curtis",
    "dale", "dan", "danny", "dave", "david", "dean", "dennis", "derek",
    "diane", "don", "doug",
    "ed", "eddie", "eric", "ernie",
    "frank", "fred",
    "gary", "george", "glen", "graham", "greg",
    "harry", "henry", "howard",
    "ian", "irv",
    "jack", "james", "jamie", "jason", "jeff", "jeffrey", "jerry", "jim",
    "jimmy", "joe", "john", "johnny", "jon", "joseph", "josh", "julian",
    "keith", "ken", "kevin", "kyle",
    "larry", "lee", "leo", "lloyd", "loren",
    "marcus", "mark", "martin", "matt", "matthew", "michael", "mike", "mickey",
    "nathan", "neil", "nick",
    "oliver", "owen",
    "patrick", "paul", "pete", "peter", "phil", "philip",
    "ralph", "randy", "ray", "richard", "rick", "rob", "robert", "robin",
    "rod", "roger", "ron", "ross", "roy", "russ", "ryan",
    "sam", "scott", "sean", "shaun", "simon", "stan", "stephen", "steve",
    "stuart", "terry", "thomas", "tim", "todd", "tom", "tony", "trevor",
    "victor", "vince",
    "walter", "wayne", "william",
}


def _redact_emails(text):
    return re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[EMAIL]", text)


def _redact_phones(text):
    # UK: +44 7xxx xxx xxx, 07xxx xxx xxx, etc.
    text = re.sub(r"\+44\s?\d[\d\s-]{8,}\d", "[PHONE]", text)
    text = re.sub(r"\b0\d{4}\s?\d{3}\s?\d{3,4}\b", "[PHONE]", text)
    # US: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx
    text = re.sub(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b", "[PHONE]", text)
    return text


def _redact_urls(text):
    # Drop entire URLs — forum posts often link personal profiles, blogs, ebay
    # listings etc. We keep the surrounding text.
    return re.sub(r"https?://\S+", "[URL]", text)


def _redact_addresses(text):
    # Street addresses: number + 1-4 words + street type
    pat = (r"\b\d{1,5}\s+([A-Z][a-zA-Z]+\s+){1,4}"
           r"(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Court|Ct|"
           r"Way|Boulevard|Blvd|Place|Pl|Highway|Hwy|Parkway|Pkwy)\.?\b")
    return re.sub(pat, "[ADDRESS]", text)


def _redact_postcodes(text):
    # UK postcode: 1-2 letters, 1-2 digits, optional letter, space, digit, 2 letters
    text = re.sub(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", "[POSTCODE]", text)
    # US ZIP: 5 digits or 5+4
    text = re.sub(r"\b\d{5}(?:-\d{4})?\b(?=\s|$|,|\.)", "[ZIP]", text)
    return text


def _redact_plates_vins(text):
    # 17-character VIN (no I/O/Q)
    text = re.sub(r"\b[A-HJ-NPR-Z0-9]{17}\b", "[VIN]", text)
    return text


def _redact_signoffs(text):
    # "Regards, Eric" / "Cheers, Jeff" / "Thanks, Mike"
    pat = r"\b(Regards|Cheers|Thanks|Sincerely|Best|Yours|Kind regards),?\s+([A-Z][a-z]+)"
    text = re.sub(pat, lambda m: f"{m.group(1)}, [NAME]", text)
    # "-Eric" or " - Eric" (end-of-message dash signoff)
    text = re.sub(r"(?:^|\n|\.\s)[-—]\s?([A-Z][a-z]+)\s*(?=\n|$|\.)",
                  lambda m: m.group(0).replace(m.group(1), "[NAME]"), text)
    return text


def _redact_openers(text):
    # "Hi Eric," / "Hello Jeff," / "Dear Brian," / "Hey Mike,"
    pat = r"\b(Hi|Hello|Dear|Hey|Howdy)\s+([A-Z][a-z]+)([,!])"
    return re.sub(pat, lambda m: f"{m.group(1)} [NAME]{m.group(3)}", text)


def _redact_quote_headers(text):
    # "In reply to # 5053411 by JoeReed"  — format_qa already strips these but
    # do it here too so they don't poison the unformatted corpus.
    text = re.sub(r"In reply to #?\s*\d+\s+by\s+\S+",
                  "In reply to [USER]", text, flags=re.IGNORECASE)
    # "Posted by SomeUser" / "by SomeUser123"
    text = re.sub(r"\bPosted by\s+\S+", "Posted by [USER]", text)
    return text


def _redact_mobile_sigs(text):
    text = re.sub(r"Sent from my (iPhone|iPad|Android|phone|mobile|Galaxy\s\S+).*",
                  "", text, flags=re.IGNORECASE)
    return text


def _redact_common_first_names(text):
    """Catch the long tail of bare-name mentions ("Eric, thanks for the help")."""
    def repl(match):
        word = match.group(0)
        if word.lower() in COMMON_FIRST_NAMES:
            return "[NAME]"
        return word
    # Only redact names that look like a name: capitalised, followed by
    # comma/space/end-of-line. Avoids gutting random capitalised words.
    return re.sub(r"\b[A-Z][a-z]+(?=[,.\s])", repl, text)


def strip_pii(text):
    """Apply all redactions in order. Order matters — URLs before phones etc."""
    text = _redact_emails(text)
    text = _redact_urls(text)
    text = _redact_mobile_sigs(text)
    text = _redact_quote_headers(text)
    text = _redact_phones(text)
    text = _redact_addresses(text)
    text = _redact_postcodes(text)
    text = _redact_plates_vins(text)
    text = _redact_signoffs(text)
    text = _redact_openers(text)
    text = _redact_common_first_names(text)
    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))

    # Quick stats — counts before/after for spot-checking
    total_redactions = {
        "[EMAIL]": 0, "[PHONE]": 0, "[URL]": 0, "[ADDRESS]": 0,
        "[POSTCODE]": 0, "[ZIP]": 0, "[VIN]": 0, "[NAME]": 0, "[USER]": 0,
    }

    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()

        cleaned = strip_pii(text)

        for tag in total_redactions:
            total_redactions[tag] += cleaned.count(tag) - text.count(tag)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

    print(f"✅ Stripped PII from {len(files)} files → {OUTPUT_DIR}/")
    print("Redaction counts:")
    for tag, n in total_redactions.items():
        if n > 0:
            print(f"  {tag:12s} {n:,}")


if __name__ == "__main__":
    main()
