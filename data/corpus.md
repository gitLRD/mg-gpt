# Building the MG Corpus

The model is only as good as its training data.
For a niche domain model, quality beats quantity — 500KB of focused MG text
will outperform 5MB of diluted general content.

## Target size

| Config  | Minimum tokens | Rough text size |
|---------|---------------|-----------------|
| nano    | ~100K         | ~500KB          |
| small   | ~500K         | ~2.5MB          |
| medium  | ~2M+          | ~10MB+          |

One token ≈ 4 characters of English text.

---

## Recommended sources

### Workshop manuals (best signal)
- **Haynes MGB Manual** — If you own one, retype or OCR key sections:
  engine, carburettors, gearbox, bodywork, electrics.
- **British Leyland factory workshop manual** — PDFs circulate online.
  OCR the text, clean it up.

### Club resources
- **MG Owners Club** technical articles
- **MGOC Bulletin** back issues
- **Octane magazine** MG features (respect copyright — paraphrase liberally)

### Online forums (good for vernacular)
- BBS threads from mgexp.com, mgoc.co.uk
- Copy thread text and clean up HTML artefacts

### Your own knowledge
Create a file `data/corpus/my_notes.txt` and write down everything you know:
- Your MGB GT restoration notes
- SU carburettor setup procedures
- Common rust spots and treatment approaches
- The difference between chrome and rubber bumper cars
- Engine rebuild specifics

This is surprisingly valuable — it teaches the model your terminology.

---

## File format

Plain `.txt` files in `data/corpus/`. One file per source is cleanest.

```
data/corpus/
  haynes_engine.txt
  haynes_bodywork.txt
  mgoc_technical.txt
  forum_carb_threads.txt
  my_restoration_notes.txt
```

Each file should be clean prose — remove navigation text, ads, boilerplate.
Document structure (headings, numbered steps) is fine and helpful.

---

## Quick bootstrap corpus

If you want to start training *right now* with minimal effort,
create `data/corpus/bootstrap.txt` and populate it with:

1. Wikipedia articles: MGB, MG Cars, SU carburettor, British Leyland
2. The text from mgb-stuff.com technical pages
3. Your own restoration notes

That'll get you ~200K tokens and enough to see the model start producing
MG-flavoured output within the first training run.

---

## Quality tips

- Remove boilerplate (cookie notices, nav menus, "Posted by user123")
- Normalise encoding — run through `iconv` if you get weird characters
- Consistent formatting helps: paragraphs, not wall-of-text
- Include both Q&A style text (forum posts) and reference text (manual excerpts)
  — they teach different things
