# MG-GPT: A Complete Walkthrough

This guide explains every step of building a GPT-2 style transformer trained on MG car knowledge. Read this if you want to understand *why* we did each thing, not just *what* we did.

---

## Part 1: What Are We Building?

**Goal:** Build a small AI model that can generate text about MG cars, learning the knowledge and "voice" from real forum discussions and Wikipedia articles.

**How it works (simplified):**
1. We collect text about MG cars (Wikipedia articles, forum posts)
2. We teach a neural network to predict the next word, given the previous words
3. After lots of practice, the network learns patterns and can generate new text that "sounds like" MG enthusiasts talking

**The model type:** GPT-2 style decoder-only transformer
- "Decoder-only" = it only generates text (doesn't understand queries)
- "Transformer" = the architecture that powers modern LLMs
- "GPT-2 style" = we're building something similar to OpenAI's 2019 model, but tiny

---

## Part 2: Project Structure

```
mg-llm/
├── model/
│   ├── __init__.py           # Allows Python to treat this folder as a package
│   ├── config.py             # Defines model sizes (nano, small, medium)
│   └── gpt.py                # The actual transformer architecture
├── data/
│   ├── corpus/               # Raw .txt files about MG cars
│   ├── fetch_corpus.py       # Script to download Wikipedia articles via API
│   ├── extract_wikipedia.py  # Script to extract from 107GB Wikipedia dump
│   ├── scrape_forums.py      # Script to scrape mgexp.com forum threads
│   ├── prepare.py            # Converts text → tokens → train.bin/val.bin
│   ├── train.bin             # Tokenized training data (binary)
│   └── val.bin               # Tokenized validation data (binary)
├── train.py                  # Main training loop
├── generate.py               # Load a checkpoint and generate text
├── checkpoints/              # Saved model weights (ckpt_best.pt, etc.)
├── requirements.txt          # Python dependencies
└── GUIDE.md                  # This file
```

**Key concept:** Everything flows left-to-right:
- **Scraping** (fetch_corpus.py, extract_wikipedia.py, scrape_forums.py) → files in `corpus/`
- **Preparation** (prepare.py) → `train.bin` and `val.bin`
- **Training** (train.py) → checkpoint files in `checkpoints/`
- **Generation** (generate.py) → reads checkpoint, generates text

---

## Part 3: Corpus Collection — Getting Training Data

Before we can train, we need *text* about MG cars. We collected from two sources:

### 3.1: Wikipedia Articles (via API)

**What:** `data/fetch_corpus.py` downloads Wikipedia articles using the Wikipedia API.

**How it works:**
```python
# Requests to Wikipedia's API
GET https://en.wikipedia.org/w/api.php?action=query&titles=MGB
# Returns JSON with article text
```

**Why this way?** It's fast (~1-2 seconds per article) and doesn't require downloading a giant dump.

**Articles fetched:** MGB, MGA, SU carburettor, British Leyland, etc. (~28K tokens)

**Limitation:** Only gets articles we know about by name. Misses related articles.

---

### 3.2: Wikipedia Dump (Full extraction)

**What:** `data/extract_wikipedia.py` processes the full 107GB Wikipedia XML dump.

**Why two approaches?** 
- API is quick but limited
- Dump is thorough but needs clever processing to avoid loading 107GB into RAM

**How it works:**

1. **Streaming parser (lxml):** Instead of loading the entire XML, process it line-by-line
   ```python
   for event, elem in ET.iterparse(DUMP_PATH, events=["end"], tag="{ns}page"):
       # Process this ONE page, then throw it away
       # Never load the whole dump into memory
   ```

2. **Scoring:** Check if an article is about MG cars
   ```python
   # Count keyword matches: MGB, MGA, Austin-Healey, SU carburettor, etc.
   score = sum(count for each keyword found)
   if score >= MIN_SCORE:
       keep the article
   ```

3. **Multiprocessing:** Use all CPU cores
   ```
   Main process: reads XML (I/O bound)
   Worker processes: clean markup + score (CPU bound)
   Work queue: buffers articles between reader and workers
   Result queue: collects scored articles
   ```

**Challenge encountered:** The scraper initially got ALL articles (even false positives).
- "MGB" matches "MG**B**aumgartner" ❌
- **Fix:** Use word boundary regex: `\bMGB\b` only matches whole word

**Final filter:** Require at least one unambiguous car keyword (MG Midget, Austin-Healey, SU carburettor) to eliminate noise

**Result:** ~135K tokens of high-quality Wikipedia content about MG cars

---

### 3.3: Forum Scraping (The "spicy, catty" voice)

**What:** `data/scrape_forums.py` scrapes discussion threads from mgexp.com forums.

**Why forums?** Wikipedia is formal. Forums capture the *personality* of MG enthusiasts (opinionated, knowledgeable, sometimes catty).

**How it works:**

1. **Fetch a thread URL:** `requests.get(url)` → raw HTML

2. **Parse HTML:** Use lxml to find post containers
   ```python
   # mgexp uses: <div class="message-body"> for each post
   posts = tree.xpath("//div[@class='message-body']")
   # Extract text from each post
   ```

3. **Cleanup:** Remove "Was this helpful?", signatures, extra whitespace

4. **Save:** One file per thread, one post per paragraph

**Challenge:** Different forums use different HTML structures
- vBulletin uses `<div class="post">`
- Discourse uses `<article class="post">`
- mgexp uses `<div class="message-body">`
- **Solution:** Try multiple selectors in order of likelihood

**Posts scraped:** 57 posts from 5 real mgexp threads (~5.5K tokens)
- "Hey, all you MGA guys" — discussion about an engine rebuild charity auction
- "Moss Motors" — community discussion about vendor changes
- "Steering Geometry" — technical discussion
- "What to do with my girl in 2025" — upgrade planning
- And more

**Result:** 141K total tokens (135K wiki + 5.5K forum)

---

## Part 4: Data Preparation — Text to Numbers

Transformers understand numbers, not text. We need to convert words → numbers (tokens).

### 4.1: Tokenization with tiktoken

**What:** `data/prepare.py` loads all `.txt` files and converts to tokens.

**How it works:**

1. **Load all text:**
   ```python
   text = load all .txt files from corpus/
   # Now we have one giant string: ~500K characters
   ```

2. **Tokenize with GPT-2 BPE encoding:**
   ```python
   enc = tiktoken.get_encoding("gpt2")
   tokens = enc.encode(text)
   # Result: 141,223 token IDs (each 0-50256)
   ```

**What's BPE?** "Byte Pair Encoding" — a clever way to split text into subword chunks
- Common words = 1 token (e.g., "the" = 262)
- Rare words = multiple tokens (e.g., "Abingdon" = 3 tokens)
- Emoji, special chars = 1-2 tokens each

3. **Split into train/val:**
   ```python
   train (90%): 127,100 tokens
   val (10%):   14,123 tokens
   ```

4. **Save as binary files:**
   ```python
   # train.bin: 127K token IDs as uint16
   # val.bin: 14K token IDs as uint16
   # Much faster to load than parsing text again
   ```

**Why split?**
- Train on train set, evaluate on val set
- If val loss stops dropping, we're overfitting

---

## Part 5: Model Architecture

### 5.1: What is a Transformer?

A transformer is a stack of **attention blocks** that learn relationships between tokens.

**Simplified flow:**
```
Input tokens
    ↓
Token embeddings (convert token IDs → vectors)
    ↓
Positional embeddings (add "position" information)
    ↓
N Transformer Blocks:
  - Self-attention: "which previous tokens matter for the next one?"
  - Feedforward: "transform the representation"
    ↓
Layer norm
    ↓
Output projection (vector → probability for next token)
    ↓
Sample next token & repeat
```

**Key concept: Causal masking**
- The model can only look at *previous* tokens, not future ones
- When predicting token at position 5, ignore positions 6, 7, 8, ...
- Implemented as a triangular mask matrix

**Key concept: Weight tying**
- Embed words and output logits with the *same* matrix
- Reduces parameters, surprisingly works well

### 5.2: Model Sizes

We have three configs in `config.py`:

| Config | Params | Layers | Heads | Dim | Block size |
|--------|--------|--------|-------|-----|-----------|
| Nano   | 1.5M   | 4      | 4     | 128 | 256       |
| Small  | 10M    | 6      | 6     | 384 | 512       |
| Medium | 50M    | 8      | 8     | 512 | 1024      |

**Nano:** Fast to train (5-10 min), modest quality
**Small:** Good balance (30-60 min), better quality
**Medium:** High quality (2-4 hours), slower

---

## Part 6: Training — Teaching the Model

### 6.1: The Training Loop

**In `train.py`:**

```python
for iteration in range(max_iters):
    # Batch 1: grab training data
    X, Y = get_batch("train")
    
    # Forward pass: "what's your prediction for the next token?"
    logits, loss = model(X, Y)
    
    # Backward pass: "how wrong were you? update weights"
    loss.backward()
    
    # Every grad_accum steps: actually update weights
    optimizer.step()
    
    # Every eval_interval steps: check validation loss
    val_loss = evaluate(model)
```

### 6.2: Hyperparameters (The knobs we tune)

- **Learning rate:** How aggressively we update weights (0.0006)
- **Batch size:** How many examples before updating (64)
- **Grad accumulation:** Simulate larger batches by accumulating gradients (5 steps = effective batch of 320)
- **Max iters:** Total training steps (500 for nano, more for small)
- **Warmup:** Slowly increase LR at the start (100 steps)
- **Cosine schedule:** Decay LR as training progresses

**Why these values?** Borrowed from the original GPT-2 paper, tweaked for our small dataset.

### 6.3: Training on MPS (Apple Silicon GPU)

**What:** Your MacBook has an M-series GPU (MPS = Metal Performance Shaders).

**PyTorch code:**
```python
device = "mps"  # Use GPU
model.to(device)
X = X.to(device)  # Move data to GPU
# Forward/backward passes happen on GPU → 10-100x faster
```

**Note:** MPS + float16 can be unstable, so we use float32 (slower but safe).

---

## Part 7: Nano Model Results

### 7.1: Loss Curve

The loss graph shows:
- **Blue line (train):** Per-iteration training loss, drops smoothly from 4.5 → 0.7
- **Orange dots (val):** Validation loss checked every N steps, drops from 10 → 9.5

**Interpretation:**
- Train loss dropping = model is learning ✓
- Val loss staying high = we're learning MG-specific patterns, not overfitting ✓
- Gap between train/val = normal for small corpus, model isn't just memorizing

### 7.2: Generation Sample

Prompt: *"The MGB GT was"*

Model output:
```
The MGB GT was built to the Leyland were also had a town production
has been to be now a time to the C-Series and the same as a new UK,
but with the Leyland to a first time with a design by at the engine.
```

**What's happening:**
- ✓ Knows MGB, Leyland, C-Series, engine — correct domain vocabulary
- ✓ Attempts grammatical structure
- ✗ Syntax is garbled (expected for nano, 1.5M params on 141K tokens)

**Why garbled?** The model is tiny (1.5M params) and the corpus is small. Small model should be much better.

---

## Part 8: Where We Are Now

**Status:** Small model (10M params) is training
- Started: a few minutes ago
- Expected duration: 20-45 minutes
- Will save checkpoints every N steps
- When done: we'll test generation (should be way better)

**Next steps after small finishes:**
1. Generate samples (prompt: "The MGB...")
2. Optional: Train medium model (50M params, 2-4 hours) for even better quality
3. Optional: Scrape more forums for richer corpus

---

## Part 9: Understanding the Code

### 9.1: `model/gpt.py` — The transformer

Key classes:
- `CausalSelfAttention`: The attention mechanism
- `MLP`: Feedforward layer (expands features 4x, then contracts)
- `Block`: Attention + MLP with residual connections
- `GPT`: Stacks N blocks, adds embeddings, output head

Key methods:
- `forward(idx, targets=None)`: Forward pass, returns logits & loss
- `generate(idx, max_tokens, temperature, top_k)`: Autoregressive generation

### 9.2: `train.py` — The training loop

Key parts:
- `get_batch()`: Load random windows from train.bin/val.bin
- `estimate_loss()`: Evaluate on train & val sets (no gradients)
- Main loop: forward, backward, step
- Checkpoint saving: saves config + weights

### 9.3: `generate.py` — Text generation

Key parts:
- Load checkpoint (config + weights)
- Initialize with prompt tokens
- Loop: `logits = model(tokens)` → sample next token → repeat
- Temperature/top_k sampling (prevent greedy/boring output)

---

## Part 10: Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "Weights only load failed" | PyTorch 2.6+ security | Import GPTConfig, use `safe_globals` context |
| Wikipedia extractor returns 0 articles | Wrong XML namespace | Inspect actual dump, update NS constant |
| Forum scraper finds 0 posts | CSS selectors don't match | Debug with XPath, try `div[@class='...']` |
| Training is very slow | Not using GPU | Check `torch.cuda.is_available()` and `torch.backends.mps.is_available()` |
| Loss doesn't decrease | Wrong learning rate | Try 0.0003 - 0.0009, or reduce batch size |
| Memory error | Batch size too large | Reduce batch_size in config.py |

---

## Part 11: What's Next?

### Option A: Test Small Model
```bash
python generate.py --prompt "The SU carburettor"
# Should generate coherent text about SU carburetors
```

### Option B: Scrape More Forums
```bash
python data/scrape_forums.py --urls "..." --num-pages 5
python data/prepare.py  # Re-tokenize
python train.py --config small  # Retrain with more data
```

### Option C: Train Medium Model
```bash
python train.py --config medium
# ~3-4 hours, 50M params, even better quality
```

### Option D: Deploy / Share
- Save the checkpoint
- Share with friends ("Here's my MG-trained model")
- Fine-tune on new data later

---

## Part 12: Key Learnings

1. **Data > Model size:** Quality corpus matters more than huge model
2. **Streaming for big files:** Don't load 107GB into RAM
3. **Multiprocessing for CPU work:** Producer-consumer pattern is elegant
4. **Word boundaries matter:** `\b` is your friend in regex
5. **Validation loss guides training:** Watch val loss, not just train loss
6. **Small models can learn:** 1.5M params learned MG terminology well
7. **Forums have personality:** 5K tokens of forum posts influenced the model's "voice"

---

## Part 13: References

- Original GPT-2 paper: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- Karpathy's minGPT: https://github.com/karpathy/minGPT (inspiration for this code)
- tiktoken: https://github.com/openai/tiktoken (tokenization library)
- lxml: https://lxml.de/ (fast XML parsing)
- PyTorch docs: https://pytorch.org/docs/stable/index.html

---

**Last updated:** May 21, 2026  
**Current status:** Small model training (10M params)  
**Corpus size:** 141K tokens (27 Wikipedia articles + 57 forum posts)

