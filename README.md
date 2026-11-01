<h1 align="center">AI-Based Lost & Found System</h1>

<p align="center">
  Semantic + visual matching of lost and found items using MPNet and CLIP embeddings.<br>
  No keyword search. No generative AI in the matching path.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/MPNet-sentence--transformers-orange" alt="MPNet">
  <img src="https://img.shields.io/badge/CLIP-ViT--B%2F32-blueviolet" alt="CLIP">
  <img src="https://img.shields.io/badge/Status-Phase%201%20complete-brightgreen" alt="Status">
</p>

---

## Why this exists

Lost-and-found boards rely on keyword search, so "red purse" never finds "red bag", and a photo can't be searched at all. This system embeds every report into vector space and matches by meaning, so different wording and mixed text/photo reports still find each other.

**Design constraint:** matching is 100% deterministic embedding similarity. No LLM decides matches, so every result is explainable: *"matched because cosine similarity was 0.78."*

## Features

- Submit **lost** or **found** reports with a text description and an optional photo
- **Semantic text matching** with MPNet (`all-mpnet-base-v2`)
- **Cross-modal matching** with CLIP: a text-only lost report can match a found item that only has a photo, and vice versa
- Matches only against the **opposite** category, ranked by similarity score
- Auto-generated Swagger docs at `/docs`

## How it works

```
 Submit item (text + optional photo)
            |
            v
 +---------------------------+
 |  FastAPI  POST /items     |
 +---------------------------+
            |
   +--------+---------+
   v                  v
 MPNet text       CLIP text + CLIP image
 embedding        embeddings (shared space)
   |                  |
   +--------+---------+
            v
 Cosine similarity vs every item in the opposite category
            |
            v
 Score >= 0.6  ->  returned as a match, sorted by score
```

### Scoring logic

| Lost item | Found item | Signals used |
|---|---|---|
| text | text | MPNet text vs text |
| text + photo | text + photo | MPNet text (0.6) + CLIP image vs image (0.4) |
| text | text + photo | MPNet text (0.6) + CLIP text vs photo (0.4) |
| text + photo | text | MPNet text (0.6) + CLIP photo vs text (0.4) |

Why CLIP and not a plain CNN classifier: CLIP gives a joint text-image embedding space, so a photo can be compared directly to a text description.

## Quick start

```bash
git clone https://github.com/likitha8276/ai-lost-and-found.git
cd ai-lost-and-found
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app
```

Open **http://127.0.0.1:8000/docs**. The first run downloads about 1 GB of model weights from Hugging Face.

## API example

**Request**: `POST /items` (multipart form)

| Field | Type | Required |
|---|---|---|
| `item_type` | `lost` or `found` | yes |
| `description` | string | yes |
| `image` | file | no |

**Response**

```json
{
  "item_id": 2,
  "item_type": "lost",
  "description": "red purse",
  "matches": [
    {
      "matched_item_id": 1,
      "matched_description": "red bag",
      "similarity_score": 0.7813
    }
  ]
}
```

## Evaluation

| Pair | Cosine score | Match (>= 0.6) |
|---|---|---|
| "blue water bottle" vs "blue coloured bottle for water" | 0.928 | Yes |
| "red handbag lost near library" vs "red color bag found near library" | 0.817 | Yes |
| "red bag" vs "red purse" | 0.781 | Yes |
| "red bag" vs "red purse missing near library" | 0.432 | No |
| "black wallet with cards" vs "found a red backpack" | 0.216 | No |
| "blue umbrella found in cafeteria" vs "red bag" | _fill in_ | No |
| Text-only vs photo (cross-modal) | _fill in_ | _fill in_ |

### Threshold: 0.6

True matches scored 0.78-0.93 and unrelated pairs scored about 0.2, so 0.6 sits in the gap between them. A false positive costs a user a pointless contact exchange. A false negative means an item is never recovered. That asymmetry is why the threshold leans toward recall without letting unrelated items through.

_Fill in: at 0.5, the umbrella pair [did / did not] false-positive._

### Known limitation: sentence dilution

"red bag" vs "red purse" scores 0.78, but adding context ("red purse missing near library") drops it to 0.43. Sentence embeddings encode the whole sentence, so filler words pull the vector away from the core object.

**Planned fix:** extract the core noun phrase (object, color, brand) before embedding, or embed object and context separately and weight the object higher.

## Project structure

```
.
├── main.py            # FastAPI app, POST /items endpoint
├── embeddings.py      # MPNet text, CLIP text, CLIP image encoders
├── matching.py        # cosine similarity + multimodal score combination
├── storage.py         # in-memory item store (Phase 1)
├── models.py          # Pydantic request/response schemas
├── test_matching.py   # hand-picked pair tests
└── requirements.txt
```

## Pre-trained vs built

- **Pre-trained:** MPNet and CLIP model weights.
- **Built by me:** the API service, embedding pipeline, cross-modal scoring and weighting, threshold tuning, and the labeled evaluation pairs.

## Tradeoffs

- Comparison is brute-force, O(n) per submission. Fine at campus scale. At larger scale the next step is `pgvector` or FAISS.
- Phase 1 storage is in memory, so a restart clears all items.
- The 0.6 / 0.4 weights are hand-chosen, not learned.
- Cold start: with no opposite-category items, `matches` is empty and the item waits for a counterpart.

## Roadmap

- [x] **Phase 1:** embedding + cosine similarity matching pipeline
- [ ] **Phase 2:** PostgreSQL persistence, JWT auth, user dashboard, admin moderation
- [ ] **Phase 3:** pgvector native search, Redis caching, match notifications
- [ ] Labeled evaluation set (40-60 items) with precision / recall / F1

## Author

**Likitha**: B.Tech CSE (Data Science), VIT-AP University
[GitHub](https://github.com/likitha8276)
