# 🎬 Movie Recommendation System (Content-Based + Collaborative)

An end-to-end recommender using the **MovieLens (ml-latest-small)** dataset.

- **Content-based**: cosine similarity on TF-IDF over movie *title + genres*  
- **Collaborative**: **matrix factorization** with **Truncated SVD** on a user-item matrix (mean-centered)
- **CLI**: recommend for a user or find movies similar to a given title
- **Artifacts**: cached models so you don’t rebuild every time

---

## 🚀 Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows PowerShell

# 2) Install dependencies
pip install -r requirements.txt

# 3) Build artifacts (auto-downloads MovieLens if not present)
python recommender.py --build

# 4a) Get top-10 recommendations for USER 1 (collaborative)
python recommender.py --user 1 --topn 10

# 4b) Get movies similar to a given title (content-based)
python recommender.py --similar "Toy Story (1995)" --topn 10
