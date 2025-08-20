# 🎬 Movie Recommendation System

This project recommends movies to users using **Content-Based Filtering** and **Collaborative Filtering (Matrix Factorization with SVD)** on the [MovieLens Dataset](https://files.grouplens.org/datasets/movielens/).

## 📂 Dataset
- Source: [MovieLens (ml-latest-small)](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)  
- Automatically downloaded if not found locally.  
- Contains **610 users**, **9,000+ movies**, and **100,000 ratings**.

## 🛠️ Approaches Used
- **Content-Based Filtering**  
  - TF-IDF on movie **titles + genres**  
  - Cosine similarity for recommendations  

- **Collaborative Filtering**  
  - User–Item matrix with mean-centering  
  - **Truncated SVD** for matrix factorization  
  - Recommends movies based on similar users  

## 📊 Example Results
| Query                            | Top Recommendations (Sample) |
|----------------------------------|-------------------------------|
| `--similar "Toy Story (1995)"`   | Jumanji (1995), Aladdin (1992), Lion King (1994) |
| `--user 1 --topn 5`              | Braveheart (1995), Apollo 13 (1995), Batman Forever (1995) |

---

## 🚀 How to Run
### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Build models
```bash
python recommender.py --build
```

### 3. Usage Examples
# 🔍 Search movies
```bash
python recommender.py --search "toy"
```
```bash
# 🎥 Recommend similar movies (content-based)
python recommender.py --similar "Toy Story (1995)" --topn 5
```

```bash
# 👤 Recommend for a user (collaborative filtering)
python recommender.py --user 1 --topn 5
```
