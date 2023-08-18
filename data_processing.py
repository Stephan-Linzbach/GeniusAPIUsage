# GET BARS

import json
import glob
import numpy as np
import scipy.stats
import spacy
from nltk.corpus import stopwords
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

data = {}

for g in glob.glob("./jsons/*"):
    with open(g) as f:
        data_f = json.load(f)
    for d in data_f:
        if not d in data:
            data[d] = data_f[d]
        else:
            if len(data[d]) < len(data_f[d]):
                data[d] = data_f[d]

data['negatiiv og'] = data['toobrokeforfiji']
del data['toobrokeforfiji']
data_new = {}
for d in data:
    if len(data[d]) > 20:
        data_new[d] = data[d]

for k, v in data_new.items():
    data_new[k] = [[x for x in vv if len(x) > 2] for vv in v if vv and len(vv) > 2]

bars = {k: [] for k in data_new.keys()}
for k, v in data_new.items():
    for vv in v:
        bars[k] += [vv[i:i + 4] for i in range(0, len(vv), 4)]

# bars = {k: [vv for vv in v if len(vv) == 4][:1400] for k, v in bars.items()}
# bars = {k: v for k, v in bars.items() if len(v) > 180}

already_known = {" ".join(v): set() for b in bars for v in bars[b]}
for b in bars:
    for v in bars[b]:
        already_known[" ".join(v)].add(b)

already_in = set()
cleaned_bars = {b: [] for b in bars}
for b in bars:
    for v in bars[b]:
        if len(already_known[" ".join(v)]) > 1:
            continue
        elif " ".join(v) in already_in:
            continue
        else:
            cleaned_bars[b].append(v)
            already_in.add(" ".join(v))

rapper = []
for k, v in cleaned_bars.items():
    rapper += [k for vv in v]

punchline = []
for k, v in cleaned_bars.items():
    punchline += [" ".join(vv) for vv in v]


# TOKENIZATION

token_punchline = []
nlp = spacy.load("de_core_news_md")

for doc in nlp.pipe(punchline, batch_size=100):
    token_punchline.append([t.text for t in doc if not t.is_punct])


# TFIDF

german_stop_words = set(stopwords.words('german'))

rapper_tok_p = {r: [] for r in set(rapper)}
for r, b in zip(rapper, token_punchline):
    rapper_tok_p[r].append(b)

rapper_tok_p = {r: [" ".join([x for x in b if not x in german_stop_words]) for b in bars] for r, bars in
                rapper_tok_p.items()}
rapper_tok_p = {r: " ".join(bars) for r, bars in rapper_tok_p.items()}
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(list(rapper_tok_p.values()))


# FASTTEXT

model = FastText(vector_size=100, window=6, min_count=10, sentences=token_punchline, epochs=50)

# Vector

## TFIDF + FASTTEXT

rapper_index = {k: i for i, k in enumerate(list(rapper_tok_p.keys()))}
feat = {k: i for i, k in enumerate(vectorizer.get_feature_names_out())}
punchline_vector_representation_tfidf = []
for r, p in zip(rapper, token_punchline):
    index = rapper_index[r]
    try:
        weights = np.array([X[index,feat[t.lower()]] if t.lower() in feat else 0 for t in p])
        vectors = model.wv.get_mean_vector(p, weights=weights)
    except Exception as e:
        vectors = np.zeros((100,))
    punchline_vector_representation_tfidf.append(vectors)
punchline_vector_representation_tfidf = np.stack(punchline_vector_representation_tfidf)

## FASTTEXT

rapper_index = {k: i for i, k in enumerate(list(rapper_tok_p.keys()))}
feat = {k: i for i, k in enumerate(vectorizer.get_feature_names_out())}
punchline_vector_representation = []
for r, p in zip(rapper, token_punchline):
    index = rapper_index[r]
    try:
        vectors = model.wv.get_mean_vector(p)
    except Exception as e:
        vectors = np.zeros((100,))
    punchline_vector_representation.append(vectors)
punchline_vector_representation = np.stack(punchline_vector_representation)


## Create Overview

distance_matrix = []

for p in punchline_vector_representation:
    distances = cdist(punchline_vector_representation, p.reshape(1, -1), metric='cosine')
    args = np.argsort(distances)[:100]
    dis = distances[args]
    distance_matrix.append([(punch, dist) for punch, dist in zip(args, dis)])

distance_matrix = np.stack(distance_matrix)

## Analyze Ginicoefficient Distribution

rapper_list = list(rapper_tok_p)
matrix_rapper_counts = np.zeros((distance_matrix.shape[0], len(rapper_list)))

for similar_punchlines in distance_matrix:
    for punch, _ in similar_punchlines:
        punch = int(punch)
        matrix_rapper_counts[punch][rapper_list.index(rapper[punch])] += 1

selection = []
for sim in distance_matrix:
    cand = set()
    for s, _ in sim:
        s = int(s)
        cand.add(rapper[s])
    selection.append(list(cand))

selection = [len(p) for p in selection]

lower_threshold = scipy.stats.scoreatpercentile(selection, 20)
higher_threshold = scipy.stats.scoreatpercentile(selection, 80)

keep_examples = np.where((selection >= lower_threshold) & (selection <= higher_threshold), True, False)

punchline_list = []
for k, v in cleaned_bars.items():
    punchline_list += [vv for vv in v]


playable_set = [p for p, k in zip(punchline_list, keep_examples) if k]
playable_true = [r for r, k in zip(rapper, keep_examples) if k]


playable_cand = []
for sim in distance_matrix[keep_examples]:
    cand = set()
    for s in sim:
        if len(cand) == 5:
            break
        cand.add(rapper[int(s[0])])
    playable_cand.append(list(cand))

punchline_quiz = []

for p, c, t in zip(playable_set, playable_cand, playable_true):
    punchline_quiz.append({"punchline": p, "candidates": c, "true": t})


with open("./punchline_quiz.json", "w") as f:
    json.dump(punchline_quiz, f)

