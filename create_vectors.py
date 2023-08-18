import glob
import json
import spacy

from gensim.models.doc2vec import TaggedDocument, Doc2Vec


nlp = spacy.load("de_core_news_md")

def fill_set_till_5(sim, correct):
    cand = {correct}
    cand_5 = set(sim[:3])
    if not cand.intersection(cand_5):
        cand_5.pop()
        cand_5.add(correct)
    return list(cand_5)

def create_punchline_quiz(bars, rapper, model):
    punchline = []
    vectors = [model.infer_vector(b) for b in bars]
    similar_docs = [model.dv.most_similar(v) for v in vectors]
    candidates = [fill_set_till_5([x[0] for x in s], r) for s, r in zip(similar_docs, rapper)]
    for b, c, t in zip(bars, candidates, rapper):
        punchline.append({"punchline": b, "candidates": c, "true": t})
    return punchline

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
import re

token_bars = []
for k, v in cleaned_bars.items():
    token_bars += [TaggedDocument(
        [t.lemma_.lower() for t in nlp(re.sub("\W", " ", " ".join([x for x in vv if not x in german_stop_words]))) if
         not t.lemma_ == ' '], [k]) for vv in v]
## Train doc2vec model
model = Doc2Vec(token_bars, vector_size=100, window=100, min_count=5, workers=4, epochs=100)
# Save trained doc2vec model
model.save("test_doc2vec.model")
## Load saved doc2vec model
model = Doc2Vec.load("test_doc2vec.model")

rapper = []
for k, v in cleaned_bars.items():
    if k == 'brutos brutaloz':
        continue
    rapper += [k for vv in v]

with open("./punchline_quiz.json", "w") as f:
    json.dump(create_punchline_quiz([t['words'] for t in token_bars], rapper, model))
