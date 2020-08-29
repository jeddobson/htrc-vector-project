#!/usr/bin/env python

from sklearn.linear_model import LinearRegression
import numpy as np
import bz2
import gensim.models.keyedvectors as kv
import collections
import scipy.spatial.distance as sd
from scipy.stats import pearsonr
import pickle
import csv

drama_model = kv.KeyedVectors.load_word2vec_format(bz2.open("../models/drama-17412-reduced.w2v.bz2"))
poetry_model = kv.KeyedVectors.load_word2vec_format(bz2.open("../models/poetry-56817-reduced.w2v.bz2"))


# extract common vocabulary
common_vocab = [word for word in drama_model.vocab if word in poetry_model.vocab]
common_vectors_drama = drama_model[common_vocab]
common_vectors_poetry = poetry_model[common_vocab]
print("common vocabulary: {0}".format(len(common_vocab)))

lin_model.fit(common_vectors_drama, 
              common_vectors_poetry)


aligned_vectors = collections.OrderedDict()
for idx, word in enumerate(common_vocab):
        wv = lin_model.predict(common_vectors_poetry[idx].reshape(1, -1))
        aligned_vectors[word] = wv.reshape(-1)

common_vocab = list(aligned_vectors.keys())
aligned_vectors = np.array(list(aligned_vectors.values()))

def neighbors(word,model,k=100):
    neighbors=list()
    idx = common_vocab.index(word)
    vec = model[idx]
    scores = sd.cdist([vec], model, "cosine")[0]
    sorted_ids = np.argsort(scores)
    for i in range(1, k + 1):
        neighbors.append((common_vocab[sorted_ids[i]], scores[sorted_ids[i]]))
    return neighbors


def pairwise_distance(word1,word2,model):
    idx1 = common_vocab.index(word1)
    idx2 = common_vocab.index(word2)
    dist = sd.cdist([model[idx1]], [model[idx2]], "cosine")[0]
    return dist[0]


def alignment_scores(word,k=100):
    n0 = neighbors(word,common_vectors_drama,k)
    n1 = neighbors(word,common_vectors_poetry,k)
    aw = list(set([x[0] for x in n0] + [x[0] for x in n1]))

    A = collections.OrderedDict()
    B = collections.OrderedDict()
    
    for w in aw:
        A[w] = pairwise_distance(word,w,common_vectors_drama)
        B[w] = pairwise_distance(word,w,common_vectors_poetry)
        
    z0 = [A[x] for x in A.keys()]   
    z1 = [B[x] for x in B.keys()]

    return pearsonr(z0,z1)[0]


rows=list()
# open Harvard Inquirer
with open('../lexicons/Harvard_Inquirer-inqtabs.txt', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for i,row in enumerate(reader):
        if i == 0:
            header = row
        else:
            rows.append(row)

# extract words of interest
# categories = ["Pleasur","Pain","Feel","Arousal", "Polit@", "Relig"]
inquirer = dict()

for c in header:
   
    # extract index from header
    idx = header.index(c)
   
    # extract words and drop to lowercase
    inquirer[c] = [w[0].lower() for w in rows if w[idx] != '']
   
    # remove alternate meanings and reduce duplicates
    inquirer[c] = list(set([w.split("#")[0] for w in inquirer[c]]))


def compare_alignments(category):
    values = dict()
    for term in inquirer[category]:
        if term in common_vocab:
            values[term] = alignment_scores(term)
    return sorted(values.items(), key=lambda x: x[1],reverse=True)


def fitted_neighbors(word,k=10):
    n0 = neighbors(word,common_vectors_drama,k)
    n1 = neighbors(word,common_vectors_poetry,k)
    for n,m in enumerate([n0,n1]):
        print("model: {0}".format(n))
        for w,v in m:
            print(w,v)
        print("\n")
    print("fitted model:")
    
    for w,v in neighbors(word,aligned_vectors,k):
        if w in [x[0] for x in n0] and w in [x[0] for x in n1]:
            print(w,v,"both")
        elif w in [x[0] for x in n0]:
            print(w,v,"m0")
        elif w in [x[0] for x in n1]:
            print(w,v,"m1")
        else:
            print(w,v)


hioi = ['Positiv', 'Negativ', 'Pstv', 'Affil', 'Ngtv', 'Hostile', 'Strong', 'Power', 'Weak', 'Submit', 'Active', 'Passive', 'Pleasur', 'Pain', 'Feel', 'Arousal', 'EMOT', 'Virtue', 'Vice', 'Ovrst', 'Undrst', 'Academ', 'Doctrin', 'Econ@', 'Exch', 'ECON', 'Exprsv', 'Legal', 'Milit', 'Polit@', 'POLIT', 'Relig', 'Role', 'COLL', 'Work', 'Ritual', 'SocRel', 'Race', 'Kin@', 'MALE', 'Female', 'Nonadlt', 'HU', 'ANI', 'PLACE', 'Social', 'Region', 'Route', 'Aquatic', 'Land', 'Sky', 'Object', 'Tool', 'Food', 'Vehicle', 'BldgPt', 'ComnObj', 'NatObj', 'BodyPt', 'ComForm', 'COM', 'Say', 'Need', 'Goal', 'Try', 'Means', 'Persist', 'Complet', 'Fail', 'NatrPro', 'Begin', 'Vary', 'Increas', 'Decreas', 'Finish', 'Stay', 'Rise', 'Exert', 'Fetch', 'Travel', 'Fall', 'Think', 'Know', 'Causal', 'Ought', 'Perceiv', 'Compare', 'Eval@', 'EVAL', 'Solve', 'Abs@', 'ABS', 'Quality', 'Quan', 'NUMB', 'ORD', 'CARD', 'FREQ', 'DIST', 'Time@', 'TIME', 'Space', 'POS', 'DIM', 'Rel', 'COLOR', 'Self', 'Our', 'You', 'Name', 'Yes', 'No', 'Negate', 'Intrj', 'IAV', 'DAV', 'SV', 'IPadj', 'IndAdj', 'PowGain', 'PowLoss', 'PowEnds', 'PowAren', 'PowCon', 'PowCoop', 'PowAuPt', 'PowPt', 'PowDoct', 'PowAuth', 'PowOth', 'PowTot', 'RcEthic', 'RcRelig', 'RcGain', 'RcLoss', 'RcEnds', 'RcTot', 'RspGain', 'RspLoss', 'RspOth', 'RspTot', 'AffGain', 'AffLoss', 'AffPt', 'AffOth', 'AffTot', 'WltPt', 'WltTran', 'WltOth', 'WltTot', 'WlbGain', 'WlbLoss', 'WlbPhys', 'WlbPsyc', 'WlbPt', 'WlbTot', 'EnlGain', 'EnlLoss', 'EnlEnds', 'EnlPt', 'EnlOth', 'EnlTot', 'SklAsth', 'SklPt', 'SklOth', 'SklTot', 'TrnGain', 'TrnLoss', 'TranLw', 'MeansLw', 'EndsLw', 'ArenaLw', 'PtLw', 'Nation', 'Anomie', 'NegAff', 'PosAff', 'SureLw', 'If', 'NotLw', 'TimeSpc', 'FormLw']



# calculate scores for each category
scores=dict()
meanv=dict()
for category in hioi:
    print("starting category: {0}".format(category))
    scores[category] = compare_alignments(category)
    meanv[category] = np.mean([x[1] for x in scores[category]])
    print("mean value: {0}".format(meanv[category]))


# save data
fp = open("inquirer-alignment-by-category.pkl","wb")
pickle.dump(scores,fp)
pickle.dump(meanv,fp)

