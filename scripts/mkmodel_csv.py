#!/bin/env python

#
# Jed Dobson
# James.E.Dobson@Dartmouth.EDU
# June 2020
# 
# htrc-vector-project
#


import csv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import bz2

pages = list()
# open CSV file and read individual pages of tokens
with bz2.open('drama_17412.csv.bz2','rt',encoding = "ISO-8859-1") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
       pages.append(row),

print("finished reading")
# produce tagged_data
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(pages)]

print("creating model")
model = Doc2Vec(tagged_data, 
                dm=1, # operate on paragraphs with distributed memory model
                vector_size=300, # larger vector size might produce better results
                min_count=5, # drop words with very few repetitions
                window=100, # larger window size because of extracted features
                workers=4)

print("saving model")
model.save("doc2vec-07-28-2020-drama.w2v")
