#!/bin/env python

# mktagged_docs.py: this script convert HTRC extracted features files into
# csv files of tokens appearing on each page and fit for parsing by Doc2Vec
#
# Jed Dobson
# James.E.Dobson@Dartmouth.EDU
# June 2020
# 
# htrc-vector-project
#

from htrc_features import FeatureReader
import csv
import os
from glob import glob

import argparse

parser = argparse.ArgumentParser(
    description='create tagged document from HTRC extracted features')

parser.add_argument('-f','--file',help="filename of HTRC file",dest='filename',action='store')
args = parser.parse_args()

if args.filename == None:
   print("Error: Need HTRC file!")
   exit()

# strip extension and add csv
output_csv = os.path.splitext(args.filename)[0]
output_csv = os.path.splitext(output_csv)[0]
output_csv = output_csv + ".csv"

# input file
htrc_file = args.filename

file = open(output_csv, 'w', newline='')
writer = csv.writer(file)

# iterate
tokens=list()

print("opening {0}".format(htrc_file))
fr = FeatureReader([htrc_file])
vol = next(fr.volumes())
ptc = vol.tokenlist(pos=False, case=False).reset_index().drop(['section'], axis=1)
page_list = set(ptc['page'])

print("reading pages")
for page in page_list:
  page_data = str()
  for page_tokens in ptc.loc[ptc['page'] == page].iterrows():
     if page_tokens[1][1].isalpha():
         page_data += (' '.join([page_tokens[1][1]] * page_tokens[1][2])) + " "
  writer.writerow(page_data.split())   
