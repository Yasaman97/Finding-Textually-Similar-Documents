# -*- coding: utf-8 -*-

# Importing useful libraries
import pandas as pd
import numpy as np
import string
import random
import time
from sympy import *
import seaborn as sns
import matplotlib.pyplot as plt

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Defining the path to the dataset
path = "pathtofile/enron.csv"
# Loading the dataset to Pandas Dataframe
df = pd.read_csv(path)

"""The dataset consists of the informations on nearly 500,000 emails between employees of Enron Corporation, and the main content of the emails can be found under 'content' column. The duplicated values are dropped, and only the content of 100 randomly selected emails are stored separately in a list as the documents we want to compare."""

df['content'].head()

# Keeping the first unique value and dropping the rest of the documents
df.drop_duplicates(subset=['content'], keep='first', inplace=True)
# Randomly picking 100 email contents as the documents
random.seed(42)
documents = random.sample(list(df['content']),20)
documents = [doc.lower() for doc in documents]
documents = [doc.translate(str.maketrans('', '', string.punctuation)) for doc in documents]

documents

"""1. The First method is Shingling. We define a function, kShingles, which receives a document, and a number k for the length of shingles. The function iterates through the document, and selects characters of length k, uses the built-in hash method to hash these characters to integers, and returns a set of the hashed shingles found in the document. Then, another function, Shingling is defined, which received a list of documents, and the shingling length, and calls the kShingles function in a loop to retrieve their set of shingles. The output of this function is a list containing the set of shingles for each document, and another list which contains all unique shingles found in all documents."""

documents = ['Cosmic hourglass captured by the James Webb Space Telescope reveals birth of a star', 'The birth of a star is revealed by a cosmic hourglass captured by the James Webb Space Telescope', 'The James Webb Space Telescope captured a cosmic hourglass that revealed the birth of a star','Historic moon mission shares first images from space', 'First images from space from a historic moon mission', 'Amazon launches message-based virtual clinic for allergies acne and hair loss', 'Amazon has launched a virtual clinic that uses messages to treat allergies acne and hair loss','Amazon has launched a virtual clinic based on messages for allergies acne and hair loss', 'Scotland was just recognized as the worlds best golf destination', 'Scotland was recently named the best golf destination in the world','Nasa space telescope reveals celestial hourglass formed by embryonic star','NASAs space telescope has discovered a celestial hourglass formed by a developing star','NASAs space telescope reveals a celestial hourglass formed by an embryonic star', 'The missile strike has ignited visceral fear in Poland and poses hard questions for Nato', 'The missile strike has instilled fear in Poland and raises difficult questions for Nato', 'The missile strike has sparked widespread fear in Poland and raises difficult questions for Nato','Meteorite that landed in Cotswolds may solve mystery of Earths water','A meteorite that landed in the Cotswolds could help to solve the mystery of Earths water','Threatening art is no answer to the climate crisis', 'Threatening art is not a viable response to the climate crisis']

def kShingles(document,k=5):
  hashed_shingles = [hash(document[i:i + k]) for i in range(len(document) - k + 1)]
  return set(hashed_shingles)

def Shingling(documents, k=5):
  shingle_list = []
  shingles_total = []
  # Start the runtime
  t0 = time.time()
  for doc in documents:
    hashed_shingles = kShingles(doc,k)
    shingle_list.append(hashed_shingles)
    shingles_total += list(hashed_shingles)
  n_shingles = len(set(shingles_total))
  print('\n The total number of unique shingles found in ' + str(len(documents)) + ' documents is ' + str(n_shingles))
  print ('\n Shingling ' + str(len(documents)) + ' docs took %.2f sec.' % (time.time() - t0))
  return shingle_list,shingles_total

shingle_list, shingles_total = Shingling(documents, k=5)

"""Then, in order to compare these documents using their hashed shingles, we first define a function, CompareSets. The function receieves two sets of shingles belonging to two documents, and calculates the jaccard similarity between them. Then, function CompareDocs is defined to compare sets of shingles for all selected documents iteratively, to eliminate the need to call the CompareSets function for each pair of documents. The CompareDocs function works by receiving the list of all hashed shingle sets, and creates an upper triangular matrix that stores the jaccard similarity beteween each set of documents"""

# First we need to define the funcion that calculates the Jaccard similarity between pairs of shingle sets
def CompareSets(x,y):
  intersection = len(list(set(x).intersection(y)))
  union = (len(x) + len(y)) - intersection
  similarity = float(intersection) / union
  return similarity

def CompareDocs(shingle_list):
  t0 = time.time()
  jSim = np.zeros((len(shingle_list),len(shingle_list)))
  for i in range(len(shingle_list)-1):
    for j in range(i+1,len(shingle_list)):
      jSim[i,j] = CompareSets(shingle_list[i],shingle_list[j])
  print('\n Calculating the Jaccard similarity between ' + str(len(shingle_list)) + ' documents took %.2f sec.' % (time.time() - t0))    
  return jSim

jaccard_similarity = CompareDocs(shingle_list)
print('\n',jaccard_similarity)

sns.heatmap(jaccard_similarity,vmin=0, vmax=1,square=True)

plt.hist(jaccard_similarity,bins=10)
plt.show()

def similar(jaccard_similarity):
  for i in range(jaccard_similarity.shape[0]):
    for j in range(i+1,jaccard_similarity.shape[1]):
      if jaccard_similarity[i,j] >= 0.5 and i !=j:
        print('document '+ str(i+1)+ ' and document '+ str(j+1) + ' are similar.')

similar(jaccard_similarity)

"""2. While comparing documents using the jaccard similarity is easy, it does not scale well there is a huge number of documents. Instead, we can use an approximation of this method which speeds up the calculations considerably. The second method is MinHashing. MinHashing applies k independent hash functions of the form (a*x + b) % c to the set of hashed shingles of documents, and then chooses the minimum value of hashed element(shingle) and places it in the signature matrix. a and b are randomly chosen integers, less than than the value of x, and c is a prime number, greater than the value of x. The signature matrix shows the minimum hash function for each document across its columns, and for each different hash function across its rows.

The MinHashing function receives the list containing the shingles for each document, and the number k indicating the number of hash functions that should be applied to the shingles. a and b are defined as arrays of randomly chosen integers, and c is a prime number greater than 2**32-1.
"""

def MinHashing(shingle_list, k, max):

  signatures = np.zeros((k, len(shingle_list)))

  c = nextprime(max)

  t0 = time.time()  
  for i in range(k): 
    a = np.random.randint(1,max,dtype=np.int64)
    b = np.random.randint(1,max,dtype=np.int64)
    for shingles in shingle_list:
      minHash = min(map(lambda shingle: (shingle*a + b)%c, shingles))
      signatures[i, shingle_list.index(shingles)] = minHash  
  print('\n Calculating the Signature Matrix for ' + str(len(shingle_list)) + ' documents, with ' + str(k) + ' different hash functions took %.2f sec.' % (time.time() - t0))
  return signatures

signature_matrix = MinHashing(shingle_list, 100, max(shingles_total))

signature_matrix

def similar_sig(jaccard_similarity):
  for i in range(jaccard_similarity.shape[0]):
    for j in range(i+1,jaccard_similarity.shape[1]):
      if jaccard_similarity[i,j] >= 0.4 and i !=j:
        print('document '+ str(i+1)+ ' and document '+ str(j+1) + ' are similar.')

def compare_signatures(signature1, signature2, k):

    intersection = len(list(signature1.intersection(signature2)))
    return float(intersection) / k

n_doc=signature_matrix.shape[1]
simSig = np.eye(n_doc,n_doc)
for i in range(n_doc):
    for j in range(i+1, n_doc):
      simSig[i,j] = compare_signatures(set(signature_matrix[:,i]),set(signature_matrix[:,j]), 100)

print(simSig)

sns.heatmap(simSig,vmin=0, vmax=1,square=True)

"""3. The idea behind LSH is that instead of comparing every pair of elements, we could just hash them into buckets, and hopefully elements that map to the same buckets will be the right level of “close” to each other.

The bucket_list function receives the signature matrix, and the number of rows that we want to put in the same band as integer r, and returns a dictionary named bucket_list, which maps the hash of column vectors of each band (keys) to the list of documents (values).
"""

def bucket_list(signature_matrix, r, hash_f=None):  
    n = signature_matrix.shape[0]    # n: length of a document signature
    b = n//r                         # b: number of bands
    buckets_list = [dict() for i in range(b)]

    if hash_f==None:
        hash_f = hash

    for i in range(0, n-r+1, r):
        band = signature_matrix.loc[i:i+r-1,:]
        band_hashing(band, hash_f, buckets_list[int(i/r)])

    return buckets_list

"""The band Hashing document receives a band and hashes it, with respect to its position, and puts it in the bucket dictionar."""

def band_hashing(band, hash_f, buckets_dict):
  for col in band.columns:
    h = hash_f(tuple(band[col].values))
    if h in buckets_dict: 
      buckets_dict[h].append(col)
    else: 
      buckets_dict[h] = [col]

"""The query band hashing function receives a band of query document and hashes it, and returns a list of hashes."""

def query_band_hashing(band, hash_f):

    hash_list = []
    h = hash_f(tuple(band.values))
    hash_list.append(h)
    
    return hash_list

"""The find similar docs takes the bucket lists and returns a set containing similar documents to given document."""

def find_similar_docs(doc_id, buckets_list, sign_mat, r, hash_f=None):
    
    # b: number of bands
    # n: length of a document signature
    # r: number of rows in a band
    
    n = sign_mat.shape[0]
    b = n//r

    if hash_f==None:
        hash_f = hash
    
    query_bucket_list = []

    for i in range(0, n-r+1, r):
        band = sign_mat.loc[i:i+r-1, int(doc_id)]
        query_bucket_list.append(query_band_hashing(band, hash_f))
    
    similar_docs = set()
    for i in range(len(query_bucket_list)):
        for j in range(len(query_bucket_list[i])):
            similar_docs.update(set(buckets_list[i][query_bucket_list[i][j]]))

    return similar_docs

sign_mat=pd.DataFrame(signature_matrix)
b_l=bucket_list(sign_mat, 5)
find_similar_docs(18, b_l, sign_mat, 5)

documents[18]

documents[19]

bucket = bucket_list(sign_mat, 2, hash_f=None)

bucket

similar_documents = find_similar_docs(6, bucket, sign_mat, 2, hash_f=None)

similar_documents

"""4. Comparison of Execution Time

In this section, we define a list with the number of documents, 20, 50, 100, 500, 1000 in order to compare the execution time for Shingling, MinHashing, and LSH methods.
"""

doc_size=[20,50,100,500,1000]
time_kshiling=[]
time_jaccard=[]
time_minhashing=[]
time_lsh_bucket=[]
time_jaccard=[]
for i in doc_size:
    print(i)
    random.seed(42)
    documents=random.sample(list(df['content'].dropna().drop_duplicates()),i)
    t0=time.time()
    shingles_list, shingles_total = Shingling(documents, k=3)
    time_kshiling.append(time.time() - t0)
    t0=time.time()
    jaccard_similarity = CompareDocs(shingles_list)
    time_jaccard.append(time.time() - t0)
    t0=time.time()
    signatures = MinHashing(shingles_list, 100)
    time_minhashing.append(time.time() - t0)

    t0=time.time()
    sign_mat=pd.DataFrame(signatures)
    b_l=bucket_list(sign_mat, 2)
    time_lsh_bucket.append(time.time() - t0)

doc_size=[20,50,100,500,1000]
time_kshiling=[]
time_jaccard=[]
time_minhashing=[]
time_lsh_bucket=[]
time_jaccard=[]
for i in doc_size:
    print(i)
    random.seed(42)
    documents=random.sample(list(df['content'].dropna().drop_duplicates()),i)
    t0=time.time()
    shingles_list, shingles_total = Shingling(documents, k=3)
    time_kshiling.append(time.time() - t0)
    t1=time.time()
    jaccard_similarity = CompareDocs(shingles_list)
    time_jaccard.append(time.time() - t1)
    t2=time.time()
    signatures = MinHashing(shingles_list, 100)
    time_minhashing.append(time.time() - t2)

    t3=time.time()
    sign_mat=pd.DataFrame(signatures)
    b_l=bucket_list(sign_mat, 2)
    time_lsh_bucket.append(time.time() - t3)

import matplotlib.pyplot as plt

plt.plot(doc_size,time_kshiling, label='Shingling')
plt.plot(doc_size, time_minhashing,label='Min Hashing')
plt.plot(doc_size, time_lsh_bucket,label='LSH Bucket')
plt.title('Comparison')
plt.ylabel('Time (in s)')
plt.xlabel('Number of Documents')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

plt.plot( time_kshiling,doc_size, label='Shingling')
plt.plot(time_minhashing,doc_size, label='Min Hashing')
plt.plot(time_lsh_bucket,doc_size, label='LSH Bucket')
plt.title('Comparison')
plt.ylabel('Time (in s)')
plt.xlabel('Number of Documents')
plt.legend()
plt.show()

plt.plot( time_kshiling,doc_size, label='Shingling')
plt.plot(time_minhashing,doc_size, label='Min Hashing')
plt.plot(time_lsh_bucket,doc_size, label='LSH Bucket')
plt.title('Comparison')
plt.xlabel('Time (in s)')
plt.ylabel('Number of Documents')
plt.legend()
plt.show()

