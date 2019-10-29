import numpy as np
import gensim
import gensim.downloader as api
import csv
import random
from collections import defaultdict

SPLIT_PERCENT = .8 
random.seed(762562)


# LOAD THE EMBEDDING MODEL
print("Loading model...")
#model = api.load('word2vec-google-news-300')
model = api.load('fasttext-wiki-news-subwords-300')
print("done")


# COUNT OCCURENCES OF CLASSES TO CREATE TRAIN/TEST SPLIT
print("Counting class occurence...")
cleaned_data_raw = open('cleaned_data.csv', 'r')
csv_reader = csv.reader(cleaned_data_raw, delimiter=',')
counts = defaultdict(int)
for row in csv_reader:
    words = row[2].split()
    if len(words) < 100:
        continue
    counts[row[1]] += 1

print(counts)
cleaned_data_raw.seek(0)
print('done')


# CREATING DATA SPLIT
print("Splitting data...")
train_counts = {}
test_counts = {}
for key, val in counts.items():
    train_counts[key] = int(val * SPLIT_PERCENT)
    test_counts[key] = val - train_counts[key]
print(train_counts)
print(test_counts)
print('done')

# DEFINE FUNCTION TO MIMIC .get()
def get_vector(model, word):
    word = word.replace("'", "")
    if word not in model:
        return model['UNK']
    return model[word]

# CREATE IMAGES AND SAVE TO FILES
classes = {}
class_index = 0
train_labels = []
test_labels = []

print('Creating dataset...')
train_idx = 0
test_idx = 0
for row in csv_reader:
    words = row[2].split()
    if len(words) < 100:
        continue

    # GET CLASS INDEX
    genre = row[1]
    if genre not in classes:
        classes[genre] = class_index
        class_index += 1

    # GET COUNTS FOR SPLIT
    split = 'train' if random.random() <= 0.8 and train_counts[genre] > 0 else 'test'
    if split == 'train':
        train_labels.append(classes[genre])
        train_counts[genre] -= 1
        idx = train_idx
        train_idx += 1
    else:
        test_labels.append(classes[genre])
        test_counts[genre] -= 1
        idx = test_idx
        test_idx += 1

    # CREATE IMAGE ARRAY
    image = np.array([get_vector(model, w) for w in words])
    # image = np.expand_dims(image, axis=0)
    np.save('data/' + split + '/img_' + str(idx), image)

np.save('data/train/labels', np.array(train_labels))
np.save('data/test/labels', np.array(test_labels))
print(classes)
print("DONE!")
