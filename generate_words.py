"""
Generate word files from NLTK corpus
"""

import nltk
from nltk.corpus import words
import random

# Download word list (one-time)
try:
    nltk.data.find('corpora/words')
except:
    nltk.download('words')

# Get all English words
english_words = words.words()

# Filter to only lowercase, alphabetic words
clean_words = [w.lower() for w in english_words if w.isalpha() and w.islower()]

# Remove duplicates and sort
clean_words = sorted(set(clean_words))

print(f"Total clean words available: {len(clean_words)}")

# Generate 1K words
words_1k = random.sample(clean_words, min(1000, len(clean_words)))
with open('data/words_1k.txt', 'w') as f:
    for word in sorted(words_1k):
        f.write(word + '\n')
print("✅ Created data/words_1k.txt")

# Generate 10K words
words_10k = random.sample(clean_words, min(10000, len(clean_words)))
with open('data/words_10k.txt', 'w') as f:
    for word in sorted(words_10k):
        f.write(word + '\n')
print("✅ Created data/words_10k.txt")

# Generate 100K words (use all if less than 100K)
if len(clean_words) >= 100000:
    words_100k = random.sample(clean_words, 100000)
else:
    words_100k = clean_words
    
with open('data/words_100k.txt', 'w') as f:
    for word in sorted(words_100k):
        f.write(word + '\n')
print(f"✅ Created data/words_100k.txt ({len(words_100k)} words)")

print("\n🎉 All word files generated!")