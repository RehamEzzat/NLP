import pandas as pd
import numpy as np
import csv
from itertools import zip_longest
import nltk
import re
from nltk import ngrams

def remove_punctuation(words_list):
    regex = "[.)(_-،('')]"
    pure_words_list = []
    for word in words_list:
        m = re.match(regex, word)
        if m is None:
            pure_words_list.append(word)
    return pure_words_list


# path = "corpus.txt"
# f= open(path, encoding="utf-8")
# corpus = f.read()

corpus = '''من بين النباتات التي تزدهر في فصل الخريف نبتة الزعرور، فهذه النبتة ليست جميلة المظهر فقط بل تستعمل في العديد من الأغراض
 العلاجية و لها فوائد متعددة أيضا. من بين النباتات التي تزدهر في فصل الخريف نبتة الزعرور، فهذه النبتة ليست جميلة المظهر فقط بل تستعمل في العديد من الأغراض
 العلاجية و لها فوائد متعددة أيضا. جميلة المظهر النبتة. عرف الإنسان قيمة الأعشاب مند القدم وأصبح لا يستغني عنها لعلاج الأمراض التي تصيبه، ورغم
  التقدم الذي عرفته مجالات طبية عديدة إلا أن استخدام هذه الأعشاب لازال حاضرا بقوة في حياتنا اليومية و بشكل كبير.
   و من بين هذه الأعشاب، نبتة "الزعرور" التي تنتمي إلى الفصيلة الوردية و تميزها أشواك حادة تشكل نهايات فروعها، وتحمل أغصانها باقات أزهار
    بثمار كروية تميل إلى الحمرة تشبه في شكلها ثمار التفاح الصغيرة. و تتفتح أزهارها في شهر مايو و تبرز فيها عناقيد حمراء أو بيضاء
     أو زهرية زهرية تنضج في بدايات الخريف.'''

sentences = nltk.sent_tokenize(corpus)

all_words = []
trigrams = []
bigrams = []

for sentence in sentences:
    sentence_words = remove_punctuation(nltk.word_tokenize(sentence))
    trigrams += ngrams(sentence_words, 3)
    bigrams += ngrams(sentence_words, 2)
    all_words += nltk.word_tokenize(sentence)

distinct_words = list(set(all_words))
# distinct_trigrams = list(set(trigrams))
distinct_bigrams = list(set(bigrams))

data = pd.DataFrame(0, index=distinct_bigrams, columns=distinct_words)

for item in trigrams:
    index = [distinct_bigrams.index(t) for t in distinct_bigrams if t[0] == item[0] and t[1] == item[1]][0]
    data[item[2]][index] += 1

data.to_csv("counter.csv")

