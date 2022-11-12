# from turtle import st
import nlpaug.augmenter.word as naw
import pandas as pd
from googletrans import Translator
import re
import urllib
import os
import zipfile
import tarfile
# import transformers
huggingface_token = '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'


def clean_it_up(text):
    texty_yo = re.sub(r'https?:\/\/.\S+', "", text)
    texty_yo = re.sub(r'#', '', texty_yo)
    texty_yo = re.sub(r'^RT[\s]+', '', texty_yo)
    texty_yo = re.sub(r'[^a-zA-Z ]+', '', texty_yo)
    texty_yo = re.sub(r' +', ' ', texty_yo)
    texty_yo = texty_yo.lower()
    return texty_yo


turnit = 4


def clean_output(text):
    cln = str(text)
    cln = cln.strip('[')
    cln = cln.strip(']')
    cln = cln.strip("'")
    return cln


f_name = input(
    "What is the name of the text file you would like to augment? :   ")
a_str = ''
f_name = str(f_name)
# f_name = a_str + f_name + a_str

twiggy = pd.read_csv(f_name, header='infer')
twiggy = twiggy.drop_duplicates()
twiggy = twiggy.drop_duplicates(subset=['text'])
text_in = twiggy['text']
# create a list with only the text input

text_in = text_in.to_list()

# Let's get a clean version of the text

clean_list = []
for a in range(len(text_in)):
    clean_version = clean_it_up(text_in[a])
    clean_version = clean_output(clean_version)
    clean_list.append(clean_version)

# Backward translation - English to Korean to Bengali to English
# using Google Translate
translator = Translator()
# for differing percentages of synonyms
stop_words = ['the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were']
syn_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.25, stopwords=stop_words)
mo_ant_aug = naw.AntonymAug(aug_p=0.5, stopwords=stop_words)
# aug_src = 'wordnet', aug_p = 0.20, stopwords = stop_words)

print()
print("Augment text by translating English to Korean to Bengali then back to English.\n")
print("This may take several minutes. \n")

syno_20 = []  # for the 20 percent synonyms
tranz = []  # for the backtranslations
syno_30 = []  # for the 30 percent synonyms


dummy_text = []

print("The first one is ", clean_list[0])
for i in range(len(clean_list)):
    if i % 10000 == 0:
        print("On text number ", i)
    texty_yo = clean_list[i]
    # k_trans = translator.translate(str(texty_yo), dest='ko')
    # b_trans = translator.translate(k_trans.text, src='ko', dest='bn')
    # e_trans = translator.translate(b_trans.text, src='bn', dest='en')
    # e_trans = clean_output(e_trans.text)
    auggy = syn_aug.augment(str(texty_yo), n=1)
    auggy = syn_aug.augment(str(auggy), n=1)
    mo_auggy = mo_ant_aug.augment(str(texty_yo), n=1)
    mo_auggy = mo_ant_aug.augment(str(mo_auggy), n=1)
    auggy = clean_output(auggy)
    mo_auggy = clean_output(mo_auggy)
    syno_20.append(auggy)
    # tranz.append(e_trans)
    dummy_text.append(texty_yo)
    syno_30.append(mo_auggy)

print()

# create sentences with more synonym substitution


# Create a dataframe with original, clean and augmented text
clean_file = pd.DataFrame({'original_text': dummy_text, 'clean_text': clean_list,
                           # 'translated': tranz,
                          'twenty_per_syn': syno_20, 'thirty_per_syn': syno_30})

print("example augmentations")
for i in range(len(clean_list)-3, len(clean_list)):
    print("The cleaned ", i, "th text is \n", clean_list[i])
    print()
   # print("The ", i, "th back translation is \n", tranz[i])
    # print()
    print("The ", i, "th twenty percent synonym sentence is \n", syno_20[i])
    print()
    print("The ", i, "th thirty percent synonym sentence is \n", syno_30[i])
    print()

clean_file.to_csv("cleaned_and_augmented.csv", header=True)
