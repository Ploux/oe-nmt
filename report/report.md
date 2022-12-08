# Abstract




# Introduction

Old English is the earliest form of the English language, spoken in England and southern and eastern Scotland in the early Middle Ages. It was brought to England by Anglo-Saxon settlers in the mid 5th century. The Norman Conquest of 1066 saw the replacement of Old English by Middle English, the language used by Chaucer. Modern English evolved from Middle English in the late 15th century. Shakespeare's plays, which many modern readers have difficulty understanding, are written in Early Modern English. Old English is for the most part unintelligible to modern English speakers, but it is still studied by linguists and historians [wikipedia]

Old English was originally written in a runic alphabet, however the Anglo-Saxons later adopted the Latin alphabet. Six letters do not exist in Old English, except potentially as loans; these are j, w, v, k, q, and z. However, Old English has four letters which do not occur in Modern English: æ (ash), þ (thorn), ð (eth), and ƿ (wynn) [wikipedia]. There are approximately 3000 surviving Old English texts [penn state], totalling about 3 million words [rutgers]. From a database, it is estimated that there are 15,000 distinct word roots in the language. When conjugations, plurals and spelling variants are accounted for, this number rises to 130,000 total words. Although some words have survived into Modern English, most have not. The grammar of Old English is most similar to German or Icelandic. An example of an archaic feature of Old English is the inflection of nouns for case. Old English nouns have five cases (nominative, accusative, genitive, dative, instrumental), which, accounting for singular and plural forms, means each noun may have up to ten different spellings. Additionally nouns are gendered, like many modern European languages. [wiki-grammar] These grammatical features make translation of Old English into Modern English more difficult than performing direct word-to-word replacement.

Many Old English texts have been lost, and many of the surviving texts are fragmentary or badly damaged. These texts have been digitized and transcribed and are generally available in the public domain in their untranslated form. For many years Old English has been translated by hand. Peter, one of the authors, has a Masters of Fine Arts in Writing, and has taken graduate coursework in Old English. At the time (2011), the Old English manuscripts were printed in books. The translator would photocopy or type the original manuscript, look up each word individually, and write the definition or definitions above it. Then from this, the most likely interpretation of the sentence would be written in Modern English. Peter used this process to translate Beowulf, and although rewarding, it was labor-intensive and tedious. Today, online texts are available, as well as online dictionaries are available [oe translator], although the latter can only process one word at a time, requiring the translator to still be familiar with Old English grammar and syntax. As of 2022, Old English is not offered as a language option on sites such as Google Translate [google translate]. The Old English corpus includes religious texts, legal documents, histories, scientific texts, riddles, and poetry, most famously the epic poem Beowulf [wiki-lit]. Currently, only dedicated scholars are able to decipher these texts, and casual readers must either purchase professional translations, or rely on translations available online, which is incomplete, and of varying quality. 

We propose a neural machine translation system to translate Old English into Modern English. This would have the benefit of bringing Old English to a much wider audience. As it is, the material which survives from this historical period is sparse. A large percentage of the United States population has at least one ancestor of English or Northern European stock, and this project is a step towards enabling them to unlock their cultural heritage and better understand the ideas, history, and lifestyles of their ancestors.
# Related Work

One of the few recent works concerning both machine learning and Old English is [large scale]. Although direct translation is not attempted, the authors use quantitative profiling to pconfirm the single authorship of Beowulf, as well as the attributing the authorship of several other poems to Cynewulf, two topics that have been debated by scholars for several hundred years.

Attempts have been made to decipher ancient languages by comparing them to similar languages. [deciphering] proposed using Basque to interpret Iberian, a lost and hitherto undecipherable language. [neural] While Old English is not a lost language, the paucity of sentence pairs, led us to consider this approach. 

Other experiments in low-resource neural machine translation include [revisiting], [word translation], [phase-based]

A number of works deal with the preservation of endangered languages , for instance Cherokee [cherokee], rare Eastern European languages [only chance], or Estonian and Slovak [trivial]

Our initial RNN model is based on an English-German translation model developed by Jason Brownlee [machine learning webpage]. This model was further adapted to Old English [github]. 

The attention layer was added, following the guidance of [attention] A helpful description of attention used in the presentation was found in [nmt]

A number of github projects dealing the conjunction of NLP and Old English were discovered, in addition to the sources previously mentioned. These are [git1, git2, git3, git4, git5]

A transformer model for translation was implemented using code from [building transformers book]

Texts used for the training, test and validation sets were obtained from [homilies, oeinfo]

Of course, this section would not be complete without mention of the original transformer paper [transformer paper]. The transformer is a neural network architecture which has revolutionized the field of natural language processing. It is a self-attention mechanism which allows the model to learn long-range dependencies. The transformer has been applied to a wide variety of tasks, including machine translation, text generation, and image captioning.

 # Proposed Method

We propose to compare three different methods of neural machine translation of Old English into Modern English. The first model uses an RNN (Recurrent Neural Network). The second model is similar, with the addition of an attention layer. The third model is a transformer. 

## Dataset
We begin with a toy dataset of 385 sentence pairs, and expand it to over 1000 sentence pairs. The initial dataset came from the Homilies of Aelfric [homilies]. The Homilies are chosen because they are in the public domain, and feature side-by-side Modern and Old English versions. They also have a single translator, Benjamin Thorpe, and a similar subject matter (religion) which lends consistency. However, the initial dataset was heavily simplified. 

[figure - old dataset]

This simplification led to inflated BLEU scores and facile results. A cursory inspection of this dataset shows that using it for training will produce a model capable of translating sentences from the original dataset, but not much else, including other sentences from the Homilies. 

[figure - six]

400,000 words is considered an absolute minimum for machine translation. The original dataset contains only 3,000. Our expanded dataset contains about 12,000. NMT begins to surpass SMT (Statistical Machine Translation) in performance at 15 million words. [six challenges]. While this information is from 2017, hence pre-Transformer models, it is clear that we are still sorely lacking in this department.

Expanding the dataset was a somewhat challenging process. Due to the extremely long sentences of the Homilies, we needed to break them into smaller independent clauses. This had to be done one at a time, manually, by someone with at least a passing familiarity with the language. 

[figure - new dataset]

Approximately 100 sentences of the new dataset are from a different source which also had side-by-side translation [oeinfo]. These sentences were chosen because they are in the public domain, and because they are from a variety of sources, including the Lord's Prayer to the Magna Carta, a text on the treatment of colds, and excerpts from the Anglo-Saxon Chronicle.

## Preprocessing

All three models use a similar preprocessing pipeline. The dataset begins in the format of Old English->tab->Modern English. Punctuation is removed, and all words are lowercased. The pickle API is then used to serialize the dataset. 

## Models
1. RNN model

[figure Plot-of-Model-Graph-for-NMT.png]

2. RNN with attention model

[figure attention]]
   
3. transformer model

[transformer - encoding figure]

[transformer - decoding figure]
## Parameters

[NITISH throw this into a table]

All models: 
optimizer - Adam
loss    - categorical crossentropy
initial learning rate - 0.1
epochs - 200 (with early stopping)
batch size - 64

Transformer model
Number of self-attention heads = 8
Dimensionality of the linearly projected queries and keys = 64
Dimensionality of the linearly projected values = 64
Dimensionality of model layers' outputs = 512
Dimensionality of the inner fully connected layer = 2048
Number of layers in the encoder stack = 6
Beta_1 = 0.9
Beta_2 = 0.98
Epsilon = 10^-9
Dropout rate = 0.1
Warmup steps = 4000


## Evaluation Procedure

The primary evaluation for the models was BLEU-1 score [bleu], which simply counts the number of words in the predicted sentence that are also in the target sentence. BLEU-2, BLEU-3, and BLEU-4, which count the number of word pairs, triplets, and quadruplets, respectively, were also used. Finally, the translation output was manually inspected for quality. This manual inspection proved important as a model can achieve a relatively high BLEU-1 score by a variety of deceptive means. For example, if the dataset consists of only sentences with the following repetitive pattern, by guessing a translation of "it is" for every sentence, the model will achieve a BLEU-1 score of approximately 0.6.

[figure - blue dataset] 


# Results

The three models are evaluated on the test set, which consists of 20% sentences randomly chosen from the dataset after the model has been trained on 80%. Both the original and expanded datasets were used. The results are shown in the table below.

[MAKE TABLE]

Model   Dataset   BLEU-1   BLEU-2   BLEU-3   BLEU-4
RNN     Original  0.25     0.11     0.05     0.02
RNN     Expanded  0.25     0.11     0.05     0.02

Attention Original  0.25     0.11     0.05     0.02
Attention Expanded  0.25     0.11     0.05     0.02

Transformer Original  0.25     0.11     0.05     0.02
Transformer Expanded  0.25     0.11     0.05     0.02

# Figures
(NITISH - put them like this with one page for original and one for expanded) 
p1  x x   p2 x x
    x x      x x
    x x      x x
## Original Dataset

RNN - training loss and validation loss.

Attention - training loss and validation loss.

Transformers - training loss and accuracy loss

## Expanded Dataset

RNN -  training loss and validation loss.

Attention - training loss and validation loss.

Transformers - training loss and accuracy loss


# Conclusions

Adding an attention layer showed little benefit from the RNN model. Although the transformer model performed more poorly than either on the original dataset, this dataset is of little serious use. Notably, the transformer model was the only one which could cope reasonably with the expanded dataset. This is likely due to the fact that the transformer model is designed to handle long sequences, and the expanded dataset contains many long sentences.

For practical application, expanding the size of the dataset far beyond even our extended version, as well as deriving sentences from a broad variety of sources is clearly necessary. 
# Contributions of Members

Both members of the group worked equally on the code. Nitish developed the version with added attention layer, while Peter focused on the transformer version. Additionally Nitish is to be credited with standardizing the three programs, generating usable testing output, and implementing features such as the early stopping monitor. Peter, due to his prior familiarity with Old English, was responsible for expanding the dataset. Both group members also collaborated equally in terms of experimental design and analysis, writing this report, and preparing the presentation.

# Code

The code for this project is available at https://github.com/Ploux/oe-nmt. The three NMT models are named translate.py, attention.py, and transformer.py. They require the corpus dataset, corpus.tsv in order to run. Additionally, the models are accessible at the following colab links, in which case the corpus will be automatically downloaded. A GPU runtime is recommended.

translate - (https://colab.research.google.com/drive/1SYTCc2L1kXTizm-SaeHoSbuC4t8x_9OD?usp=sharing)

attention - (https://colab.research.google.com/drive/162H4r-QJFdkRN6kPWlGpB48nl9VTTasB?usp=sharing)

transformer - https://colab.research.google.com/drive/1g9SCvSoQmHn28Niiqz06n9a0MJRs6jpa?usp=sharing


# References


[attention] https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/

[trivial] Trivial Transfer Learning for Low-Resource Neural Machine Translation
Tom Kocmi
Ondřej Bojar
Proceedings of the 3rd Conference on Machine Translation 2018
https://arxiv.org/abs/1809.00357

[transformer paper] Attention Is All You Need
Vaswani, Ashish, et al.
(https://arxiv.org/abs/1706.03762)

[phrase-based] Phrase-Based & Neural Unsupervised Machine Translation
Guillaume Lample, Myle Ott, Alexis Conneau, Ludovic Denoyer, Marc'Aurelio Ranzato
https://arxiv.org/abs/1804.07755
2018

[nmt] 2017
Statistical Machine Translation
Draft of Chapter 13: Neural Machine Translation
Philipp Koehn
https://arxiv.org/abs/1709.07809att

[building transformers book] Building Transformer Models with Attention, Stefania Cristina, Mehreen Saeed, 2022, MachineLearningMastery.com

[machine learning webpage](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)

[word translation] 
Word Translation Without Parallel Data
Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou
https://arxiv.org/abs/1710.04087
 	ICLR 2018

[wiki-grammar] https://en.wikipedia.org/wiki/Old_English_grammar

[homilies] aelfrics homilies from project gutenberg

[github project](https://github.com/kel-c-lm/translator)

[six challenges] Six Challenges for Neural Machine Translation
Philipp Koehn, Rebecca Knowles
First Workshop on Neural Machine Translation, 2017
https://arxiv.org/abs/1706.03872

[only chance] The only chance to understand: machine translation of the severely endangered low-resource languages of Eurasia
Anna Mosolova
Kamel Smaïli
Proceedings of the Fifth Workshop on Technologies for Machine Translation of Low-Resource Languages (LoResMT 2022)
October 2022,     
(https://aclanthology.org/2022.loresmt-1.4) 

[oeinfo] https://oldenglish.info/textselect.html

[wikipedia] old english page 

[bleu] https://en.wikipedia.org/wiki/BLEU
[wiki lit] old english literature

[revisiting] Revisiting Low-Resource Neural Machine Translation: A Case Study
Rico Sennrich, Biao Zhang
https://arxiv.org/abs/1905.11901
2019


[cherokee] ChrEn: Cherokee-English Machine Translation for Endangered Language Revitalization
Shiyue Zhang
Benjamin Frey
Mohit Bansal
https://arxiv.org/abs/2010.04791
2020

[deciphering]  Deciphering Undersegmented Ancient Scripts Using Phonetic Prior 
Jiaming Luo,
Frederik Hartmann,
Enrico Santus,
Regina Barzilay,
Yuan Cao
Transactions of the Association for Computational Linguistics (2021) 9: 69–81.
https://doi.org/10.1162/tacl_a_00354

[neural] Neural Decipherment via Minimum-Cost Flow: from Ugaritic to Linear B  https://arxiv.org/abs/1906.06718  Jiaming Luo, Yuan Cao, Regina Barzilay

https://open.psu.edu/databases/psu00859 [penn state]

https://www.libraries.rutgers.edu/databases/dictionary-old-english-web-corpus [rutgers]

[database]https://github.com/iggy12345/OE_Sentence_Generator

[google translate] https://translate.google.com/

oe translator https://www.oldenglishtranslator.co.uk/

[git1] https://github.com/chadmorgan/OldEnglishPoetryCorpus
[git2] https://github.com/iafarhan/Machine-Translation-for-Endangered-Language-Revitalization
[git3] https://github.com/nbsnyder/OldEnglishNLP
[git4] https://github.com/sharris-umass/oenouns
[git5] https://github.com/old-english-learner-texts/old-english-texts


[large scale] Neidorf, L., Krieger, M. S., Yakubek, M., Chaudhuri, P., & Dexter, J. P. (2019). Large-scale quantitative profiling of the Old English verse tradition. Nature Human Behaviour. doi:10.1038/s41562-019-0570-1 
