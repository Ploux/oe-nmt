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

The attention layer was added, following the guidance of [NITISH put your reference here] A helpful description of attention used in the presentation was found in [nmt]

A number of github projects dealing the conjunction of NLP and Old English were discovered, in addition to the sources previously mentioned. These are [git1, git2, git3, git4, git5]

A transformer model for translation was implemented using code from [building transformers book]

Texts used for the training, test and validation sets were obtained from [homilies, oeinfo]

Of course, this section would not be complete without mention of the original transformer paper [transformer paper]. The transformer is a neural network architecture which has revolutionized the field of natural language processing. It is a self-attention mechanism which allows the model to learn long-range dependencies. The transformer has been applied to a wide variety of tasks, including machine translation, text generation, and image captioning.

 # Proposed Method

We propose to compare three different methods of neural machine translation of Old English into Modern English. The first model uses an RNN (Recurrent Neural Network). The second model is similar, with the addition of an attention layer. The third model is a transformer. 

## Dataset
We begin with a toy dataset of 385 sentence pairs, and expand it to 1000 sentence pairs. The initial dataset came from the Homilies of Aelfric [homilies]. This dataset has several

## Preprocessing
## Models
1. The RNN model

2. The RNN with attention model
   
3. The transformer model

## Parameters, Epochs, Etc

## Evaluation Procedure



# Results



# Conclusions

size of dataset

which model worked best

# Contributions of Members

Both members of the group worked equally on the code. Nitish developed the version with added attention layer, while Peter focused on the transformer version. Additionally Nitish is to be credited with standardizing the three programs, generating usable testing output, and implementing features such as the early stopping monitor. Peter, due to his prior familiarity with Old English, was responsible for expanding the dataset. Both group members also collaborated equally in terms of experimental design and analysis, writing this report, and preparing the presentation.

# Code

The code for this project is available at https://github.com/Ploux/oe-nmt. Three 



# References

[git1] https://github.com/chadmorgan/OldEnglishPoetryCorpus
[git2] https://github.com/iafarhan/Machine-Translation-for-Endangered-Language-Revitalization
[git3] https://github.com/nbsnyder/OldEnglishNLP
[git4] https://github.com/sharris-umass/oenouns
[git5] https://github.com/old-english-learner-texts/old-english-texts


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

[large scale] Neidorf, L., Krieger, M. S., Yakubek, M., Chaudhuri, P., & Dexter, J. P. (2019). Large-scale quantitative profiling of the Old English verse tradition. Nature Human Behaviour. doi:10.1038/s41562-019-0570-1 