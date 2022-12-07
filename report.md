# Introduction

Old English is the earliest form of the English language, spoken in England and southern and eastern Scotland in the early Middle Ages. It was brought to England by Anglo-Saxon settlers in the mid 5th century. The Norman Conquest of 1066 saw the replacement of Old English by Middle English, the language used by Chaucer. Modern English evolved from Middle English in the late 15th century. Shakespeare's plays, which many modern readers have difficulty understanding, are written in Early Modern English. Old English is for the most part unintelligible to modern English speakers, but it is still studied by linguists and historians.

Old English was originally written in a runic alphabet, however the Anglo-Saxons later adopted the Latin alphabet. Six letters do not exist in Old English, except potentially as loans; these are j, w, v, k, q, and z. However, Old English has four letters which do not occur in Modern English: æ (ash), þ (thorn), ð (eth), and ƿ (wynn). There are approximately 3000 surviving Old English texts [penn state], totalling about 3 million words [rutgers]. From a database, it is estimated that there are 15,000 distinct word roots in the language. When conjugations, plurals and spelling variants are accounted for, this number rises to 130,000 total words. Although some words have survived into Modern English, most have not. The grammar of Old English is most similar to German or Icelandic. An example of an archaic feature of Old English is the inflection of nouns for case. Old English nouns have five cases (nominative, accusative, genitive, dative, instrumental), which, accounting for singular and plural forms, means each noun may have up to ten different spellings. Additionally nouns are gendered, like many modern European languages. These grammatical features make translation of Old English into Modern English more difficult than performing direct word-to-word replacement.

Many Old English texts have been lost, and many of the surviving texts are fragmentary or badly damaged. These texts have been digitized and transcribed and are generally available in the public domain in their untranslated form. For many years Old English has been translated by hand. Peter, one of the authors, has a Masters of Fine Arts in Writing, and has taken graduate coursework in Old English. At the time (2011), the Old English manuscripts were printed in books. The translator would photocopy or type the original manuscript, look up each word individually, and write the definition or definitions above it. Then from this, the most likely interpretation of the sentence would be written in Modern English. Peter used this process to translate Beowulf, and although rewarding, it was labor-intensive and tedious. Today, online texts are available, as well as online dictionaries are available [oe translator], although the latter can only process one word at a time, requiring the translator to still be familiar with Old English grammar and syntax. As of 2022, Old English is not offered as a language option on sites such as Google Translate [google translate]. The Old English corpus includes religious texts, legal documents, histories, scientific texts, riddles, and poetry, most famously the epic poem Beowulf. Currently, only dedicated scholars are able to decipher these texts, and casual readers must either purchase professional translations, or rely on translations available online, which is incomplete, and of varying quality. 

We propose a neural machine translation system to translate Old English into Modern English. This would have the benefit of bringing Old English to a much wider audience. As it is, the material which survives from this historical period is sparse. A large percentage of the United States population has at least one ancestor of English or Northern European stock, and this project is a step towards enabling them to unlock their cultural heritage and better understand the ideas, history, and lifestyles of their ancestors.








# Related Work




# Proposed Method


 We propose a neural machine translation system to translate Old English into Modern English. We begin with a toy dataset of 385 sentence pairs, and expand it to 1000 sentence pairs. We use three different models to translate Old English into Modern English. The first model uses an RNN (Recurrent Neural Network). The second model is similar, with the addition of an attention layer. The third model is a transformer. We compare the results of the three models, and find that the transformer model performs the best.



We implement three different versions of neural machine translation to translate Old English into Modern English
a separate independent validation set is created for testing.

# Experiments 

# Conclusions

# Contributions of Members



# References

building transformers book

[machine learning webpage](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)

[github project](https://github.com/kel-c-lm/translator)

wikipedia old english page [wikipedia] 

https://open.psu.edu/databases/psu00859 [penn state]

https://www.libraries.rutgers.edu/databases/dictionary-old-english-web-corpus [rutgers]

[database]https://github.com/iggy12345/OE_Sentence_Generator

[google translate] https://translate.google.com/

oe translator https://www.oldenglishtranslator.co.uk/

# Code
(codes and datasets should be open to the public, with separate files or GitHub repository)