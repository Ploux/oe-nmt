# Introduction

Old English is the earliest form of the English language, spoken in England and southern and eastern Scotland in the early Middle Ages. It was brought to England by Anglo-Saxon settlers in the mid 5th century. The Norman Conquest of 1066 saw the replacement of Old English by Middle English, the language used by Chaucer. Modern English evolved from Middle English in the late 15th century. Shakespeare's plays, which many modern readers have difficulty understanding, are written in Early Modern English. Old English is for the most part unintelligible to modern English speakers, but it is still studied by linguists and historians.

Old English was originally written in a runic alphabet, however the Anglo-Saxons later adopted the Latin alphabet. Six letters do not exist in Old English, except potentially as loans; these are j, w, v, k, q, and z. However, Old English has four letters which do not occur in Modern English: æ (ash), þ (thorn), ð (eth), and ƿ (wynn).

For many years old english is translated by hand. This is a time consuming process, and the quality of the translation is not always good. We propose a neural machine translation system to translate Old English into Modern English. We begin with a toy dataset of 385 sentence pairs, and expand it to 1000 sentence pairs. We use three different models to translate Old English into Modern English. The first model is an RNN. The second model is similar, with the addition of an attention layer. The third model is a transformer. We compare the results of the three models, and find that the transformer model performs the best.

## Corpus

There are approximately 3000 surviving Old English texts [penn state], totalling about 3 million words [rutgers]. From a database, it is estimated that there are 15,000 distinct word roots in the language. When conjugations, plurals and spelling variants are accounted for, this number rises to 130,000 total words.

We implement three different versions of neural machine translation to translate Old English into Modern English






# Related Work




# Proposed Method

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

# Code
(codes and datasets should be open to the public, with separate files or GitHub repository)