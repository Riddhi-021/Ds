#!/usr/bin/env python
# coding: utf-8

# In[8]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[11]:


document = "This is an example document for tokenization. , This is an example document for POS tagging ,stemming"


# In[13]:


#Document Preprocessing
tokens = word_tokenize(document)
print(tokens)


# In[15]:


#POS Tagging
pos_tags = pos_tag(tokens)
print(pos_tags)


# In[19]:


#Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print(stemmed_tokens)


# In[24]:


#Lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print(lemmatized_tokens)


# In[ ]:





# In[ ]:




