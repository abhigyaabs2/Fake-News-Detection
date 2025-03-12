#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')


# In[6]:


df_fake['label'] = 0 
df_real['label'] = 1


# In[7]:


df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)


# In[8]:


df = df[['title', 'text', 'label']]
df.dropna(inplace=True)
print(df.head())


# In[9]:


import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


stop_words = set(stopwords.words('english'))


# In[12]:


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


# In[13]:


df['text'] = df['text'].apply(preprocess_text)


# In[14]:


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[18]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# In[19]:


import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))


# In[ ]:




