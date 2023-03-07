```python
import pandas as pd
import numpy as np
import keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense
```


```python
data=pd.read_csv("train.txt",sep=';')
```


```python
data.columns=["Text","Emotions"]
```


```python
print(data.head(25))
```

                                                     Text  Emotions
    0   i can go from feeling so hopeless to so damned...   sadness
    1    im grabbing a minute to post i feel greedy wrong     anger
    2   i am ever feeling nostalgic about the fireplac...      love
    3                                i am feeling grouchy     anger
    4   ive been feeling a little burdened lately wasn...   sadness
    5   ive been taking or milligrams or times recomme...  surprise
    6   i feel as confused about life as a teenager or...      fear
    7   i have been with petronas for years i feel tha...       joy
    8                                 i feel romantic too      love
    9   i feel like i have to make the suffering i m s...   sadness
    10  i do feel that running is a divine experience ...       joy
    11  i think it s the easiest time of year to feel ...     anger
    12                 i feel low energy i m just thirsty   sadness
    13  i have immense sympathy with the general point...       joy
    14    i do not feel reassured anxiety is on each side       joy
    15               i didnt really feel that embarrassed   sadness
    16            i feel pretty pathetic most of the time   sadness
    17  i started feeling sentimental about dolls i ha...   sadness
    18  i now feel compromised and skeptical of the va...      fear
    19  i feel irritated and rejected without anyone d...     anger
    20  i am feeling completely overwhelmed i have two...      fear
    21    i have the feeling she was amused and delighted       joy
    22  i was able to help chai lifeline with your sup...       joy
    23  i already feel like i fucked up though because...     anger
    24  i still love my so and wish the best for him i...   sadness
    


```python
texts=data["Text"].tolist()
labels=data["Emotions"].tolist()

tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts)
```


```python
sequences=tokenizer.texts_to_sequences(texts)
max_length=max([len(seq) for seq in sequences])
padded_sequences=pad_sequences(sequences,maxlen=max_length)
```


```python
label_encoder=LabelEncoder()
labels=label_encoder.fit_transform(labels)
```


```python
one_hot_labels=keras.utils.to_categorical(labels)
```


```python
xtrain,xtest,ytrain,ytest=train_test_split(padded_sequences,one_hot_labels,
                                          test_size=0.2)
```


```python
model=Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                   output_dim=128,input_length=max_length))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=len(one_hot_labels[0]),activation='softmax'))
```


```python
model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=10,batch_size=32,validation_data=(xtest,ytest))
```

    Epoch 1/10
    400/400 [==============================] - 13s 32ms/step - loss: 1.3558 - accuracy: 0.4713 - val_loss: 0.8977 - val_accuracy: 0.6894
    Epoch 2/10
    400/400 [==============================] - 13s 32ms/step - loss: 0.3476 - accuracy: 0.8943 - val_loss: 0.5827 - val_accuracy: 0.8169
    Epoch 3/10
    400/400 [==============================] - 13s 33ms/step - loss: 0.0587 - accuracy: 0.9854 - val_loss: 0.6271 - val_accuracy: 0.8206
    Epoch 4/10
    400/400 [==============================] - 13s 32ms/step - loss: 0.0243 - accuracy: 0.9952 - val_loss: 0.6647 - val_accuracy: 0.8206
    Epoch 5/10
    400/400 [==============================] - 13s 32ms/step - loss: 0.0170 - accuracy: 0.9968 - val_loss: 0.6706 - val_accuracy: 0.8206
    Epoch 6/10
    400/400 [==============================] - 13s 32ms/step - loss: 0.0140 - accuracy: 0.9974 - val_loss: 0.7030 - val_accuracy: 0.8188
    Epoch 7/10
    400/400 [==============================] - 13s 32ms/step - loss: 0.0126 - accuracy: 0.9971 - val_loss: 0.7156 - val_accuracy: 0.8125
    Epoch 8/10
    400/400 [==============================] - 14s 35ms/step - loss: 0.0113 - accuracy: 0.9975 - val_loss: 0.7814 - val_accuracy: 0.8087
    Epoch 9/10
    400/400 [==============================] - 17s 42ms/step - loss: 0.0111 - accuracy: 0.9973 - val_loss: 0.7969 - val_accuracy: 0.8078
    Epoch 10/10
    400/400 [==============================] - 16s 40ms/step - loss: 0.0092 - accuracy: 0.9974 - val_loss: 0.8108 - val_accuracy: 0.8125
    




    <keras.callbacks.History at 0x212ba9b2650>




```python
input=input("Enter your feelings:")
input_sequence=tokenizer.texts_to_sequences([input])
padded_input_sequence=pad_sequences(input_sequence,maxlen=max_length)
prediction=model.predict(padded_input_sequence)
prediction_label=label_encoder.inverse_transform([np.argmax(prediction[0])])
print(prediction_label)
```
