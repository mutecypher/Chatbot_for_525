from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
import pickle
import json
import random
from tensorflow.keras.optimizers import SGD, Adadelta
from tensorflow.keras.models import Sequential
import numpy as np
##import tensorflow
import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = []
labels = []
docs = []
ignore_list = ['?', '!']

dataset = open('mo_intents_to_add.json').read()
##dataset = open('tiny.json').read()
intents = json.loads(dataset)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize each word
        word_token = nltk.word_tokenize(pattern)
        words.extend(word_token)
        # add documents in the corpus
        docs.append((word_token, intent['tag']))

        # add to our labels list
        if intent['tag'] not in labels:
            labels.append(intent['tag'])


# lemmatize each word, and sort words by removing duplicates:
words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in ignore_list]
words = sorted(list(set(words)))
# sort labels:
labels = sorted(list(set(labels)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))


# creating our training data:
training_data = []
# creating an empty array for our output (with size same as length of labels):
output = [0]*len(labels)

for doc in docs:
    bag_of_words = []
    pattern_words = doc[0]
    # lemmatize pattern words:
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]

    for w in words:
        if w in pattern_words:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)

    output_row = list(output)
    output_row[labels.index(doc[1])] = 1

    training_data.append([bag_of_words, output_row])

# convert training_data to numpy array and shuffle the data:
random.shuffle(training_data)
training_data = np.array(training_data)

# Now we have to create training list:
x_train = list(training_data[:, 0])
y_train = list(training_data[:, 1])


# Creating Model:

model = Sequential()
model.add(Dense(256, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(len(y_train[0]), activation='softmax'))
model.summary()

sgd_optimizer = SGD(learning_rate=0.02, decay=1e-6,
                    momentum=0.9, nesterov=True)
addy = Adadelta(learning_rate=0.005, rho=0.96, epsilon=1e-6)

model.compile(loss='categorical_crossentropy',
              # optimizer=addy, metrics=['accuracy'])
              optimizer=sgd_optimizer, metrics=['accuracy'])

# fit the model
history = model.fit(np.array(x_train), np.array(y_train),
                    epochs=100, batch_size=12)
# , verbose=1)

# fit the model

model.save("chatbot_Application_model.h5")
# print(model_json)
print("saved model")


print("done now")

model_file = 'chatbot_Application_model.h5'

model = load_model(model_file)


intents = json.loads(open('mo_intents_to_add.json').read())
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))


def bank_of_words(s, words, show_details=True):
    bag_of_words = [0 for _ in range(len(words))]
    sent_words = nltk.word_tokenize(s)
    sent_words = [lemmatizer.lemmatize(word.lower()) for word in sent_words]
    for sent in sent_words:
        for i, w in enumerate(words):
            if w == sent:
                bag_of_words[i] = 1
    return np.array(bag_of_words)


def predict_label(s, model):
    # filtering out predictions
    pred = bank_of_words(s, words, show_details=False)
    response = model.predict(np.array([pred]))[0]
    ERROR_THRESHOLD = 0.25
    final_results = [[i, r]
                     for i, r in enumerate(response) if r > ERROR_THRESHOLD]
    final_results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in final_results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list


def Response(ints, intents_json):
    tags = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tags):
            response = random.choice(i['responses'])
            break
        else:
            response = "I am not sure I understand, please try something else."
    return response


def chatbot_response(msg):
    ints = predict_label(msg, model)
    response = Response(ints, intents)
    return response


def chat():
    print("Begin your conversation with Mike's ChatBot")
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break
        response = chatbot_response(inp)
        print("\n BOT: " + response + '\n\n')


chat()
