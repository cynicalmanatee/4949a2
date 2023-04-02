import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import streamlit as st
import pandas as pd

st.title("***Mushroom Support:***")
st.header("Is your mushroom safe to eat?")
if "expanded" not in st.session_state:
    st.session_state["expanded"] = True
if "reset" not in st.session_state:
    st.session_state["reset"] = False

df = {
    'Cap Shape': ['Bell', 'Conical', 'Convex', 'Flat', 'Knobbed', 'Sunken'],
    'Cap Surface': ['Fibrous', 'Grooves', 'Scaly', 'Smooth'],
    'Cap Color': ['Brown', 'Buff', 'Cinnamon', 'Gray', 'Green', 'Pink', 'Purple', 'red', 'white', 'yellow'],
    'Bruises': ['Yes', 'No'],
    'Odor': ['Almond', 'Anise', 'Creosote', 'Fishy', 'Foul', 'Musty', 'None', 'Pungent', 'Spicy'],
    'Gill Attachment': ['Attached', 'Descending', 'Free', 'Notched'],
    'Gill Spacing': ['Close', 'Crowded', 'Distant'],
    'Gill Size': ['Broad', 'Narrow'],
    'Gill Color': ['Black', 'Brown', 'Buff', 'Chocolate', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow '],
    'Stalk Shape': ['Enlarging', 'Tapering'],
    'Stalk Root': ['Bulbous', 'Club', 'Cup', 'Equal', 'Rhizomorphs', 'Rooted', ',Missing'],
    'Stalk Surface Above Ring': ["Fibrous", "Scaly", "Silky", "Smooth"],
    'Stalk Surface Below Ring': ["Fibrous", "Scaly", "Silky", "Smooth"],
    'Stalk Color Above Ring': ['Brown', 'buff', 'Cinnamon', 'Gray', 'Green', 'Pink', 'Purple', 'red', 'white', 'yellow'],
    'Stalk Color Below Ring': ['Brown', 'buff', 'Cinnamon', 'Gray', 'Green', 'Pink', 'Purple', 'red', 'white', 'yellow'],
    'Veil Type': ["Partial", "Universal"],
    'Veil Color': ["Brown", "Orange", "White", "Yellow"],
    'Ring Number': ["None", "One", "Two"],
    'Ring Type': ["Evanescent", "Flaring", "Large", "None", "Pendant", "Sheathing", "Zone"],
    'Spore Print Color': ["Black", "Brown", "Buff", "Chocolate", "Green", "Orange", "Purple", "White", "Yellow"],
    'Population': ["Abundant", "Clustered", "Numerous", "Scattered", "Several", "Solitary"],
    'Habitat': ["Grasses", "Leaves", "Meadows", "Paths", "Urban", "Waste", "Woods"]
}

legends = {
    'Cap Shape': ['b', 'c', 'x', 'f', 'k', 's'],
    'Cap Surface': ['f', 'g', 'y', 's'],
    'Cap Color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
    'Bruises': ['t', 'f'],
    'Odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
    'Gill Attachment': ['a', 'd', 'f', 'n'],
    'Gill Spacing': ['c', 'w', 'd'],
    'Gill Size': ['b', 'n'],
    'Gill Color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
    'Stalk Shape': ['e', 't'],
    'Stalk Root': ['b', 'c', 'u', 'e', 'z', 'r', '?'],
    'Stalk Surface Above Ring': ["f", "y", "k", "s"],
    'Stalk Surface Below Ring': ["f", "y", "k", "s"],
    'Stalk Color Above Ring': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'red', 'w', 'y'],
    'Stalk Color Below Ring': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'red', 'w', 'y'],
    'Veil Type': ["p", "u"],
    'Veil Color': ["n", "o", "w", "y"],
    'Ring Number': ["n", "o", "t"],
    'Ring Type': ["e", "f", "l", "n", "p", "s", "z"],
    'Spore Print Color': ["k", "n", "b", "h", "r", "o", "u", "w", "y"],
    'Population': ["a", "c", "n", "s", "v", "y"],
    'Habitat': ["g", "l", "m", "p", "u", "w", "d"]
}

numerics = {
    "Cap Shape": {
        "b": "0",
        "c": "1",
        "f": "2",
        "k": "3",
        "s": "4",
        "x": "5"
    },
    "Cap Surface": {
        "f": "0",
        "g": "1",
        "s": "2",
        "y": "3"
    },
    "Cap Color": {
        "b": "0",
        "c": "1",
        "e": "2",
        "g": "3",
        "n": "4",
        "p": "5",
        "r": "6",
        "u": "7",
        "w": "8",
        "y": "9"
    },
    "Bruises": {
        "f": "0",
        "t": "1"
    },
    "Odor": {
        "a": "0",
        "c": "1",
        "f": "2",
        "l": "3",
        "m": "4",
        "n": "5",
        "p": "6",
        "s": "7",
        "y": "8"
    },
    "Gill Attachment": {
        "a": "0",
        "f": "1"
    },
    "Gill Spacing": {
        "c": "0",
        "w": "1"
    },
    "Gill Size": {
        "b": "0",
        "n": "1"
    },
    "Gill Color": {
        "b": "0",
        "e": "1",
        "g": "2",
        "h": "3",
        "k": "4",
        "n": "5",
        "o": "6",
        "p": "7",
        "r": "8",
        "u": "9",
        "w": "10",
        "y": "11"
    },
    "Stalk Shape": {
        "e": "0",
        "t": "1"
    },
    "Stalk Root": {
        "?": "0",
        "b": "1",
        "c": "2",
        "e": "3",
        "r": "4"
    },
    "Stalk Surface Above Ring": {
        "f": "0",
        "k": "1",
        "s": "2",
        "y": "3"
    },
    "Stalk Surface Below Ring": {
        "f": "0",
        "k": "1",
        "s": "2",
        "y": "3"
    },
    "Stalk Color Above Ring": {
        "b": "0",
        "c": "1",
        "e": "2",
        "g": "3",
        "n": "4",
        "o": "5",
        "p": "6",
        "w": "7",
        "y": "8"
    },
    "Stalk Color Below Ring": {
        "b": "0",
        "c": "1",
        "e": "2",
        "g": "3",
        "n": "4",
        "o": "5",
        "p": "6",
        "w": "7",
        "y": "8"
    },
    "Veil Type": {
        "p": "0"
    },
    "Veil Color": {
        "n": "0",
        "o": "1",
        "w": "2",
        "y": "3"
    },
    "Ring Number": {
        "n": "0",
        "o": "1",
        "t": "2"
    },
    "Ring Type": {
        "e": "0",
        "f": "1",
        "l": "2",
        "n": "3",
        "p": "4",
    },
    "Spore Print Color": {
        "b": "0",
        "h": "1",
        "k": "2",
        "n": "3",
        "o": "4",
        "r": "5",
        "u": "6",
        "w": "7",
        "y": "8"
    },
    "Population": {
        "a": "0",
        "c": "1",
        "n": "2",
        "s": "3",
        "v": "4",
        "y": "5"
    },
    "Habitat": {
        "d": "0",
        "g": "1",
        "l": "2",
        "m": "3",
        "p": "4",
        "u": "5",
        "w": "6"
    }
}


def changeLabel(label):
    labels = legends.get('Cap Shape')

    input = labels.index(label)
    return df['Cap Shape'][input]

my_expander = st.expander("Mushroom Data Form", expanded=st.session_state["expanded"])
option1 = my_expander.radio(
    'What is the Cap Shape of the mushroom?',
    df['Cap Shape'],
    horizontal=True,
    key="cap-shape")

option2 = my_expander.radio(
    'What is the mushroom cap texture?',
    df['Cap Surface'],
    horizontal=True,
    key="cap-surface",
    )

option3 = my_expander.radio(
    'What is the mushroom cap color?',
    df['Cap Color'],
    horizontal=True,
    key="cap-color",
    )

option4 = my_expander.radio(
    'Does the mushroom cap have bruises?',
    df['Bruises'],
    horizontal=True,
    key="bruises",
    )

option5 = my_expander.radio(
    'What does the mushroom smell like?',
    df['Odor'],
    horizontal=True,
    key="odor",
    )

option6 = my_expander.radio(
    'What does the gill attachments look like?',
    df['Gill Attachment'],
    horizontal=True,
    key="gill-attachment",
    )

option7 = my_expander.radio(
    'What does the gill spacing look like?',
    df['Gill Spacing'],
    horizontal=True,
    key="gill-spacing",
    )

option8 = my_expander.radio(
    'What is the mushroom gill size',
    df['Gill Size'],
    horizontal=True,
    key="gill-size",
    )

option9 = my_expander.radio(
    'What is the mushroom gill color?',
    df['Gill Color'],
    horizontal=True,
    key="gill-color",
    )

option10 = my_expander.radio(
    'what is the shape of the mushroom stalk?',
    df['Stalk Shape'],
    horizontal=True,
    key="stalk-shape",
    )

option11 = my_expander.radio(
    'What does the mushroom stalk root look like?',
    df['Stalk Root'],
    horizontal=True,
    key="stalk-root",
    )

option12 = my_expander.radio(
    'What is the texture of the stalk surface *ABOVE* the ring?',
    df['Stalk Surface Above Ring'],
    horizontal=True,
    key="stalk-surface-above-ring",
    )

option13 = my_expander.radio(
    'What is the texture of the stalk surface *BELOW* the ring?',
    df['Stalk Surface Below Ring'],
    horizontal=True,
    key="stalk-surface-below-ring",
    )

option14 = my_expander.radio(
    'What is the color of the stalk *ABOVE* the ring?',
    df['Stalk Color Above Ring'],
    horizontal=True,
    key="stalk-color-above-ring",
    )

option15 = my_expander.radio(
    'What is the color of the stalk *BELOW* the ring?',
    df['Stalk Color Below Ring'],
    horizontal=True,
    key="stalk-color-below-ring",
    )

option16 = my_expander.radio(
    'What is the type?',
    df['Veil Type'],
    horizontal=True,
    key="veil-type",
    )

option17 = my_expander.radio(
    'What is the color of the veil?',
    df['Veil Color'],
    horizontal=True,
    key="veil-color",
    )

option18 = my_expander.radio(
    'How many rings does the mushroom have?',
    df['Ring Number'],
    horizontal=True,
    key="ring-number",
    )

option19 = my_expander.radio(
    'What is the shape of the rings?',
    df['Ring Type'],
    horizontal=True,
    key="ring-type",
    )

option20 = my_expander.radio(
    'What is the color of the spore print?',
    df['Spore Print Color'],
    horizontal=True,
    key="spore-print-color",
    )

option21 = my_expander.radio(
    'How were the mushroom found/grown?',
    df['Population'],
    horizontal=True,
    key="population",
    )

option22 = my_expander.radio(
    'What is the habitat of the mushroom?',
    df['Habitat'],
    horizontal=True,
    key="habitat",
    )

response = {
    'Cap Shape':                option1,
    'Cap Surface':              option2,
    'Cap Color':                option3,
    'Bruises':                  option4,
    'Odor':                     option5,
    'Gill Attachment':          option6,
    'Gill Spacing':             option7,
    'Gill Size':                option8,
    'Gill Color':               option9,
    'Stalk Shape':              option10,
    'Stalk Root':               option11,
    'Stalk Surface Above Ring': option12,
    'Stalk Surface Below Ring': option13,
    'Stalk Color Above Ring':   option14,
    'Stalk Color Below Ring':   option15,
    'Veil Type':                option16,
    'Veil Color':               option17,
    'Ring Number':              option18,
    'Ring Type':                option19,
    'Spore Print Color':        option20,
    'Population':               option21,
    'Habitat':                  option22,
}

def convert_response(response: dict):
    result = []
    for key in response:
        choice = response.get(key)
        index = df.get(key).index(choice)
        legend = legends.get(key)[index]
        result.append(numerics.get(key).get(legend))
        
    encoded = []
    counterCol = 0
    counterRes = 0
    for key in numerics:
        res_val = result[counterRes]
        for key2 in numerics.get(key):
            if numerics.get(key).get(key2) == res_val:
                temp = [1.0]
            else:
                temp = [0.0]
            encoded.append(temp)
            counterCol += 1
        counterRes += 1
    return np.column_stack(encoded)


encoded = convert_response(response)
if "input" not in st.session_state:
    st.session_state["input"] = encoded
    
def my_callback():
    is_poisonous = ""
    st.session_state["expanded"] = False
    X = pd.DataFrame(encoded)

    model_names = ['dt_model', 'lr_model', 'nn_model',
                'rf_model', 'svc_model', 'stacked_model']
    models = []
    for i in model_names:
        path = './models/' + i + '.pkl'
        with open(path, 'rb') as f:
            models.append(pickle.load(f))

    new_pred1 = models[0].predict(X)
    new_pred2 = models[1].predict(X)
    new_pred3 = models[2].predict(X)
    new_pred3 = (new_pred3 > 0.5)
    new_pred4 = models[3].predict(X)
    new_pred5 = models[4].predict(X)
    new_X_test = np.column_stack((new_pred1, new_pred2, new_pred3, new_pred4, new_pred5))
    new_pred6 = models[5].predict(new_X_test)
    
    
    is_poisonous = ""

    if new_pred6[0] == 0:
        is_poisonous = True
    else:
        is_poisonous = False
        
    if "result" not in st.session_state:
        st.session_state["result"] = is_poisonous
    st.session_state["reset"] = True
    
def form_reset():
    del st.session_state["result"]
    del st.session_state["input"]
    st.session_state["expanded"] = True  
    st.session_state["reset"] = False    
    

predict = st.button('Predict', key = "form", on_click=my_callback)
if st.session_state["reset"]:
    reset = st.button("RESET", on_click=form_reset)

if "result" in st.session_state:
    if st.session_state["result"]:
        st.header("TASTY!")
        st.image("tasty.png", width=500)
    else:
        st.header("POISONOUS!")
        st.image("poison.jpg", width=500)


    

