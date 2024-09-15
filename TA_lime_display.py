# visualize LIME explanations

# import libraries
import pandas as pd
import numpy as np
import string
import ast

# prepare data
questions = pd.read_csv('explanations.csv', sep=';')
explanations_file = open('exps.txt', 'r')
explanations = explanations_file.readlines()

# formatting parameters
negative_background = np.array([192, 192, 192])
neutral_background = np.array([255, 255, 255])
positive_background = np.array([0, 255, 0])
negative_text = np.array([255, 0, 0])
neutral_text = np.array([0, 0, 0])
positive_text = np.array([0, 0, 255])

lowest = -0.6226656933285978
highest = 0.4962006553869094

def lerp(start, end, percent):
    return start+percent*(end-start)

# generate an HTML file for visualizations
with open('lime_display.html', 'w') as display:
    display.write('<html><head></head><body>')

    for idx, row in questions.iterrows():
        sentence = row.Question.split(' ')
        explanation = explanations[idx]
        exp_dictionary = dict(ast.literal_eval(explanation))
        for word in sentence:
            word_base = word.translate(str.maketrans('', '', string.punctuation))
            if word_base in exp_dictionary:
                influence = exp_dictionary[word_base]
                if influence > 0:
                    background = lerp(neutral_background, positive_background, influence/highest)
                    text = lerp(neutral_text, positive_text, influence/highest)
                else:
                    background = lerp(neutral_background, negative_background, influence/lowest)
                    text = lerp(neutral_text, negative_text, influence/lowest)
            else:
                background = neutral_background
                text = neutral_text
            display.write(f'<span style="background-color: rgb({background[0]}, {background[1]}, {background[2]}); color: rgb({text[0]}, {text[1]}, {text[2]})">{word}</span> ')
        display.write('<br/>')

# (the HTML file is later manually edited)