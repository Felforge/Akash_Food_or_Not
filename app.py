from fastai.vision.all import load_learner
import gradio as gr

LEARNER = load_learner("Akash_Food_or_Not.pkl")

def will_akash_eat(learn):
    def will_akash_eat_inner(image_name):
        labels = ["STEALING THIS OUT THE FRIDGE 😋🍴", "H HEEEELLLL NAAAAHHH 🤮"]
        pred, idx, probs = learn.predict(image_name)
        return {labels[i]: float(probs[i]) for i in range(len(labels))}
    return will_akash_eat_inner

image = gr.Image()
label = gr.Label()
title = "Will Akash eat this food?"
examples = ['example_pictures/butter_chicken.png', 'example_pictures/samosa.png', 'example_pictures/burger.png']

iface = gr.Interface(fn=will_akash_eat(LEARNER), inputs=image, outputs=label, examples=examples, title=title)
iface.launch()
