from fastai.vision.all import load_learner
import gradio as gr

LEARNER = load_learner("Akash_Food_or_Not.pkl")

def will_akash_eat(learn) -> bool:
    def will_akash_eat_inner(image_name):
        labels = learn.dls.vocab
        pred, idx, probs = learn.predict(image_name)
        return {labels[i]: float(probs[i]) for i in range(len(labels))}
    return will_akash_eat_inner

image = gr.Image(shape=(192,192))
label = gr.Label()
examples = ['example_pictures/butter_chicken.png', 'example_pictures/samosa.png', 'example_pictures/burger.png']

iface = gr.Interface(fn=will_akash_eat(LEARNER), inputs=image, outputs=label, examples=examples)
iface.launch()
