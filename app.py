from fastai.vision.all import load_learner
import gradio as gr

LEARNER = load_learner("Akash_Food_or_Not.pkl")

def will_akash_eat(learn) -> bool:
    def will_akash_eat_inner(image_name):
        food_type, _, _ = learn.predict(image_name)
        if food_type == 'indian_food':
            return True
        else:
            return False
    return will_akash_eat_inner

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()

iface = gr.Interface(fn=will_akash_eat(LEARNER), inputs=image, outputs=label)
iface.launch()
