from click import command
import streamlit as st 
import os
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


st.title('Stable Diffusion (AI that can imagine)')

    

prompt=st.text_input('Enter the prompt')
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 
def generate(): 
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save(prompt+'.png')
    img = st.image(prompt+'.png')

st.button('Generate',command=generate)