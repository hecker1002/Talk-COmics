import gradio as gr 

from panels import divide_panels  # divid image in panels 
 
from langchain_integrate import get_story # get the ful lstory ans answer queries 
from speech_bubble_ocr import print_dial , extract_speech_from_panels
from PIL import Image 

from transformers import T5Tokenizer 
from transformers import T5ForConditionalGeneration

def call(  img , query   ) :

    temp_path = "uploaded_image.jpg"
    image  = Image.fromarray( img )
    image.save(temp_path)  

    num_img  = divide_panels( temp_path )
    panel_filenames = [f"panel_{i}.jpg" for i in range(num_img)]
    
    # Extract speech from all panels
    speech = extract_speech_from_panels(panel_filenames)
    
    answer   = get_story( speech , query  )
    return answer  



app = gr.Interface ( fn = call , inputs= [ gr.Image( label="upload Comics" , ) , gr.Textbox( label="Your Query?")] , outputs="text")

app.launch()


