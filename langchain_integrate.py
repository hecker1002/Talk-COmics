'''  
LangChain -> framwork ( like Tensorflow ) that helps to build good apps using LLms 
and it hdies intricasis ( like compuation , storage ,api calls etc) and make a seamless llm-based  apps 

it uses  4 main process - ( Model ( LLm wrapper ) ) + ( memory ( chat hostory)) + ( chains ( satck of rule /prompt)) + ( prompt)=> output 

IMPORTANT -- LANCHAIN USED ONLY WHEN WE AR EMAKING API CALLS 
FOR FINETUNED MOEL NO USE 
'''
import re 
from langchain_community.llms import HuggingFaceHub
from speech_bubble_ocr import print_dial , extract_speech_from_panels

from transformers import T5Tokenizer 
from transformers import T5ForConditionalGeneration

def get_story(speech , query_user   ) :


    # speech = contain lsit of dialogeues 

    # preproces st dialogues 
    def clean_dialogues(dialogues):
        cleaned = []
        for text in dialogues:
            # Fix exact **duplicate consecutive words** (but not partial words!)
            text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text)  

            # Remove unnecessary symbols **EXCEPT punctuation & ":" (for dialogue formatting)**
            text = re.sub(r'[^\w\s:,.!?]', '', text)  

            # Ensure space between "User 1:" and "User 2:"
            text = re.sub(r'(User \d+:)', r'\n\1', text)

            cleaned.append(text.strip())  
        return cleaned

    cleaned_dialogues =  clean_dialogues( speech )
    dialogue_text = "\n".join(cleaned_dialogues)



    # to talk with mdoel , create an isntance in your env
    # # LLm wrapper ( software app that interact with llm using api calls ) 

    ''' USe a FIen tuned Model on specific comics Dataset '''

   

    model_path = "my_fine_tuned_t5"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # llm = HuggingFaceHub(mdo
    #     repo_id="google/flan-t5-base",  
    #     model_kwargs={"temperature": 0.4 } , # creatives 

    #    
    # )

    # send the msg directly ( llm (message ))
    # response  = llm.invoke( "hello , how is it going man ?")

    # print( response )


    # to store Past conversation with USer  ,use a buffer meory object 
    # from langchain.memory import ConversationBufferMemory 
    # memory  = ConversationBufferMemory(  memory_key="history") # insatnce of meory object 

    # # make a chain  to store recpie (process of genration)  output form llms  ( conversation -> meory store ->conversation->memory ... )
    # # from langchain_core.runnables.history import RunnableWithMessageHistory 
    # # chain = RunnableWithMessageHistory ( llm = llm , memory  =memory  )
    # from langchain.chains import ConversationChain 
    # chain = ConversationChain(llm=llm, memory=memory)

    # now when we chat with it , it keep on  storing memory (history of our chats)


    # send the msg directly ( llm (message ))
    # response  = llm.invoke( "hello , how is it going man ?")

    # print( response )

    # to tlak with LLm with a default prompot always + our query 

#     from langchain.prompts import PromptTemplate 


#     # fixed prompt for AI LLM model 
#     template = PromptTemplate.from_template(
#     "Convert the following dialogues into a compelling short story. Maintain the conversational tone, infer missing details, and make it engaging.\n\n{dialogues}\n\nStory:"
# )


    def generate_story(dialogue):

        input_text = f"Summarize this whole dialogue sequence in the form of a good story: {dialogue}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=150)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    story = generate_story(dialogue_text)

    print("The story : " ,  story )

    # suer query 


    def answer_question(story, question):
        input_text = f"Based on this story, answer the question: {story} \n\nQuestion: {question}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=100)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    answer_your = answer_question(story, query_user)

    return answer_your

    
    # ask qeustions from stroy ---> 

    # print("\nYou can now ask questions about the story. Type 'exit' to stop.\n")

    # while True:
    #     user_question = input("Ask a question about the story: ")
    #     if user_question.lower() == "exit":
    #         print("Exiting...")
    #         break
        
    #     #  Create a prompt for answering questions based on the story
    #     question_prompt = PromptTemplate.from_template(
    #         "Based on the following story, answer the user's question.\n\nStory:\n{story}\n\nUser Question: {question}\nAnswer:"
    #     )

    #     # Pass story and user query to LLM
    #     answer_your = llm.invoke(question_prompt.format(story= answer , question=user_question))
        
    #     print("\nAI Answer:", answer_your , "\n")

  
