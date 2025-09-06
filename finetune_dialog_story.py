'''   Fien tuen a small LLm that can take dialogues and genertae meaningful context and story ans answer quesn form ot 
In simpel words , FInetunign over a dialgues-story-query dataset 
 '''

# load the tokenizer ( that will tokenize  the input text and store its tokens' input_ids ( used for embeddign later  ))
from transformers import T5Tokenizer 
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# laod the actual llm model( t5-small) for text 2 text seq gen task ( this wil lbe fien tuend on oir specific daataset )
from transformers import T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# our dataset ( we need to prwpare or scrap ourselves )
dataset = {
    "dialogue": [
        "User1: The math homework was hard. User2: Yeah! Really.",
        "User1: I love this new movie. User2: Me too! It was fantastic.",
        "User1: It's raining again. User2: I know, I forgot my umbrella!",
        "User1: I think I’m going to fail this test. User2: Don’t worry, you’ll do fine.",
        "User1: What’s your favorite pizza topping? User2: I love pepperoni!",
        "User1: I’m so tired today. User2: Me too, I stayed up late last night.",
        "User1: I think I lost my keys. User2: Oh no, did you check your bag?",
        "User1: Have you seen the new superhero movie? User2: Yeah, it was amazing!",
        "User1: I’m craving chocolate. User2: Same here, I could eat an entire bar.",
        "User1: I need to go to the store. User2: Do you need any help?",
        "User1: I can’t believe it’s already Friday! User2: I know, this week flew by.",
        "User1: It’s so hot today! User2: Yeah, I think it might hit 90 degrees.",
        "User1: I’m looking forward to the weekend. User2: Me too, I need some rest.",
        "User1: I just got a new job! User2: That’s awesome, congrats!",
        "User1: I don’t know what to wear for the party. User2: Maybe you should wear something fun.",
        "User1: I’m planning a vacation next month. User2: That sounds exciting, where are you going?",
        "User1: I haven’t seen you in ages! User2: I know, we should catch up soon.",
        "User1: I love walking by the beach at sunset. User2: That sounds beautiful.",
        "User1: Do you want to grab some coffee later? User2: Sure, that sounds great!",
        "User1: I’ve been feeling a bit down lately. User2: I’m sorry to hear that, do you want to talk about it?"
    ],
    "story": [
        "One person finds the math homework hard and his friend agrees.",
        "Both users enjoyed the movie a lot.",
        "One person complains about the rain, and the other forgot their umbrella.",
        "One person worries about failing the test, but the other reassures them.",
        "User 1 and User 2 discuss their favorite pizza toppings.",
        "Both users are tired after staying up late.",
        "User 1 loses their keys and asks User 2 for help.",
        "Both users talk about enjoying a superhero movie they saw together.",
        "User 1 and User 2 crave chocolate and joke about eating a whole bar.",
        "User 1 asks if User 2 needs help with a trip to the store.",
        "User 1 and User 2 are surprised how quickly the week has passed and are ready for the weekend.",
        "User 1 and User 2 discuss how hot the weather is today.",
        "User 1 and User 2 both look forward to relaxing over the weekend.",
        "User 1 shares their excitement about landing a new job.",
        "User 1 asks for advice on what to wear to a party, and User 2 suggests something fun.",
        "User 1 talks about their upcoming vacation, and User 2 shows excitement.",
        "User 1 and User 2 reconnect after a long time and plan to catch up soon.",
        "User 1 enjoys walking by the beach at sunset, and User 2 agrees it sounds beautiful.",
        "User 1 invites User 2 to grab coffee later, and User 2 happily agrees.",
        "User 1 shares they’ve been feeling down, and User 2 offers a listening ear."
    ]
}


# convert to proepr troch Dtaaset for LLM 
from datasets import Dataset 

dataset  = Dataset.from_dict( dataset )
# to preorcess OUR original daatset into a format accpetable to LLM (w e want to fine-tune )

def preprocess( examples ) :
    # Paas a parent prompt + tokenize the dialgue ( and stor e input ids )
    inp = ["Summarize this whole dialgue as a story :" + d for d in examples["dialogue"] ]
    model_tokens  = tokenizer( inp , max_length = 216 , truncation =True  , padding = "max_length" )

    # target labels(story) and tokenize it 
    labels = [s for s in examples["story"]  ]  # strings 
    label_tokens  = tokenizer( labels , max_length = 216 , truncation =True  , padding = "max_length" ) # tokenize (numerical)

    model_tokens["labels"] = label_tokens["input_ids"]

    return model_tokens # finald aatset of tokens of ( dialogue input) + tokens of ( story output )

final_data  = dataset.map( preprocess , batched =True ) 
split_data = final_data.train_test_split(test_size=0.2)  # 80% train, 20% validation

# defien the trianigna rgument (f ro fientinign the model ) 

from transformers import TrainingArguments

train_args = TrainingArguments(
    output_dir="output_models" , 
    per_device_train_batch_size=2 , 
    evaluation_strategy="epoch" , 
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=10 , 
    num_train_epochs= 100 , 
    learning_rate= 1e-3  
)

# give the mdoe 
# trian and valid key smade in dtaset only when we split 
from transformers import Trainer # helps to finutuen accesory 
trainer  = Trainer( model = model , args = train_args , train_dataset= split_data["train"] ,  eval_dataset=split_data["test"] ) # whel ampign preproces fcntino to orginald aatset train and validatio keys automccatically created in dict 


trainer.train()

# save mdoel adn tokenizer 
model.save_pretrained("my_fine_tuned_t5")
tokenizer.save_pretrained("my_fine_tuned_t5")


# inference  from fine-tuned model ( mdoel,geegrnate ( input_ids ))
# load the saved tokenize and saved mdoel
tokenizer_new  = T5Tokenizer.from_pretrained("my_fine_tuned_t5")
model_new   = T5ForConditionalGeneration.from_pretrained("my_fine_tuned_t5")

def generate_story(  dialogue ) :
    input_ids = tokenizer_new( "Summarize this whole dialgue as a story : " + dialogue , return_tensors  ="pt" ).input_ids # extraxct from tokenize r, input ids seprately 
    
    output_ids = model_new.generate( input_ids  , max_length =150 )

    # output_ids[0] => contains actual  output tokens 
    return tokenizer_new.decode (  output_ids[0] , skip_special_tokens = True )

dial_ = "User 1 : THis is a very sunny day . User 2 : yeah , I wish it would rain ."

story  = generate_story (  dial_)
 
print ( "For your inference" , story )


    