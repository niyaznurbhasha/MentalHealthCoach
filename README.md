Mental Health Support LLM 

## Setup

Note if you don't have a gpu available, then uncomment the torch  line in requirements.txt and skip the torch gpu install command
And I found my torch install command for my specific cuda version on here: https://pytorch.org/get-started/previous-versions/

conda create -n finetune python=3.10
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

### Then run training and inference. 

python prepare_data.py
python train.py
python eval_model.py (can also do in streamlit app)
streamlit run app.py
- Explore the dataset (token length distributions, sample training pairs)
- Visualize model results with Histograms of ROUGE-L and semantic similarity metrics
- Chat with the model with multi-turn context to see if its effectiveness in mental health coaching in real time

### Model Weights
The fine-tuned model weights are stored on Hugging Face due to GitHub's file size limits.

You can download the weights from the following link and then place them in a models directory inside this folder directory. 

https://huggingface.co/yazpasha/mentalhealthcoach

This includes:
- `model.safetensors`
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `training_args.bin` (optional metadata)

## Motivation

I've always been passionate about mental health. With the explosion of llm's, many people have been turning to ChatGPT to use as a therapist, but recent lawsuits
against OpenAI have demonstrated that this is an incomplete solution, sometimes with tragic consequences: 
https://apnews.com/article/openai-chatgpt-lawsuit-suicide-56e63e5538602ea39116f1904bf7cdc3

While the model I finetuned for this project is nowhere near ChatGPT's effectiveness at providing advice, I hope to build on this in the future. I will discuss 
those potential next steps later in this report. 

## Summary 

### Dataset / Preprocessing / Exploration 

I selected the EmpatheticDialogues dataset (https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm) for finetuning. There were several good options available on HF to train this model, but I decided to go with this one because it is reputable, has clean conversational text, rich emotional context, larger than others (25K Conversations), and easy to preprocess into input to target pairs. 

Another reason I selected it was the people curating the dataset had already restructured and cleaned the data for LLM training and inference i.e. Replaced instances of '_comma_' with ',' for improved readability.

I did still do some basic data cleaning like removing extra white space and line breaks. Train/valid/test splits were written to an SQLite database for querying and reproducibility. Token length statistics and histograms were generated as part of initial data exploration, informing token limits in data prep and training. 

### Finetuning Model
I used google's flan-t5-small model to finetune because its lightweight, fast to finetune, strong at instruction following, and ideal for short form generation tasks. I engineered a specific instruction prefix to frame the model as a supportive friend providing both empathy and practical suggestions while avoiding medical advice. I fine-tuned using eval loss rather than generating text during training, which is faster and avoids decode bugs. Standard regularization techniques like label smoothing, gradient clipping, weight decay, and warmup were applied, with a fixed seed to ensure reproducible results.

### After training, ROUGE-L is manually computed on 200 validation samples. Note ROUGE-L focuses on actual word overlap whereas semantic similarity is more focused on meaning of the sentences. 
### ROUGE-L: 0.1470 
### Semantic Similarity: 0.2539

## Example Inputs/ Outputs 

Sample predictions:

--- Example 1 (Similarity: 0.359) ---
User: Today,as i was leaving for work in the morning,i had a tire burst in the middle of a busy road. That scared the hell out of me!
Expected: Are you fine now?
Model: Oh no! Did you get hurt?

--- Example 2 (Similarity: 0.092) ---
User: Yeah,i'm doing alright now, but with minor injuries.
Expected: Cool :) Is your car damaged a lot?
Model: I'm sorry to hear that. I hope you have a good time.

--- Example 3 (Similarity: 0.310) ---
User: A few weeks ago, I was walking through my hallway, minding my own business, when all of a sudden a hand reached out from under a table and grabbed my ankle. I was so suprised. I thought i was got. Turns out, it was my son.
Expected: That's funny, hope he didn't give you a heart attack.
Model: Oh no! I'm sorry to hear that. I'm sorry to hear that.

## Next Steps

Given the limited training time (~1 hour, 3 epochs), the results are reasonable but have clear room for improvement. The model demonstrates basic empathetic response patterns but would benefit from extended training and the improvements outlined below.

Longer training - Train for 10-15 epochs instead of 3
Larger model - Use flan-t5-base (220M params) or flan-t5-large (770M) instead of small (60M)
Better prompt engineering - Experiment with different instruction prefixes and few-shot examples in the prompt
Crisis Detection - For significant health crises, need to operate differently, i.e. don't encourage significant action like ChatGPT, potentially reach out to family, etc. 


