import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments

class T5model:

    def __init__(self, preferred_name:str, model_path:str) -> None:
        self.preferred_name = preferred_name
        self.model_checkpoint_path = model_path
        print(model_path)

        self.model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint_path)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    
    def predict(self):
        # Input text
        input_text = f"Convert name to phonetic pronunciation: {self.preferred_name}"
        print(input_text)

        # Tokenize the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate output
        output = self.model.generate(input_ids)

        # Decode the output tokens to text
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return output_text
    



