import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import weave
from weave import Model, StringPrompt

from .config import PROJECT_NAME, SYSTEM_PROMPT, ANSWER_INSTRUCTION_PROMPT

class LLaVAModel:
    def __init__(self, model_name="llava-hf/llava-1.5-13b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Initialize Weave
        weave.init(PROJECT_NAME)
        
        # Set up prompts
        self.system_prompt = StringPrompt(SYSTEM_PROMPT)
        weave.publish(self.system_prompt, name="LLAVA_system_prompt")
        
        self.answer_instruction_prompt = StringPrompt(ANSWER_INSTRUCTION_PROMPT)
        weave.publish(self.answer_instruction_prompt, name="LLAVA_answer_instruction_prompt")
    
    def generate_prompt(self, question, candidates):
        prompt = f"{self.system_prompt}\nQuestion: {question}\nOptions:\n"
        for i, candidate in enumerate(candidates):
            prompt += f"{chr(65+i)}. {candidate}\n"
        prompt += self.answer_instruction_prompt
        return prompt
    
    def generate(self, video, prompt, max_new_tokens=512):
        inputs = self.processor(
            text=prompt,
            images=video,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def evaluate(self, video, question, candidates, answer):
        prompt = self.generate_prompt(question, candidates)
        response = self.generate(video, prompt)
        return response 