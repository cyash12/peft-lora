import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
import transformers
from datasets import load_dataset
import sys

os.environ["WANDB_DISABLED"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def merge_columns(example):
    example["best_answer"] = example["best_answer"] + " Topic: " + str(example["topic_text"])
    return example

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'train':
        model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m", 
        load_in_8bit=True, 
        device_map={'':torch.cuda.current_device()},)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()  # reduce number of stored activations
        model.enable_input_require_grads()
        model.lm_head = CastOutputToFloat(model.lm_head)
        config = LoraConfig(
            r=16, #attention heads
            lora_alpha=32, #alpha scaling
            # target_modules=["q_proj", "v_proj"], #if you know the 
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM") # set this for CLM or Seq2Seq
        model = get_peft_model(model, config)
        print_trainable_parameters(model)
        data = load_dataset("yahoo_answers_topics")
        classes = [k.replace("_", " ") for k in data["train"].features["topic"].names]
        data = data.map(
            lambda x: {"topic_text": [classes[label] for label in x["topic"]]},
            batched=True,
            num_proc=1,
        )
        data['train'] = data['train'].map(merge_columns)
        data = data.map(lambda samples: tokenizer(samples['best_answer']), remove_columns=['id', 'topic', 'question_content', 'topic_text'], batched=True)
        trainer = transformers.Trainer(
            model=model, 
            train_dataset=data['train'],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=4, 
                gradient_accumulation_steps=4,
                warmup_steps=1, 
                max_steps=50, 
                learning_rate=2e-4, 
                fp16=True,
                logging_steps=1, 
                output_dir='outputs'
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        model.save_pretrained('lora_qa')
    elif mode == 'eval':
        model_id = sys.argv[2]
        prompt = 'best_answer: ' + sys.argv[3] + ' Topic: '
        config = PeftConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_id)
        batch = tokenizer(prompt, return_tensors='pt')
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=50)
        print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
