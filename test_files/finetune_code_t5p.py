"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""

import os
import pprint
import argparse
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding


# Create CodeT5Classification class in order to create classification model
class CodeT5ClassificationModel(nn.Module):
    def __init__(self):
        super(CodeT5ClassificationModel, self).__init__()

        self.base_model = AutoModel.from_pretrained("Salesforce/codet5p-110m-embedding", trust_remote_code=True).to("cpu")
        self.fc1 = nn.Linear(256,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,7)
        

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids)[0]

        outputs = F.relu(self.fc1(outputs))
        outputs = F.relu(self.fc2(outputs))
        outputs = self.fc3(outputs)

        return outputs

def run_training(args, model, train_data, data_collator):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='none',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args):
    # Load and tokenize data
    # if os.path.exists(args.cache_data):
    if False:
        train_data = load_from_disk(args.cache_data)
        print(f'  ==> Loaded {len(train_data)} samples')
        return train_data
    else:
        # Example code to load and process code_x_glue_ct_code_to_text python dataset for code summarization task
        
        # TODO: modify this
        datasets = load_dataset("systemk/codenet", split="train[10:20]")
        datasets.select_columns(["code", "status"])
        tokenizer = AutoTokenizer.from_pretrained(args.load)

        def preprocess_function(examples):
            return tokenizer(examples["code"], truncation=True)

        train_data = datasets.map(
            preprocess_function
        )
        print(f'  ==> Loaded {len(train_data)} samples')
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        return train_data, data_collator


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    train_data, data_collator = load_tokenize_data(args)

    # Load model from `args.load`
    model = CodeT5ClassificationModel()
    # model = AutoModelForSequenceClassification.from_pretrained(args.load)
    # print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, data_collator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Classification task")
    parser.add_argument('--data-num', default=-1, type=int)

    # Can we change this? ####
    parser.add_argument('--max-source-len', default=320, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    #############

    parser.add_argument('--cache-data', default='cache_data/classification-test', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/classification-test", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)