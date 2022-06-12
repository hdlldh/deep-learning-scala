import os
import shutil

import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoConfig,
    TFAutoModelForSequenceClassification,
    TFAutoModelForQuestionAnswering,
    TFAutoModelForTokenClassification,
    TFAutoModelForMaskedLM
)


def download_hf_model(app, model_provider, model_name, max_length, num_labels=-1, do_lower_case=True, output_path="."):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    if model_provider == "pytorch":
        if app in ["text_classification", "token_classification"]:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, torchscript=True)
            if app == "text_classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            elif app == "token_classification":
                model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
        elif app in ["question_answering", "fill_mask"]:
            config = AutoConfig.from_pretrained(model_name, torchscript=True)
            if app == "question_answering":
                model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
            elif app == "fill_mask":
                model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
        else:
            print("Unknown application: " + app)
            return

        model.eval()
        dummy_input = "This is a dummy input for torch jit trace"
        inputs = tokenizer.encode_plus(dummy_input, max_length=max_length, pad_to_max_length=True,
                                       add_special_tokens=True, return_tensors='pt')
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        traced_model = torch.jit.trace(model, (input_ids, attention_mask))
        torch.jit.save(traced_model, model_name + ".pt")
        os.makedirs(os.path.join(output_path, app, model_provider, model_name), exist_ok=True)
        shutil.move(model_name + ".pt", os.path.join(output_path, app, model_provider, model_name, model_name + ".pt"))
        tokenizer.save_pretrained(os.path.join(output_path, app, model_provider, model_name))

    elif model_provider == "tensorflow":
        if app in ["text_classification", "token_classification"]:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            if app == "text_classification":
                model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            elif app == "token_classification":
                model = TFAutoModelForTokenClassification.from_pretrained(model_name, config=config)
        elif app in ["question_answering", "fill_mask"]:
            config = AutoConfig.from_pretrained(model_name)
            if app == "question_answering":
                model = TFAutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
            elif app == "fill_mask":
                model = TFAutoModelForMaskedLM.from_pretrained(model_name, config=config)
        else:
            print("Unknown application: " + app)
            return
        model.save(os.path.join(output_path, app, model_provider, model_name))
        tokenizer.save_pretrained(os.path.join(output_path, app, model_provider, model_name))


if __name__ == "__main__":
    print('Transformers version: ', transformers.__version__)

    model_name = "bert-base-uncased"
    num_labels = -1
    max_length = 512
    do_lower_case = False
    model_provider = "tensorflow"
    app = "fill_mask"
    download_hf_model(app, model_provider, model_name, max_length, do_lower_case=False, output_path=f"{os.path.expanduser('~')}/Workspace/deep-learning-scala/build/huggingface")
