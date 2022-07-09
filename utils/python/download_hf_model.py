import os
import shutil

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoConfig,
    TFAutoModelForSequenceClassification,
    TFAutoModelForQuestionAnswering,
    TFAutoModelForTokenClassification,
    TFAutoModelForMaskedLM
)

from HfBaseModelType import HfBaseModelType as hfm

def download_hf_model(app, model_provider, model_name, max_length, num_labels=-1, do_lower_case=True, output_path="."):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    if model_provider == "pytorch":
        if app in [hfm.TEXT_CLASSIFICATION, hfm.TOKEN_CLASSIFICATION]:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, torchscript=True)
            if app == hfm.TEXT_CLASSIFICATION:
                model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            elif app == hfm.TOKEN_CLASSIFICATION:
                model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
        elif app in [hfm.QUESTION_ANSWERING, hfm.FILL_MASK]:
            config = AutoConfig.from_pretrained(model_name, torchscript=True)
            if app == hfm.QUESTION_ANSWERING:
                model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
            elif app == hfm.FILL_MASK:
                model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
        elif app in [hfm.ZERO_SHOT_CLASSIFICATION]:
            config = AutoConfig.from_pretrained(model_name, torchscript=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        else:
            print("Unknown application: " + app)
            return

        model.eval()
        dummy_input = "This is a dummy input for torch jit trace"
        if app == hfm.ZERO_SHOT_CLASSIFICATION:
            hypothesis = "This example is test."
            inputs = tokenizer.encode(dummy_input, hypothesis, return_tensors='pt',
                                          truncation_strategy='only_first')
            traced_model = torch.jit.trace(model, (inputs, ))
        else:
            inputs = tokenizer.encode_plus(dummy_input, max_length=max_length, pad_to_max_length=True,
                                       add_special_tokens=True, return_tensors='pt')
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            traced_model = torch.jit.trace(model, (input_ids, attention_mask))
        model_name = model_name.split("/")[-1]
        torch.jit.save(traced_model, model_name + ".pt")
        os.makedirs(os.path.join(output_path, app, model_provider, model_name), exist_ok=True)
        shutil.move(model_name + ".pt", os.path.join(output_path, app, model_provider, model_name, model_name + ".pt"))
        tokenizer.save_pretrained(os.path.join(output_path, app, model_provider, model_name))

    elif model_provider == "tensorflow":
        if app in [hfm.TEXT_CLASSIFICATION, hfm.TOKEN_CLASSIFICATION]:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            if app == hfm.TEXT_CLASSIFICATION:
                try:
                    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)
                except:
                    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config, from_pt=True)
            elif app == hfm.TOKEN_CLASSIFICATION:
                try:
                    model = TFAutoModelForTokenClassification.from_pretrained(model_name, config=config)
                except:
                    model = TFAutoModelForTokenClassification.from_pretrained(model_name, config=config, from_pt=True)
        elif app in [hfm.QUESTION_ANSWERING, hfm.FILL_MASK]:
            config = AutoConfig.from_pretrained(model_name)
            if app == hfm.QUESTION_ANSWERING:
                try:
                    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
                except:
                    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name, config=config, from_pt=True)
            elif app == hfm.FILL_MASK:
                try:
                    model = TFAutoModelForMaskedLM.from_pretrained(model_name, config=config)
                except:
                    model = TFAutoModelForMaskedLM.from_pretrained(model_name, config=config, from_pt=True)
        else:
            print("Unknown application: " + app)
            return
        tokenizer.save_pretrained(os.path.join(output_path, app, model_provider, model_name))
        model_name = model_name.split("/")[-1]
        model(model.dummy_inputs)
#         model.save(os.path.join(output_path, app, model_provider, model_name))
        model.save_pretrained(os.path.join(output_path, app, model_provider, model_name), saved_model=True)


if __name__ == "__main__":
    print('Transformers version: ', transformers.__version__)

#     model_name = "bert-base-uncased"
#     model_name = "deepset/bert-base-cased-squad2"
    model_name = "facebook/bart-large-mnli"
    num_labels = -1
    max_length = 512
    do_lower_case = True
    model_provider = "pytorch"
#     model_provider = "tensorflow"
#     app = "fill_mask"
#     app = "question_answering"
#     app = "text_classification"
    app = "zero-shot-classification"
#     output_path = f"{os.path.expanduser('~')}/Workspace/deep-learning-scala/build/huggingface"
    output_path = "../../build/huggingface"
    download_hf_model(app, model_provider, model_name, max_length, do_lower_case=do_lower_case, output_path=output_path)
