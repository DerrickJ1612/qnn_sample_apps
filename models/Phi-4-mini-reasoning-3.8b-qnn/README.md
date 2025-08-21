---
language:
- en
library_name: transformers
license: mit
license_link: https://huggingface.co/microsoft/Phi-4-mini-instruct-reasoning/resolve/main/LICENSE
pipeline_tag: text-generation
tags:
- nlp
- math
- code
widget:
- messages:
  - role: user
    content: How to solve 3*x^2+4*x+5=1?
---

## Model Summary
 
Phi-4-mini-reasoning is a lightweight open model built upon synthetic data with a focus on high-quality, reasoning dense data further finetuned for more advanced math reasoning capabilities. 
The model belongs to the Phi-4 model family and supports 128K token context length. 
 
📰 [Phi-4-mini-reasoning Blog](https://aka.ms/phi4-mini-reasoning/blog), and [Developer Article](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/make-phi-4-mini-reasoning-more-powerful-with-industry-reasoning-on-edge-devices/4409764)<br>
📖 [Phi-4-mini-reasoning Technical Report](https://aka.ms/phi4-mini-reasoning/techreport) | [HF paper](https://huggingface.co/papers/2504.21233) <br>
👩‍🍳 [Phi Cookbook](https://github.com/microsoft/PhiCookBook) <br>
🏡 [Phi Portal](https://azure.microsoft.com/en-us/products/phi) <br>
🖥️ Try It [Azure](https://aka.ms/phi4-mini-reasoning/azure) <br>
 
 
🎉**Phi-4 models**: [[Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning)] | [[multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | [onnx](https://huggingface.co/microsoft/Phi-4-multimodal-instruct-onnx)]; 
[[mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | [onnx](https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx)]

## Intended Uses
 
### Primary Use Cases

Phi-4-mini-reasoning is designed for multi-step, logic-intensive mathematical problem-solving tasks under memory/compute constrained environments and latency bound scenarios.
Some of the use cases include formal proof generation, symbolic computation, advanced word problems, and a wide range of mathematical reasoning scenarios. 
These models excel at maintaining context across steps, applying structured logic, and delivering accurate, reliable solutions in domains that require deep analytical thinking.

### Use Case Considerations
 
This model is designed and tested for math reasoning only. It is not specifically designed or evaluated for all downstream purposes. 
Developers should consider common limitations of language models, as well as performance difference across languages, as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios. 
Developers should be aware of and adhere to applicable laws or regulations (including but not limited to privacy, trade compliance laws, etc.) that are relevant to their use case. 
 
***Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.***
 
## Release Notes
 
This release of Phi-4-mini-reasoning addresses user feedback and market demand for a compact reasoning model. 
It is a compact transformer-based language model optimized for mathematical reasoning, built to deliver high-quality, step-by-step problem solving in environments where computing or latency is constrained.
The model is fine-tuned with synthetic math data from a more capable model (much larger, smarter, more accurate, and better at following instructions), which has resulted in enhanced reasoning performance. 
Phi-4-mini-reasoning balances reasoning ability with efficiency, making it potentially suitable for educational applications, embedded tutoring, and lightweight deployment on edge or mobile systems.
If a critical issue is identified with Phi-4-mini-reasoning, it should be promptly reported through the MSRC Researcher Portal or secure@microsoft.com  
 
### Model Quality
 
To understand the capabilities, the 3.8B parameters Phi-4-mini-reasoning model was compared with a set of models over a variety of reasoning benchmarks. 
A high-level overview of the model quality is as follows:

| Model                              | AIME  | MATH-500 | GPQA Diamond |
|------------------------------------|-------|----------|--------------|
| o1-mini*                           | 63.6  | 90.0     | 60.0         |
| DeepSeek-R1-Distill-Qwen-7B        | 53.3  | 91.4     | 49.5         |
| DeepSeek-R1-Distill-Llama-8B       | 43.3  | 86.9     | 47.3         |
| Bespoke-Stratos-7B*                | 20.0  | 82.0     | 37.8         |
| OpenThinker-7B*                    | 31.3  | 83.0     | 42.4         |
| Llama-3.2-3B-Instruct              | 6.7   | 44.4     | 25.3         |
| Phi-4-Mini (base model, 3.8B)      | 10.0  | 71.8     | 36.9         |
|**Phi-4-mini-reasoning (3.8B)**     | **57.5** | **94.6** | **52.0**  |
 
Overall, the model with only 3.8B-param achieves a similar level of multilingual language understanding and reasoning ability as much larger models.
However, it is still fundamentally limited by its size for certain tasks. The model simply does not have the capacity to store too much factual knowledge, therefore, users may experience factual incorrectness. However, it may be possible to resolve such weakness by augmenting Phi-4 with a search engine, particularly when using the model under RAG settings.
 
## Usage
 
### Tokenizer
 
Phi-4-mini-reasoning supports a vocabulary size of up to `200064` tokens. The [tokenizer files](https://huggingface.co/microsoft/Phi-4-mini-reasoning/blob/main/added_tokens.json) already provide placeholder tokens that can be used for downstream fine-tuning, but they can also be extended up to the model's vocabulary size.
 
### Input Formats
 
Given the nature of the training data, the Phi-4-mini-instruct
model is best suited for prompts using specific formats.
Below are the two primary formats:
 
#### Chat format
 
This format is used for general conversation and instructions:
 
```yaml
<|system|>Your name is Phi, an AI math expert developed by Microsoft.<|end|><|user|>How to solve 3*x^2+4*x+5=1?<|end|><|assistant|>
```
### Inference with transformers

Phi-4-mini-reasoning has been integrated in the `4.51.3` version of `transformers`. The current `transformers` version can be verified with: `pip list | grep transformers`.
Python 3.8 and 3.10 will work best. 
List of required packages:

```
flash_attn==2.7.4.post1
torch==2.5.1
transformers==4.51.3
accelerate==1.3.0
```
 
Phi-4-mini-reasoning is also available in [Azure AI Studio](https://aka.ms/phi-4-mini-reasoning/azure)

#### Example
 
After obtaining the Phi-4-mini-instruct model checkpoints, users can use this sample code for inference.
 
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
torch.random.manual_seed(0)

model_id = "microsoft/Phi-4-mini-reasoning"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{
    "role": "user",
    "content": "How to solve 3*x^2+4*x+5=1?"
}]   
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

outputs = model.generate(
    **inputs.to(model.device),
    max_new_tokens=32768,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])

print(outputs[0])
```
 
## Training
 
### Model
 
+ **Architecture:** Phi-4-mini-reasoning shares the same architecture as Phi-4-Mini, which has 3.8B parameters and is a dense decoder-only Transformer model. When compared with Phi-3.5-Mini, the major changes with Phi-4-Mini are 200K vocabulary, grouped-query attention, and shared input and output embedding.<br>
+ **Inputs:** Text. It is best suited for prompts using the chat format.<br>
+ **Context length:** 128K tokens<br>
+ **GPUs:** 128 H100-80G<br>
+ **Training time:** 2 days<br>
+ **Training data:** 150B tokens<br>
+ **Outputs:** Generated text<br>
+ **Dates:** Trained in February 2024<br>
+ **Status:** This is a static model trained on offline datasets with the cutoff date of February 2025 for publicly available data.<br>
+ **Supported languages:** English<br>
+ **Release date:** April 2025<br>
 
### Training Datasets
 
The training data for Phi-4-mini-reasoning consists exclusively of synthetic mathematical content generated by a stronger and more advanced reasoning model, Deepseek-R1. 
The objective is to distill knowledge from this model. This synthetic dataset comprises over one million diverse math problems spanning multiple levels of difficulty (from middle school to Ph.D. level).
For each problem in the synthetic dataset, eight distinct solutions (rollouts) were sampled, and only those verified as correct were retained, resulting in approximately 30 billion tokens of math content.
The dataset  integrates three primary components: 
1) a curated selection of high-quality, publicly available math questions and a part of the SFT(Supervised Fine-Tuning) data that was used to train the base Phi-4-Mini model;
2) an extensive collection of synthetic math data generated by the Deepseek-R1 model, designed specifically for high-quality supervised fine-tuning and model distillation; and
3) a balanced set of correct and incorrect answers used to construct preference data aimed at enhancing Phi-4-mini-reasoning's reasoning capabilities by learning more effective reasoning trajectories

## Software
* [PyTorch](https://github.com/pytorch/pytorch)
* [Transformers](https://github.com/huggingface/transformers)
* [Flash-Attention](https://github.com/HazyResearch/flash-attention)
 
## Hardware
Note that by default, the Phi-4-mini-reasoning model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:
* NVIDIA A100
* NVIDIA H100
 
If you want to run the model on:
* NVIDIA V100 or earlier generation GPUs: call AutoModelForCausalLM.from_pretrained() with attn_implementation="eager"

## Safety Evaluation and Red-Teaming
 
The Phi-4 family of models has adopted a robust safety post-training approach. This approach leverages a variety of both open-source and in-house generated datasets. The overall technique employed to do the safety alignment is a combination of SFT, DPO (Direct Preference Optimization), and RLHF (Reinforcement Learning from Human Feedback) approaches  by utilizing human-labeled and synthetic English-language datasets, including publicly available datasets focusing on helpfulness and harmlessness, as well as various questions and answers targeted to multiple safety categories. 

Phi-4-Mini-Reasoning was developed in accordance with Microsoft's responsible AI principles. Potential safety risks in the model’s responses were assessed using the Azure AI Foundry’s Risk and Safety Evaluation framework, focusing on harmful content, direct jailbreak, and model groundedness. The Phi-4-Mini-Reasoning Model Card contains additional information about our approach to safety and responsible AI considerations that developers should be aware of when using this model.

## Responsible AI Considerations
 
Like other language models, the Phi family of models can potentially behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:
 
+ Quality of Service: The Phi models are trained primarily on English text and some additional multilingual text. Languages other than English will experience worse performance as well as performance disparities across non-English. English language varieties with less representation in the training data might experience worse performance than standard American English.  
+ Multilingual performance and safety gaps: We believe it is important to make language models more widely available across different languages, but the Phi 4 models still exhibit challenges common across multilingual releases. As with any deployment of LLMs, developers will be better positioned to test for performance or safety gaps for their linguistic and cultural context and customize the model with additional fine-tuning and appropriate safeguards.
+ Representation of Harms & Perpetuation of Stereotypes: These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups, cultural contexts, or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases.
+ Inappropriate or Offensive Content: These models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the case.
+ Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.  
+	Election Information Reliability : The model has an elevated defect rate when responding to election-critical queries, which may result in incorrect or unauthoritative election critical information being presented. We are working to improve the model's performance in this area. Users should verify information related to elections with the election authority in their region.
+ Limited Scope for Code: The majority of Phi 4 training data is based in Python and uses common packages such as "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, it is  strongly recommended that users manually verify all API uses.
+ Long Conversation: Phi 4 models, like other models, can in some cases generate responses that are repetitive, unhelpful, or inconsistent in very long chat sessions in both English and non-English languages. Developers are encouraged to place appropriate mitigations, like limiting conversation turns to account for the possible conversational drift.
 
Developers should apply responsible AI best practices, including mapping, measuring, and mitigating risks associated with their specific use case and cultural, linguistic context. Phi 4 family of models are general purpose models. As developers plan to deploy these models for specific use cases, they are encouraged to fine-tune the models for their use case and leverage the models as part of broader AI systems with language-specific safeguards in place. Important areas for consideration include:  
 
+ Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
+ High-Risk Scenarios: Developers should assess the suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.
+ Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).  
+ Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.
+ Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.
 
## License
The model is licensed under the [MIT license](./LICENSE).
 
## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
 
 
## Appendix A: Benchmark Methodology
 
We include a brief word on methodology here - and in particular, how we think about optimizing prompts. In an ideal world, we would never change any prompts in our benchmarks to ensure it is always an apples-to-apples comparison when comparing different models. Indeed, this is our default approach, and is the case in the vast majority of models we have run to date. For all benchmarks, we consider using the same generation configuration such as max sequence length (32768), the same temperature for the fair comparison.
Benchmark datasets
We evaluate the model with three of the most popular math benchmarks where the strongest reasoning models are competing together. Specifically:
-	Math-500: This benchmark consists of 500 challenging math problems designed to test the model's ability to perform complex mathematical reasoning and problem-solving.
-	AIME 2024: The American Invitational Mathematics Examination (AIME) is a highly regarded math competition that features a series of difficult problems aimed at assessing advanced mathematical skills and logical reasoning.
-	GPQA Diamond: The Graduate-Level Google-Proof Q&A (GPQA) Diamond benchmark focuses on evaluating the model's ability to understand and solve a wide range of mathematical questions, including both straightforward calculations and more intricate problem-solving tasks.
