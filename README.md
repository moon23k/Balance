## Transformer Balance
&nbsp; To address Natural Language Generation tasks using Transformer models, both Encoder and Decoder components are essential. 
Typically, a well-balanced structure between the Encoder and Decoder yields good performance. 
However, depending on the nature of the task, there are cases where either the Encoding or Decoding aspect becomes more critical.

To investigate the direct impact of emphasizing specific aspects on performance, this repository explores six combinations in terms of Width and Depth for models in three natural language generation tasks: Translation, Dialogue Generation, and Text Summarization. 
These combinations include models with a balanced emphasis on both Encoder and Decoder, models with a stronger focus on Encoder, and models with a stronger focus on Decoder.

<br><br> 


## Model Architecture
&nbsp; In this project, we use a Transformer structure that builds upon the standard Transformer architecture proposed in "Attention is All You Need" by adding linear layers before and after the Encoder and Decoder. 
This improvement allows the Transformer model to handle input and output values of different dimensions flexibly. 
All other aspects of the structure remain the same as the standard Transformer.

To measure the performance of each model, we conduct experiments using a baseline model and six variant models. 
We categorize these models based on their focus between the Encoder and Decoder and the model's width and depth. 
Detailed information for each model can be found in the table below.

<br> 

| Model Name | Balance | Type | Note |
|---|---|---|---|
| Equal Default Model | Equal | Default | Base Line Model, Not weighted to one side, well balanced. |
| Equal Wide Model | Equal | Wide | Not weighted to one side, but has double hidden dimension size both on Encoder and Decoder |
| Equal Deep Model | Equal | Deep | Not weighted to one side, but has double Layer Numbers both on Encoder and Decoder |
| Encoder Wide Model | Encoder | Wide | Encoder Weighted Model, with doubled hidden dimension size only on Encoder |
| Encoder Deep Model | Encoder | Deep | Encoder Weighted Model, with doubled Layer Numbers only on Encoder |
| Decoder Wide Model | Decoder | Wide | Decoder Weighted Model, with doubled hidden dimension size only on Decoder |
| Decoder Deep Model | Decoder | Deep | Decoder Weighted Model, with doubled Layer Numbers only on Decoder |

<br><br>


## Result
| Model | Machine Translation | Dialogue Generation | Text Summarization |
|:---:|:---:|:---:|:---:|
| Equal Default Model | 11.23 | 29.65 | - |
| Equal Wide Model | 0.00 | 0.00 | - |
| Equal Deep Model | 0.00 | 0.00 | - |
| Encoder Wide Model | 0.00 | 14.67 | - |
| Encoder Deep Model | 0.00 | 21.02 | - |
| Decoder Wide Model | 0.00 | 0.00 | - |
| Decoder Deep Model | 0.00 | 0.00 | - |

<br><br> 

## Reference
* [**Attention is All You Need**](https://arxiv.org/abs/1706.03762)
<br> 
