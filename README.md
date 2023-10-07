## Transformer Balance

Transformer Sequence to Sequence 모델은 Encoder와 Decoder로 구성되어 있습니다. 일반적으로 Encoder와 Decoder가 균형적인 구조에서 좋은 성능을 발휘하지만
해결하고자 하는 과제의 특성에 따라, 상대적으로 Encoding이 더욱 중요한 과제도 있으며, 반대로 Decoding이 더욱 중요한 과제도 존재합니다.

필요한 부분만을 강조하는 것이 얼마나 성능에 효과적인지를 직접 확인하기 위해 
이 repo에서는 Translation, Dialogue Generation, Text Summarization 이라는 세 가지 Natural Language Generation Task에서
Encoder와 Decoder의 Balance가 맞춰진 모델, Encoder가 강조된 모델, Deocder가 강조된 모델을 Width와 Depth관점에서 총 7가지 변인을 만들어
실제 결과를 확인해봅니다.

<br><br> 


## Architecture Desc

아래와 같은 변인에 대처할 수 있도록 BaseLine Transformer Model은 Standard Transformer의 Encoder와 Decoder 시작 및 마무리 시점에
Linear 레이어가 추가된 모델을 사용합니다. 실험의 변수로 작용할 모델의 변인에 대한 상세설명은 아래와 같습니다.

### Balance
* **Equally-Weighted** <br> 
  Standard Transformer 처럼 Encoder와 Decoder의 Balance가 맞춰진 모델 구조

* **Encoder-Weighted** <br> 
  의도적으로 Encoder에 더 큰 비중을 둔 모델

* **Decoder-Weighted** <br> 
  의도적으로 Decoder에 더 큰 비중을 둔 모델

<br> 

### Model Type

* **Default Model** <br> 
  의도적으로 딥하거나 와이드하게 변형을 거치는 아래의 모델들과 달리 config.yaml 파일에 사용자가 명시한 그대로의 하이퍼 파라미터로 구성된 모델.

* **Wide Model** <br> 
  config.yaml 파일에 사용자가 명시한 내용중에서 모델의 width를 결정짓는 Hidden Dimension를 의도적으로 2배 만큼 증가시킨 모델

* **Deep Model** <br> 
  config.yaml 파일에 사용자가 명시한 내용중에서, 모델의 depth를 결정짓는 Num of Layers를 의도적으로 2배 만큼 증가시킨 모델


<br><br>

## Result
| Model | Translation | Dialogue Generation | Summarization |
|:---:|:---:|:---:|:---:|
| Equal Default Model | - | - | - |
| Equal Wide Model | - | - | - |
| Equal Deep Model | - | - | - |
| Encoder Wide Model | - | - | - |
| Encoder Deep Model | - | - | - |
| Decoder Wide Model | - | - | - |
| Decoder Deep Model | - | - | - |


<br><br> 

## Reference
Attention is All You Need

<br> 
