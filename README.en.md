![ime](https://user-images.githubusercontent.com/2136700/160290194-4f30a796-876a-4750-bb3b-b5b62c4676c5.png)
# Transformers4IME

Transformers4IME is repo for exploring and adapting transformer-based models to IME.

## PinyinGPT

PinyinGPT is a model from [Exploring and Adapting Chinese GPT to Pinyin Input Method](https://arxiv.org/abs/2203.00249) 
which appears in ACL2022.
```bibtex
@article{tan2022exploring,
  title={Exploring and Adapting Chinese GPT to Pinyin Input Method},
  author={Tan, Minghuan and Dai, Yong and Tang, Duyu and Feng, Zhangyin and Huang, Guoping and Jiang, Jing and Li, Jiwei and Shi, Shuming},
  journal={arXiv preprint arXiv:2203.00249},
  year={2022}
}
```
While GPT has become the de-facto method for text generation tasks, its application to pinyin inp
ut method remains unexplored. 
In this work, we make the first exploration to leverage Chinese GPT for pinyin input method. 
We find that a frozen GPT achieves state-of-the-art performance on perfect pinyin. 
However, the performance drops dramatically when the input includes abbreviated pinyin. 
A reason is that an abbreviated pinyin can be mapped to many perfect pinyin, 
which links to even larger number of Chinese characters. 
We mitigate this issue with two strategies, including enriching the context with pinyin and optimizing the 
training process to help distinguish homophones. 
To further facilitate the evaluation of pinyin input method, 
we create a dataset consisting of 270K instances from 15 domains. 
Results show that our approach improves performance on abbreviated pinyin across all domains. 
Model analysis demonstrates that both strategies contribute to the performance boost.

![pinyinGPT-method](https://user-images.githubusercontent.com/2136700/160290180-ad531d81-4d47-48a9-a924-001780d5c5cf.png)

_Corpus Preparation_

```python
{'words': [['观众', '姥爷'], ['，'], ['如果', '你', '有', '超神', '超', '秀'], ['、'], ['坑爹', '搞笑', '素材'], ['，'],
           ['欢迎', '给', '苍', '姐', '投稿'], ['，'], ['采用', '有奖', '哦'], ['！']],
 'tokens': [[['观', '众'], ['姥', '爷']], ['，'], [['如', '果'], ['你'], ['有'], ['超', '神'], ['超'], ['秀']], ['、'],
            [['坑', '爹'], ['搞', '笑'], ['素', '材']], ['，'], [['欢', '迎'], ['给'], ['苍'], ['姐'], ['投', '稿']], ['，'],
            [['采', '用'], ['有', '奖'], ['哦']], ['！']],
 'pinyin': [[['guan', 'zhong'], ['lao', 'ye']], ['，'],
            [['ru', 'guo'], ['ni'], ['you'], ['chao', 'shen'],
             ['chao'], ['xiu']], ['、'],
            [['keng', 'die'], ['gao', 'xiao'], ['su', 'cai']],
            ['，'], [['huan', 'ying'], ['gei'], ['cang'], ['jie'],
                    ['tou', 'gao']], ['，'],
            [['cai', 'yong'], ['you', 'jiang'], ['o']], ['！']],
 'abbr': [[['g', 'z'], ['l', 'y']], ['，'], [['r', 'g'], ['n'], ['y'], ['c', 's'], ['c'], ['x']], ['、'],
          [['k', 'd'], ['g', 'x'], ['s', 'c']], ['，'], [['h', 'y'], ['g'], ['c'], ['j'], ['t', 'g']], ['，'],
          [['c', 'y'], ['y', 'j'], ['o']], ['！']]}
```

_Model List_

* GPT2
    * GPT2-Public (uer/gpt2-chinese-cluecorpussmall) [🤗 models](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)
    * GPT2-Ours (visualjoyce/gpt2-zh-21k) [🤗 models](https://huggingface.co/visualjoyce/gpt2-zh-21k)
* PinyinGPT2Concat
    * Directly
    * Segmented (visualjoyce/transformers4ime-pinyingpt-concat) [🤗 models](https://huggingface.co/visualjoyce/transformers4ime-pinyingpt-concat)
* PinyinGPT2Compose
    * PinyinGPT2ComposeBottom
    * PinyinGPT2ComposeTop
        * logits
        * states
        * residual

_Training Mode_

* AbbrOnly: full abbreviation
* PinyinOnly: none abbreviation
* PinyinAbbr: mixed (not covered in this paper)

```shell
sh pretrain_pinyingpt.sh
```

_Benchmarking_

Benchmarking dataset is shared via:

* [百度网盘](https://pan.baidu.com/s/1YEG54GSRfPzKO2gQD1IiHw?pwd=7j6v)

![99E333F0B1C6D7B67ACB9D9E61A73DA8](https://user-images.githubusercontent.com/2136700/160289844-924ef07f-b983-4e9c-b07a-45ad042e17da.png)

PD benchmarking
```shell
python3 benchmarks.py --samples_json data/benchmarks/PD/samples_0.json \
  --pretrained_model_name_or_path data/pretrained_models/gpt2-zh-ours \
  --additional_special_tokens data/pretrained/additional_special_tokens.json \
  --pinyin2char_json data/pretrained/pinyin2char.json \
  --pinyin_logits_processor_cls pinyingpt-compatible \
  --num_beams 16 \
  --abbr_mode none
```

Benchmarking with specific checkpoint
```shell
sh benchmarks.sh pinyingpt-concat data/output/pinyingpt \
  data/output/models/ckpt50000/pytorch_model.bin
```

_Acknowledgment_

Work done during internship at Tencent AI Lab.