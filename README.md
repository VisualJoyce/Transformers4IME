![IME](ime_logo.png)
# Transformers4IME

*其他语言版本: [English](README.en.md)

Transformers4IME是尝试将预训练模型运用于输入法的软件包。

## PinyinGPT

PinyinGPT模型源于我们发表于ACL2022的工作 [Exploring and Adapting Chinese GPT to Pinyin Input Method](https://arxiv.org/abs/2203.00249) 。
```bibtex
@article{tan2022exploring,
  title={Exploring and Adapting Chinese GPT to Pinyin Input Method},
  author={Tan, Minghuan and Dai, Yong and Tang, Duyu and Feng, Zhangyin and Huang, Guoping and Jiang, Jing and Li, Jiwei and Shi, Shuming},
  journal={arXiv preprint arXiv:2203.00249},
  year={2022}
}
```
本文主要研究了将中文GPT的预训练模型适配到拼音输入法的问题。我们发现，在GPT的广泛使用中，仍然缺少对拼音输入法的探索。
经过对生成过程加上拼音的限制，全拼场景下的GPT的效果十分突出，在传统的数据集上就能达到SOTA。
然而，对于首字母的情形，GPT的效果出现大幅下滑，这与同声母字的候选大幅增加相关。
我们采取两种策略来解决这个问题，一方面让模型充分使用上下文信息和拼音信息，另一方面增强训练过程中对同声母字的辨析。
为了助力拼音输入法的评测，我们基于最新的语料，构建了跨15个新闻领域的270k的测试集合，集合的样本覆盖多种上文的长度和预测长度组合。
通过对模型的分析和消融，我们发现模型的两个策略都对最后的效果有促进作用。
实验结果对输入法的研究具有参考意义。

_语料整理_
例如，处理拼音的相关语料时, 我们会得到如下数据格式
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


选定需要的db文件进行合并通过transformers支持的tokenizer转换成`token id`，得到一个模型可直接使用的`txt_db`。

```shell
PYTHONPATH=src RAW_DIR=data/raw ANNOTATION_DIR=data/annotations_db TXT_DIR=data/txt_db python convert.py --domain CLUECorpusSmall --genre news2016zh_corpus --config=config/gpt2zh/pretrain_pinyin.json --use_proxy
```



```
PYTHONPATH=src RAW_DIR=data/raw ANNOTATION_DIR=data/annotations_db2 TXT_DIR=data/txt_db ANNOTATOR_TAGGER=whitespace ADDITIONAL_SPECIAL_TOKENS=data/pretrained/additional_special_tokens.json PRETRAINED_MODEL_NAME_OR_PATH=data/pretrained/uer/gpt2-chinese-cluecorpussmall python convert.py --domain 300g_word --genre train.txt07 --config=config/gpt2zh/pretrain_pinyin.json --use_proxy --split train
```

_模型列表_

* GPT2
* PinyinGPT2Concat
    * Directly
    * Segmented
* PinyinGPT2Compose
    * PinyinGPT2ComposeBottom
    * PinyinGPT2ComposeTop
        * logits
        * states
        * residual

_训练模式_

* AbbrOnly 全缩写
* PinyinOnly 全拼音
* PinyinAbbr 混合模式

```shell
sh pretrain_pinyingpt.sh
```

_基线测试_

支持对特定模型的特定checkpoint进行评测
```shell
sh benchmarks.sh pinyingpt-concat data/output/pinyingpt data/output/models/ckpt50000/pytorch_model.bin
```

_鸣谢_
该工作在腾讯AI Lab实习期间完成。