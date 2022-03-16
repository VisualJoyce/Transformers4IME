![IME](ime_logo.png)
# Transformers4IME

```shell
           _____ _               ____             _____ __  __ _    _   _____  _____ 
     /\   |_   _| |        /\   |  _ \     _     / ____|  \/  | |  | | / ____|/ ____|
    /  \    | | | |       /  \  | |_) |  _| |_  | (___ | \  / | |  | || (___ | |  __ 
   / /\ \   | | | |      / /\ \ |  _ <  |_   _|  \___ \| |\/| | |  | | \___ \| | |_ |
  / ____ \ _| |_| |____ / ____ \| |_) |   |_|    ____) | |  | | |__| | ____) | |__| |
 /_/    \_\_____|______/_/    \_\____/          |_____/|_|  |_|\____(_)_____/ \_____|

```

Transformers4IME是尝试将GPT运用于汉字输入法儿开发的预训练模型。

## 语料整理

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

## 模型列表

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

## 训练模式

* AbbrOnly 全缩写
* PinyinOnly 全拼音
* PinyinAbbr 混合模式

## 开始训练

物理机上直接使用docker

## 基线测试

```shell
PYTHONPATH=src python benchmarks.py --pretrained_model_name_or_path=data/pretrained/uer/gpt2-chinese-cluecorpussmall --abbr_mode=full --raw_dir data/raw --samples_json data/raw/wudao/samples_医学问答.json --num_beams 16 --num_processes 12 --best_pt /apdcephfs/share_916081/yongdai/linyang_600/transformers/model/ckpt1075000/pytorch_model.bin --additional_special_tokens=data/pretrained/additional_special_tokens.json --gpus 3,4,7
```

支持对特定模型的特定checkpoint进行评测
```shell
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src python benchmarks.py --pretrained_model_name_or_path=data/pretrained/uer/gpt2-chinese-cluecorpussmall --abbr_mode=full --raw_dir data/raw --samples_json data/raw/benchmarks/PD/samples_0.json --best_pt data/output/pinyin-gpt2-compose-top-abbr-only/ckpt1075000/8gpu_lr-1e-05_accu-8_100000steps/max-prefix-0_max-len-128/finetune/nlpcB3935B26031B427F9E64B3ADEF8F/e4ccc2707bbe335b017bdf9e926c3cf0/ckpt/model_step_2000.pt --num_beams 16 --num_processes 16 --model_name pinyin-gpt2-compose-top-abbr-only --additional_special_tokens=data/pretrained/additional_special_tokens.json
```