import argparse
import json
import logging
import os

import pandas as pda
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from transformers4ime.data.benchmark import BENCHMARK_REGISTRY
from transformers4ime.utils.logger import LOGGER

domains = (
    "医学问答",
    "体育",
    "军事",
    "农业",
    "国际",
    "娱乐",
    "房产",
    "文化",
    "教育",
    "旅行",
    "汽车",
    "游戏",
    "社会",
    "科技",
    "经济",
    "财经",
)

sample_jsons = (
    "_0-3_0-3",
    "_0-3_10+",
    "_0-3_4-9",
    "_10+_0-3",
    "_10+_10+",
    "_10+_4-9",
    "_4-9_0-3",
    "_4-9_10+",
    "_4-9_4-9",
)

splits = ["0-3", "4-9", "10+"]


def benchmark_json(benchmark, samples_json, opts):
    with open(samples_json, 'r') as f:
        samples_final = json.load(f)

    LOGGER.info(f'Total samples: {len(samples_final)}')

    prefix = f'{opts.benchmark_name}-{os.path.basename(samples_json)}-{opts.abbr_mode}-{opts.global_step}'
    result_file = os.path.join(opts.output_dir, f'{prefix}.txt')
    LOGGER.info(f'Result file: {result_file}')
    inferences_file = os.path.join(opts.output_dir, f'{prefix}.json')
    LOGGER.info(f'Inference file: {inferences_file}')
    if not os.path.isfile(result_file) or not os.path.isfile(inferences_file):
        tb_logger = SummaryWriter(opts.output_dir, filename_suffix=f'.{opts.benchmark_name}')

        in_top_k, avg_time, inferences = benchmark.run_eval(samples_final)

        if not opts.debug:
            with open(result_file, 'w') as f:
                f.write(f'average\t{avg_time}\n')
                for top_k in in_top_k:
                    value = sum(in_top_k[top_k]) / len(in_top_k[top_k]) * 100
                    tb_logger.add_scalar(
                        f'{opts.benchmark_name}-{os.path.basename(samples_json)}-{opts.abbr_mode}/top-{top_k}',
                        value,
                        opts.global_step)
                    f.write(f'{top_k}\t{value}\n')
                    LOGGER.info(
                        [f'{opts.benchmark_name}-{os.path.basename(samples_json)}-{opts.abbr_mode}', top_k, value])

            with open(inferences_file, 'w') as f:
                json.dump(inferences, f, ensure_ascii=False, indent=4)

            tb_logger.add_scalar(
                f'{opts.benchmark_name}-{os.path.basename(samples_json)}-{opts.abbr_mode}/average-time', avg_time,
                opts.global_step)
            tb_logger.close()
    else:
        LOGGER.info(f'Evaluation is finished already at {result_file}')
    return result_file, inferences_file


def build_report(filenames, opts):
    data = {}
    for f in tqdm(filenames):
        _, d, s1, s2, *argss = os.path.basename(f).replace(".json", '_').split("_")
        data.setdefault(d, {})
        data[d].setdefault(s1, {})
        data[d][s1].setdefault(s2, {})
        for item in pda.read_csv(f, sep='\t', names=['key', 'value']).to_dict('records'):
            if item['key'] in ['average', '1', '5', '10']:
                data[d][s1][s2][item['key']] = item['value']

    output = []
    for domain in domains:
        for k in ['average', '1', '5', '10']:
            try:
                ol = []
                for s1 in splits:
                    for s2 in splits:
                        ol.append(data[domain][s1][s2][k])
                output.append(f"{domain}-{k} {' '.join(map(str, ol))}")
            except Exception as e:
                print(domain, k, e)

    report_file = os.path.join(opts.output_dir, f'report.txt')
    with open(report_file, 'w') as f:
        for o in output:
            f.write(o + '\n')


def benchmark_wd(benchmark, opts):
    results = []
    for d in domains:
        for sub in sample_jsons:
            samples_json = os.path.join(opts.samples_json_dir, f"samples_{d}{sub}.json")
            result_file, inferences_file = benchmark_json(benchmark, samples_json, opts)
            results.append(result_file)

    build_report(results, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    parser.add_argument('--device', help='JSON config files', type=int, default=0)
    # parser.add_argument('--gpus', help='JSON config files', default='0')
    group.add_argument('--samples_json', default=None, help='JSON config files')
    group.add_argument('--samples_json_dir', default=None, help='JSON config files')
    parser.add_argument('--num_beams', type=int, default=8, help='JSON config files')
    parser.add_argument('--abbr_mode', type=str, choices=['none', 'xone', 'full'],
                        help='JSON config files')
    parser.add_argument('--length', type=int,
                        help='JSON config files')
    parser.add_argument('--topk', type=int,
                        help='JSON config files')
    parser.add_argument('--topp', type=int,
                        help='JSON config files')
    parser.add_argument('--temperature', type=int,
                        help='JSON config files')
    parser.add_argument('--repetition_penalty', type=int,
                        help='JSON config files')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='JSON config files')
    parser.add_argument('--output_dir', type=str, default='data/logs',
                        help='JSON config files')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        help='JSON config files')
    parser.add_argument('--additional_special_tokens', type=str, default=None,
                        help='JSON config files')
    parser.add_argument('--pinyin2char_json', type=str, default=None,
                        help='JSON config files')
    parser.add_argument('--model_name', default='gpt2',
                        help='JSON config files')
    parser.add_argument('--pinyin_logits_processor_cls', default='pinyingpt-compatible',
                        help='JSON config files')
    parser.add_argument('--benchmark_name', default='wudao',
                        help='JSON config files')
    parser.add_argument('--best_pt', type=str, default=None,
                        help='JSON config files')
    parser.add_argument('--gpt2_fixed', action='store_true',
                        help='JSON config files')
    parser.add_argument('--debug', action='store_true',
                        help='JSON config files')
    parser.add_argument('--global_step', type=int, default=0,
                        help='JSON config files')
    args = parser.parse_args()

    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 此处设置程序使用哪些显卡
    length = args.length
    # batch_size = args.batch_size
    # nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    # gpus = list(map(int, args.gpus.split(',')))
    repetition_penalty = args.repetition_penalty

    if args.model_name in ['gpt2', 'pinyingpt-compatible']:
        args.model_cls = 'pinyingpt-compatible'
        args.pinyin_logits_processor_cls = 'pinyingpt-compatible'
        args.benchmark_cls = 'pinyingpt-compatible'
    elif args.model_name in ['pinyin-cpm-compatible']:
        args.model_cls = 'pinyin-cpm-compatible'
        args.pinyin_logits_processor_cls = 'pinyin-cpm-compatible'
        args.benchmark_cls = 'pinyin-cpm-compatible'
    else:
        args.model_cls = args.benchmark_cls = args.model_name
        args.concat_mode = "segmented"
        args.position_mode = "aligned"
    #     parse_model_name(args.model_name, args)

    if args.best_pt:
        if '/fixed/' in args.best_pt:
            args.gpt2_fixed = True
        ckpt_dir = os.path.dirname(args.best_pt)
        models_dir = os.path.dirname(ckpt_dir)
        job_dir = os.path.dirname(models_dir)
        args.output_dir = os.path.join(job_dir, 'logs')
        ckpt = os.path.basename(os.path.dirname(args.best_pt).replace('ckpt', ''))
        args.global_step = int(ckpt)

    bm = BENCHMARK_REGISTRY[args.benchmark_cls](args.model_cls, args)
    if args.samples_json is not None:
        benchmark_json(bm, args.samples_json, args)
    else:
        benchmark_wd(bm, args)
