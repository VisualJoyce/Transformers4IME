import argparse
import json
import logging
import os

from torch.utils.tensorboard import SummaryWriter

from transformers4ime.data.benchmark import BENCHMARK_REGISTRY
from transformers4ime.utils.logger import LOGGER
from transformers4ime.utils.misc import parse_model_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='JSON config files', type=int, default=0)
    # parser.add_argument('--gpus', help='JSON config files', default='0')
    parser.add_argument('--samples_json', help='JSON config files')
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

    with open(args.samples_json, 'r') as f:
        samples_final = json.load(f)

    LOGGER.info(f'Total samples: {len(samples_final)}')

    if args.model_name in ['gpt2', 'pinyingpt-compatible']:
        args.model_cls = 'pinyingpt-compatible'
        args.pinyin_logits_processor_cls = 'pinyingpt-compatible'
        args.benchmark_cls = 'pinyingpt-compatible'
    elif args.model_name in ['pinyin-cpm-compatible']:
        args.model_cls = 'pinyin-cpm-compatible'
        args.pinyin_logits_processor_cls = 'pinyin-cpm-compatible'
        args.benchmark_cls = 'pinyin-cpm-compatible'
    else:
        parse_model_name(args.model_name, args)

    if args.best_pt:
        ckpt = os.path.basename(os.path.dirname(args.best_pt).replace('/', '_'))
        if '/fixed/' in args.best_pt:
            args.gpt2_fixed = True
    else:
        ckpt = os.path.basename(args.pretrained_model_name_or_path)

    os.makedirs(os.path.join(args.output_dir, ckpt), exist_ok=True)

    prefix = f'{args.benchmark_name}-{os.path.basename(args.samples_json)}-{args.global_step}-{args.abbr_mode}'
    result_file = os.path.join(args.output_dir, ckpt, f'{prefix}_{ckpt}_abbr.txt')
    inferences_file = os.path.join(args.output_dir, ckpt, f'{prefix}_{ckpt}_abbr.json')
    if not os.path.isfile(result_file) or not os.path.isfile(inferences_file):
        tb_logger = SummaryWriter(args.output_dir, filename_suffix=f'.{args.benchmark_name}')

        benchmark = BENCHMARK_REGISTRY[args.benchmark_cls](args.model_cls, args)
        in_top_k, avg_time, inferences = benchmark.run_eval(samples_final)

        if not args.debug:
            with open(result_file, 'w') as f:
                f.write(f'average\t{avg_time}\n')
                for top_k in in_top_k:
                    value = sum(in_top_k[top_k]) / len(in_top_k[top_k]) * 100
                    print(top_k, value)
                    tb_logger.add_scalar(
                        f'{args.benchmark_name}-{os.path.basename(args.samples_json)}-{args.abbr_mode}/top-{top_k}',
                        value,
                        args.global_step)
                    f.write(f'{top_k}\t{value}\n')

            with open(inferences_file, 'w') as f:
                json.dump(inferences, f, ensure_ascii=False, indent=4)

            tb_logger.add_scalar(
                f'{args.benchmark_name}-{os.path.basename(args.samples_json)}-{args.abbr_mode}/average-time', avg_time,
                args.global_step)
            tb_logger.close()
    else:
        LOGGER.info(f'Evaluation is finished already at {result_file}')
