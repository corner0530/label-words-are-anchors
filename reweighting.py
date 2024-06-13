from transformers.hf_argparser import HfArgumentParser

from icl.analysis.reweighting import ReweightingArgs, train

parser = HfArgumentParser((ReweightingArgs,))
(args,) = parser.parse_args_into_dataclasses()
train(args)
