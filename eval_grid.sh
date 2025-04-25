#! /bin/bash
n=15
d=8
temp=0.0
#for n in 1 5 10 15 20
#for d in 10 9 8 7
#for idx in 0 1 2
eagle2_path="./ckpt/EAGLE-LLaMA3.1-Instruct-8B"
eagle3_path="./ckpt/EAGLE3-LLaMA3.1-Instruct-8B"
baseline_path="./ckpt/Llama-3.1-8B-Instruct"
for idx in 0
do
  max_tok=$(($n*6))
  python eagle/evaluation/gen_ea_answer_llama3chat.py \
    --ea-model-path ${eagle3_path} \
    --base-model-path ${baseline_path} \
    --top-k $n \
    --total-token $max_tok \
    --depth $d \
    --max-new-token 2048 \
    --temperature $temp \
    --use-eagle3 \
    --model-id eagle3-llama3.1-instruct-8b-temperature-${temp}-topk-${n}-maxtokens-${max_tok}-depth-${d}-run-${idx}.jsonl

  python eagle/evaluation/gen_ea_answer_llama3chat.py \
    --ea-model-path ${eagle2_path} \
    --base-model-path ${baseline_path} \
    --top-k $n \
    --total-token $max_tok \
    --depth 5 \
    --max-new-token 2048 \
    --temperature $temp \
    --model-id eagle2-llama3.1-instruct-8b-temperature-${temp}-topk-${n}-maxtokens-${max_tok}-depth-${d}-run-${idx}.jsonl

  python eagle/evaluation/gen_baseline_answer_llama3chat.py \
    --ea-model-path ${eagle3_path} \
    --base-model-path ${baseline_path} \
    --top-k $n \
    --total-token $max_tok \
    --depth $d \
    --max-new-token 2048 \
    --temperature $temp \
    --model-id baseline-llama3.1-instruct-8b-temperature-${temp}-topk-${n}-maxtokens-${max_tok}-depth-${d}-run-${idx}.jsonl
done
