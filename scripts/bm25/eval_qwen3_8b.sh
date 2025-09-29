num=1
cuda=0
model=Qwen/Qwen3-8B
log_prefix=Qwen3-8B

hit=3


context=background
log_path=logs/${log_prefix}_${hit}_${context}_${num}.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --model ${model} --log_path $log_path --y_path $y_path --hit $hit --context $context


context=legal_framework
log_path=logs/${log_prefix}_${hit}_${context}_${num}.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --model ${model} --log_path $log_path --y_path $y_path --hit $hit --context $context


context=entity+relation
log_path=logs/${log_prefix}_${hit}_${context}_${num}.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --model ${model} --log_path $log_path --y_path $y_path --hit $hit --context $context


context=inferred_alignment
log_path=logs/${log_prefix}_${hit}_${context}_${num}.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --model ${model} --log_path $log_path --y_path $y_path --hit $hit --context $context


context=background+legal_framework
log_path=logs/${log_prefix}_${hit}_${context}_${num}.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --model ${model} --log_path $log_path --y_path $y_path --hit $hit --context $context


context=entity+relation+inferred_alignment
log_path=logs/${log_prefix}_${hit}_${context}_${num}.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --model ${model} --log_path $log_path --y_path $y_path --hit $hit --context $context


context=background+legal_framework+entity+relation  
log_path=logs/${log_prefix}_${hit}_${context}_${num}.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --model ${model} --log_path $log_path --y_path $y_path --hit $hit --context $context


context=background+legal_framework+entity+relation+inferred_alignment
log_path=logs/${log_prefix}_${hit}_${context}_${num}.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --model ${model} --log_path $log_path --y_path $y_path --hit $hit --context $context

