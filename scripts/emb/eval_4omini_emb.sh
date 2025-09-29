num=1
cuda=1
api_name=gpt-4o-mini
api_model=gpt-4o-mini
log_prefix=gpt-4o-mini


hit=3


context=background
log_path=logs/${log_prefix}_${hit}_${context}_${num}_emb.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}_emb.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --api_name ${api_name} --api_model ${api_model} --log_path $log_path --y_path $y_path --hit $hit --context $context --emb_search yes


context=legal_framework
log_path=logs/${log_prefix}_${hit}_${context}_${num}_emb.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}_emb.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --api_name ${api_name} --api_model ${api_model} --log_path $log_path --y_path $y_path --hit $hit --context $context --emb_search yes


context=entity+relation
log_path=logs/${log_prefix}_${hit}_${context}_${num}_emb.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}_emb.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --api_name ${api_name} --api_model ${api_model} --log_path $log_path --y_path $y_path --hit $hit --context $context --emb_search yes


context=inferred_alignment
log_path=logs/${log_prefix}_${hit}_${context}_${num}_emb.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}_emb.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --api_name ${api_name} --api_model ${api_model} --log_path $log_path --y_path $y_path --hit $hit --context $context --emb_search yes


context=background+legal_framework
log_path=logs/${log_prefix}_${hit}_${context}_${num}_emb.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}_emb.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --api_name ${api_name} --api_model ${api_model} --log_path $log_path --y_path $y_path --hit $hit --context $context --emb_search yes


context=entity+relation+inferred_alignment
log_path=logs/${log_prefix}_${hit}_${context}_${num}_emb.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}_emb.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --api_name ${api_name} --api_model ${api_model} --log_path $log_path --y_path $y_path --hit $hit --context $context --emb_search yes


context=background+legal_framework+entity+relation
log_path=logs/${log_prefix}_${hit}_${context}_${num}_emb.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}_emb.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --api_name ${api_name} --api_model ${api_model} --log_path $log_path --y_path $y_path --hit $hit --context $context --emb_search yes


context=background+legal_framework+entity+relation+inferred_alignment
log_path=logs/${log_prefix}_${hit}_${context}_${num}_emb.txt
y_path=ys/${log_prefix}_${hit}_${context}_${num}_emb.pt
CUDA_VISIBLE_DEVICES=${cuda} python eval.py --api_name ${api_name} --api_model ${api_model} --log_path $log_path --y_path $y_path --hit $hit --context $context --emb_search yes
