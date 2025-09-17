
## alpaca eval


Ss=("128" "64")
bss=("128" "256" "64")
ks=("5")
d_types=("topk" "categorical" "conv" "llada_conv" "semiar")
counter=0
start_from=1
for S in "${Ss[@]}"; do
    for b in "${bss[@]}"; do
        for k in "${ks[@]}"; do
            for d_type in "${d_types[@]}"; do
                ((counter++))

                if [ "$counter" -lt "$start_from" ]; then
                    continue
                fi

                k_suffix=""
                if [ "$d_type" != "llada" ] && [ "$d_type" != "lladaconv" ]; then
                    k_suffix="_k${k}"
                fi

                python gen2_answer_generation.py \
                init_from_checkpoint.init_file=checkpoint_path.ckpt \
                +model_path=kuleshov-group/mdlm-owt \
                +decode_type=${d_type} \
                +block_length=${b} \
                +temperature=1 \
                lora.bool=False \
                +steps=${S} \
                +gen_length=512 \
                sampling.topk_k=${k} \
                +mask_id=126336 \
                +cfg=0 \
                +remasking=low_confidence \
                +eos_fill=False \
                +category=small_trained \
                +eval_data=alpaca_eval \
                +generator=small-alpaca_${d_type}${k_suffix}_temp1_L512_S${S}_bs${b} \
                master_port=65535 \
                +rand_value=860${counter} \
                bidirection=False \
                +batch_size=1



            done
        done
    done
done


