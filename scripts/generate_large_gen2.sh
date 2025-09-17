
## alpaca_eval


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
                +model_path=GSAI-ML/LLaDA-8B-Base \
                +decode_type=${d_type} \
                +block_length=${b} \
                +temperature=1 \
                lora.bool=True \
                +steps=${S} \
                +gen_length=512 \
                sampling.topk_k=${k} \
                +mask_id=126336 \
                +cfg=0 \
                +remasking=low_confidence \
                +eos_fill=False \
                +category=large8B_trained_20250702_alpaca_L512 \
                +eval_data=alpaca_eval \
                +generator=llada8B_${d_type}${k_suffix}_temp1_L512_S${S}_bs${b} \
                master_port=65535 \
                +rand_value=860${counter} \
                bidirection=False \
                +batch_size=1



            done
        done
    done
done



##   gsm8k  (llada-inst)
d_types=("conv" "topk" "categorical" "llada" "remdm_loop_conv" "remdm_loop_topk" )
bss=("128" "64" "256")
Ss=("128" "64" "256")
ks=("1")
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



                        if [ "$d_type" == "llada" ] || [ "$d_type" == "lladaconv" ]; then
                            temp="0"
                        else
                            temp="1"
                        fi


                        python gen2_answer_generation.py \
                        init_from_checkpoint.init_file='' \
                        +model_path=GSAI-ML/LLaDA-8B-Instruct \
                        +decode_type=${d_type} \
                        +block_length=${b} \
                        +temperature=1 \
                        +steps=${S} \
                        +gen_length=512 \
                        sampling.topk_k=${k} \
                        +mask_id=126336 \
                        sampling.alpha_on=0.9 \
                        sampling.t_on=0.55 \
                        sampling.t_off=0.9 \
                        sampling.eta=0.02 \
                        +cfg=0 \
                        +remasking=low_confidence \
                        +eos_fill=False \
                        +category=large8B_trained_gsm8k_20250331 \
                        +eval_data=gsm8k \
                        +generator=llada_inst_${d_type}${k_suffix}_temp1_L512_S${S}${bs_suffix}${bi_suffix} \
                        master_port=65535 \
                        +rand_value=870${counter} \
                        +batch_size=1



                    done
            done
        done
done


