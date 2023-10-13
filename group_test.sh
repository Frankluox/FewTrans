# myarray=(4 0 1 3 5 6 7 8 9)
myarray=(3)
# myarray=(2 3)
# myarray=(4 5)
# myarray=(6 7)
# myarray=(8 9)
gpuid=3

# myarray=(0 1 2 3 4 5 6 7 8 9 10 11 12)
# myarray=(17)
# myarray=(2 3 4 5 6 7 8 9 10 11 12)
# myarray=(13)
vary=0
way=5
shot=1
cross_dataset=0
sample_all=0


# modelname=PN_res18_ImageNet
# modelname=test
# modelname=clip_zero_shot
# modelname=clip_vpt_only_center_hyper_without_epochensemble
# modelname=clip_vpt_better_hyper
# modelname=clip_visualFT_only_center_hyper_without_epochensemble
# modelname=clip_coOp
# modelname=clip_CoCoOp
# modelname=clip_LR
# modelname=clip_ProGrad
# modelname=clip_kgcoop
# modelname=clip_maple_again
# modelname=clip_vpt_visualonly
# modelname=clip_finetune_visualonly_only_center_hyper
# modelname=clip_LoRA_visualonly
# modelname=clip_adapter_visualonly
# modelname=clip_finetune_visualonly_again
# modelname=ImageNetResnet50
# modelname=MAEbase
# modelname=SwinBase
# modelname=IBOT_ViT
# modelname=verify_large_variance_20shot
# modelname=compare_ensemble_epoch
# modelname=compare_ensemble_all_5times
modelname=compare_ensemble
# modelname=compare_ensemble_all
# modelname=DINOsmall16_official
# modelname=PN_ImageNet_ptrainfromDINOsmall16_official

for dataset in ${myarray[@]}
    do
        python write_yaml_test_with_arg_visual_only.py --dataset_id ${dataset} --gpu_id ${gpuid} --vary ${vary} --way ${way} --shot ${shot} --cross_dataset ${cross_dataset} --model_name ${modelname} --sample_all ${sample_all}
        # python write_yaml_test_with_arg.py --dataset_id ${dataset} --gpu_id ${gpuid} --vary ${vary} --way ${way} --shot ${shot} --cross_dataset ${cross_dataset} --model_name ${modelname}
    done
for dataset in ${myarray[@]}
    do
        if [ $vary == 1 ]
        then
            python main.py --cfg configs/PN/PN_singledomain_test_vary_way_vary_shot_${dataset}_${modelname}.yaml --is_train 0 --tag ${modelname}/test_vary_way_vary_shot
            # python main.py --cfg configs/PN/PN_singledomain_test_vary_way_vary_shot_${dataset}.yaml --is_train 0 --pretrained ../data2/new_metadataset_result/ImageNet_PNptrainfromDINOsmall16official/task10000lr5e-6warm2000fromE-6/ckpt_epoch_36_top1.pth --tag ${modelname}/test_vary_way_vary_shot
        else
            python main.py --cfg configs/PN/PN_singledomain_test_${way}w_${shot}s_${dataset}_${modelname}.yaml --is_train 0 --tag ${modelname}/test_${way}way_${shot}shot 
            # python main.py --cfg configs/PN/PN_singledomain_test_${way}w_${shot}s_${dataset}.yaml --is_train 0 --pretrained ../data2/new_metadataset_result/ImageNet_PNptrainfromDINOsmall16official/task10000lr5e-6warm2000fromE-6/ckpt_epoch_36_top1.pth --tag ${modelname}/test_${way}way_${shot}shot
        fi
    done


