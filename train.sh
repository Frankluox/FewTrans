#test
# python write_yaml_test.py
# python main.py --cfg configs/evaluation/finetune_res12_CE.yaml --tag test

# training CE model
# python write_yaml_CE.py
# python write_yaml_PN.py
# python write_yaml_search_clip.py 
# python main.py --cfg configs/CE/miniImageNet_res12.yaml --tag color_jitter
# python main.py --cfg configs/CE/miniImageNet_res12.yaml --tag test
# python main.py --cfg configs/CE/ImageNet_res18.yaml --tag lr0.1
# python main.py --cfg configs/PN/ImageNet_res18.yaml --tag lr0.02
vary=0
# myarray=(1 2 3 4 6 7 8 9)
myarray=(3)
# myarray=(3 4)
# myarray=(6 7)
# myarray=(8 9)
datasets=(ImageNet QuickD VGG Aircraft DTD Fungi CIFAR EuroSAT ucf plantD)
sample_all=0
way=5
shot=1
gpuid=3
# modelname=hyparameter_dataset
# modelname=hyparameter_dataset_5way
# modelname=hyparameter_shot
modelname=compare_ensemble_single_5times
# modelname=compare_ensemble_single
# tag=hyparameter_dataset/
# training PN model
# python write_yaml_PN.py
# python main.py --cfg configs/PN/miniImageNet_res12.yaml --tag main

# searching for hyperparameters for finetune.
# python write_yaml_search.py
# # python write_yaml_search_clip.py
# # python search_hyperparameter.py --cfg configs/search/finetune_res12_CE.yaml  --is_train 0 --tag clip_VPT/coarse_grain
# python search_hyperparameter.py --cfg configs/search/finetune_res12_CE.yaml  --is_train 0 --tag hyparameter_dataset/quickdraw

for dataset in ${myarray[@]}
    do
        python write_yaml_search.py --dataset_id ${dataset} --gpu_id ${gpuid} --vary ${vary} --way ${way} --shot ${shot} --model_name ${modelname} --sample_all ${sample_all}
        # python write_yaml_test_with_arg.py --dataset_id ${dataset} --gpu_id ${gpuid} --vary ${vary} --way ${way} --shot ${shot} --cross_dataset ${cross_dataset} --model_name ${modelname}
    done
for dataset in ${myarray[@]}
    do
        if [ $vary == 1 ]
        then
            python search_hyperparameter.py --cfg configs/PN/PN_singledomain_test_vary_way_vary_shot_${dataset}_${modelname}.yaml --is_train 0 --tag ${modelname}/${datasets[${dataset}]}
            # python search_hyperparameter.py --cfg configs/PN/PN_singledomain_test_vary_way_vary_shot_${dataset}_${modelname}.yaml --is_train 0 --tag ${modelname}/shot_${shot}
            # python main.py --cfg configs/PN/PN_singledomain_test_vary_way_vary_shot_${dataset}.yaml --is_train 0 --pretrained ../data2/new_metadataset_result/ImageNet_PNptrainfromDINOsmall16official/task10000lr5e-6warm2000fromE-6/ckpt_epoch_36_top1.pth --tag ${modelname}/test_vary_way_vary_shot
        else
            python search_hyperparameter.py --cfg configs/PN/PN_singledomain_test_${way}w_${shot}s_${dataset}_${modelname}.yaml --is_train 0 --tag ${modelname}/${datasets[${dataset}]} 
            # python search_hyperparameter.py --cfg configs/PN/PN_singledomain_test_${way}w_${shot}s_${dataset}_${modelname}.yaml --is_train 0 --tag ${modelname}/shot_${shot}
            # python main.py --cfg configs/PN/PN_singledomain_test_${way}w_${shot}s_${dataset}.yaml --is_train 0 --pretrained ../data2/new_metadataset_result/ImageNet_PNptrainfromDINOsmall16official/task10000lr5e-6warm2000fromE-6/ckpt_epoch_36_top1.pth --tag ${modelname}/test_${way}way_${shot}shot
        fi
    done