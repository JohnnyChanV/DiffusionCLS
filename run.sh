#!/opt/conda/bin/activate


# 1     
# python arg_main.py --dataset_name SST-2-5\
#                     --epochs_ct_model 80 \
#                     --epochs_dif_augmentation_model 180 \
#                     --epochs_dif_model 20\
#                     --bag_size 1 \
#                     --step_group_num 32\
#                     --max_diffusion_step 64\
#                     --plm bert-base-uncased
                    
# # 2             
# python arg_main.py --dataset_name SST-2-5\
#                     --epochs_ct_model 80 \
#                     --epochs_dif_augmentation_model 180 \
#                     --epochs_dif_model 20\
#                     --bag_size 2 \
#                     --step_group_num 16\
#                     --max_diffusion_step 64\
#                     --plm bert-base-uncased
                    
# # # 8 
python arg_main.py --dataset_name IndiaEngCovidDataset-aeda\
                    --epochs_ct_model 15 \
                    --run_type ct \
                    --plm bert-base-uncased
                    
# # 8 
python arg_main.py --dataset_name SST-2-5\
                    --epochs_ct_model 80 \
                    --epochs_dif_augmentation_model 200 \
                    --epochs_dif_model 20\
                    --bag_size 16 \
                    --step_group_num 2\
                    --max_diffusion_step 64\
                    --plm bert-base-uncased

                    
# # 1     
python arg_main.py --dataset_name SST-2-5-aeda-4\
                    --epochs_ct_model 180 \
                    --run_type ct\
                    --plm bert-base-uncased
                    
# # 1     
python arg_main.py --dataset_name SST-2-5-aeda-8\
                    --epochs_ct_model 180 \
                    --run_type ct\
                    --plm bert-base-uncased