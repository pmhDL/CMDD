import os

def run_exp(shot=1, query=15, lr=0.0001, eps=0.1, mg=0.1, coef=1.0, ld1=0.5, ld2=0.5):
    way = 5
    gpu = 0
    dataname = 'mini' #mini, tiered, cifar_fs, cub
    modelname = 'wrn28' #wrn28, res12

    the_command = 'python3 main.py' \
                  + ' --gpu=' + str(gpu) \
                  + ' --shot=' + str(shot) \
                  + ' --val_query=' + str(query) \
                  + ' --way=' + str(way) \
                  + ' --lr=' + str(lr) \
                  + ' --eps=' + str(eps) \
                  + ' --ld1=' + str(ld1) \
                  + ' --ld2=' + str(ld2) \
                  + ' --coef=' + str(coef) \
                  + ' --stopmargin=' + str(mg) \
                  + ' --model_type=' + modelname \
                  + ' --dataset=' + dataname \
                  + ' --dataset_dir=' + '/data/'+dataname+'/'+modelname \

    os.system(the_command + ' --phase=cmdd')

run_exp(shot=1, query=15, lr=1e-3, eps=0.01, mg=1e-2, coef=0.5, ld1=1, ld2=0.05)
