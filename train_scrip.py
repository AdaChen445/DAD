import subprocess
import os

## test
# subprocess.run('python tl.py -n tt -m xception -d stage2c3FFT -l mse -e 50 -c True',shell=True)
# subprocess.run('python tl.py -n tt -m xception -d ok-ng -l mse -e 50',shell=True)
# subprocess.run('python tl.py -n tt -m vit -d ok-ng -l mse -e 50',shell=True)
# subprocess.run('python tl.py -n tt -m deit -d ok-ng -l mse -e 2',shell=True)
# subprocess.run('python tl.py -n tt -m cait -d ok-ng -l mse -e 2',shell=True)
# subprocess.run('python tl.py -n tt -m stack -d ok-ng -l mse -e 50',shell=True)




## stage1

### ok-outlier
# subprocess.run('python tl.py -n xcep_tl20_stage1outlier -d stage1outlier -e 20',shell=True)
# subprocess.run('python mk_csv_delta.py -n xcep_tl20_stage1outlier',shell=True)
# subprocess.run("python mk_plot_delta.py -n xcep_tl20_stage1outlier",shell=True)

### SM ok-ng
# subprocess.run('python tl.py -n xceptl50_n2ocleaner_ok-ng -d ok-ng -e 50 -tt 200',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_ok1-ng -d ok1-ng -e 50 -tt 200',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_ok2-ng -d ok2-ng -e 50 -tt 200',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_ok3-ng -d ok3-ng  -e 50 -tt 200',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_ok4-ng -d ok4-ng  -e 50 -tt 200',shell=True)
# subprocess.run('python tl.py -n tfdeit50_stage1_ok-ng -m deit -d ok-ng -l mse -e 50',shell=True)
# subprocess.run('python tl.py -n tfcait50_stage1_ok-ng -m cait -d ok-ng -l mse -e 50',shell=True)
# subprocess.run('python tl.py -n tfvit50_stage1_ok-ng -m vit -d ok-ng -l mse -e 50',shell=True)

### MC ok-ng
# subprocess.run('python tl.py -n xceptl50_stage1_MC_ok-ng -d MC_ok-ng -e 50 -tt 140',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MC_ok1-ng -d MC_ok1-ng -e 50 -tt 140',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MC_ok2-ng -d MC_ok2-ng -e 50 -tt 140',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MC_ok3-ng -d MC_ok3-ng -e 50 -tt 140',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MC_ok4-ng -d MC_ok4-ng -e 50 -tt 140',shell=True)

### MM ok-ng
# subprocess.run('python tl.py -n xceptl50_stage1_MM_ok-ng -d MM_ok-ng -e 50 -tt 168',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MM_ok1-ng -d MM_ok1-ng -e 50 -tt 168',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MM_ok2-ng -d MM_ok2-ng -e 50 -tt 168',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MM_ok3-ng -d MM_ok3-ng -e 50 -tt 168',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MM_ok4-ng -d MM_ok4-ng -e 50 -tt 168',shell=True)
# subprocess.run('python tl.py -n xceptl50_stage1_MM_ok5-ng -d MM_ok5-ng -e 50 -tt 168',shell=True)





## stage2

### custom 3-channel dataset
# subprocess.run('python tl.py -n SOTAc3SMFFT -m xception -d stage2c3FFT -e 100 -c True',shell=True) #1024/512/2048
# subprocess.run('python tl.py -n SOTAc3MCFFT -m xception -d stage2c3MCFFT -e 100 -c True',shell=True) #512/1024/2048

### vit
# subprocess.run('vit.py',shell=True)
# subprocess.run('python tl.py -n tfvitb16_c3SMfft_mse -m vit -d stage2c3FFT -e 100 -c True',shell=True)
# subprocess.run('python tl.py -n tfvitb32_c3SMfft_mse -m vit -d stage2c3FFT -e 50 -c True',shell=True)

### models
# subprocess.run('python ts.py -n xception_ts50_stage2_poisson -m xception -d stage2 -l poisson -e 50',shell=True)
# subprocess.run('python ts.py -n effiv2l_ts50_stage2_poisson -m effiv2l -d stage2 -l poisson -e 50',shell=True)
##
# subprocess.run('python tl.py -n effiv2l_tl20_stage2_log_cosh -m effiv2l -d stage2 -l log_cosh -e 20',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_log_cosh -m xception -d stage2 -l log_cosh -e 20',shell=True)

### loss functions
# subprocess.run('python ts.py -n xception_ts20_stage2_poisson -m xception -d stage2 -l Poisson',shell=True)
# subprocess.run('python ts.py -n xception_ts20_stage2_kld -m xception -d stage2 -l kld',shell=True)
# subprocess.run('python ts.py -n xception_ts20_stage2_cosine_similarity  -m xception -d stage2 -l cosine_similarity ',shell=True)
# subprocess.run('python ts.py -n xception_ts20_stage2_binary_crossentropy -m xception -d stage2 -l binary_crossentropy',shell=True)
##
# subprocess.run('python tl.py -n 2xception_tl20_stage2_mean_squared_error  -m xception -d stage2 -l mean_squared_error -e 20',shell=True)
# subprocess.run('python tl.py -n 2xception_tl20_stage2_huber  -m xception -d stage2 -l huber -e 20',shell=True)
# subprocess.run('python tl.py -n 2xception_tl20_stage2_log_cosh -m xception -d stage2 -l log_cosh -e 20',shell=True)
# subprocess.run('python tl.py -n 2xception_tl20_stage2_hinge -m xception -d stage2 -l hinge -e 20',shell=True)