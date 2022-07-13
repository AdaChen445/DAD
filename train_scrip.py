import subprocess
import os

### test
# subprocess.run('python tl.py -n tt -m xception -d stage2c3FFT -l mse -e 50 -c True',shell=True)
# subprocess.run('python tl.py -n tt -m vit -d stage2c3FFT -l mse -e 50 -c True',shell=True)
# subprocess.run('python tl.py -n tt -m stack -d stage2c3FFT -l mse -e 50 -c True',shell=True)
# subprocess.run('python tl.py -n tt -m xception -d n2ocleaner -l mse -e 20',shell=True)




### stage1
# subprocess.run('python tl.py -n xcep_tl20_stage1outlier -m xception -d stage1outlier -l mse -e 20',shell=True)
# subprocess.run('python mk_csv_delta.py -n xcep_tl20_stage1outlier',shell=True)
# subprocess.run("python mk_plot_delta.py -n xcep_tl20_stage1outlier",shell=True)

# subprocess.run('python tl.py -n xcep_tl20_stage1arg -m xception -d stage1arg -l mse -e 20',shell=True)
# subprocess.run('python mk_csv_delta.py -n xcep_tl20_stage1arg',shell=True)
# subprocess.run("python mk_plot_delta.py -n xcep_tl20_stage1arg",shell=True)

subprocess.run('python tl.py -n xcep_tl50_stage1_cleaned -m xception -d stage1 -l mse -e 50',shell=True)
# subprocess.run('python train.py -n mynet -d stage1 -e 50',shell=True)

# subprocess.run('python tl.py -n xcep_tl100_n2ocleaner_db0.21 -m xception -d n2ocleaner -l mse -e 100',shell=True)
# subprocess.run('python train.py -n mynet_n2ocleaner_mel -d n2ocleanerMel -e 200',shell=True)





### stage2 SOTA
# subprocess.run('python tl.py -n SOTAc3SMFFT -m xception -d stage2c3FFT -l mse -e 100 -c True',shell=True) #1024/512/2048
# subprocess.run('python tl.py -n SOTAc3MCFFT -m xception -d stage2c3MCFFT -l mse -e 100 -c True',shell=True) #512/1024/2048

### vit
# subprocess.run('vit.py',shell=True)
# subprocess.run('python tl.py -n tfvitb16_c3SMfft_mse -m vit -d stage2c3FFT -l mse -e 100 -c True',shell=True)
# subprocess.run('python tl.py -n tfvitb32_c3SMfft_mse -m vit -d stage2c3FFT -l mse -e 50 -c True',shell=True)

### nFFT
# subprocess.run('python tl.py -n xception_tl20_stage2512_mae -m xception -d stage2512 -l mse -e 20 ',shell=True)
# subprocess.run('python tl.py -n xception_tl40_stage22048_mae -m xception -d stage22048 -l mse -e 40 ',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage24096_mae -m xception -d stage24096 -l mse -e 20 ',shell=True)

### custom 3-channel dataset
# subprocess.run('python tl.py -n xception_tl50_stage2c3SMmc_mse -m xception -d stage2c3SMmc -l mse -e 50',shell=True) #spec/mfcc/melchroma
# subprocess.run('python tl.py -n xception_tl20_stage2c3hFFT_mae -m xception -d stage2c3hFFT -l mse -e 20 -c True',shell=True) #1024/4096/2048
# subprocess.run('python tl.py -n xception_tl50_stage2c3SM_mse -m xception -d stage2c3SM -l mse -e 50 -c True',shell=True) #spec/mfcc/specmfcc
# subprocess.run('python tl.py -n xception_tl50_stage2c3MC_mse -m xception -d stage2c3MC -l mse -e 50 -c True',shell=True) #mel/chroma/melchroma
# subprocess.run('python tl.py -n 2xception_tl100_stage2c3FFT_mse -m xception -d stage2c3FFT -l mse -e 100 -c True',shell=True) #1024/512/2048
# subprocess.run('python tl.py -n 2xception_tl100_stage2c3MCFFT_mse -m xception -d stage2c3MCFFT -l mse -e 100 -c True',shell=True) #512/1024/2048
# subprocess.run('python tl.py -n 2xception_tl100_stage2c3MMFFT_mse -m xception -d stage2c3MMFFT -l mse -e 100 -c True',shell=True) #512/1024/2048
# subprocess.run('python tl.py -n xception_tl100_stage2c3MCqtFFT_mse -m xception -d stage2c3MCqtFFT -l mse -e 100 -c True',shell=True) #512/1024/2048

### datasets
# subprocess.run('python tl.py -n xception_tl40_stage2_mae -m xception -d stage2 -l mse -e 40 ',shell=True)
# subprocess.run('python tl.py -n xception_tl40_stage2melchroma_mae -m xception -d stage2melchroma -l mse -e 40 ',shell=True)
# subprocess.run('python tl.py -n xception_tl40_stage2spectorgram_mae -m xception -d stage2spectrogram -l mse -e 40 ',shell=True)
# subprocess.run('python tl.py -n xception_tl40_stage2smc_mae -m xception -d stage2smc -l mse -e 40 ',shell=True) #spec/mel/chroma
# subprocess.run('python tl.py -n xception_tl40_stage2ssc_mae -m xception -d stage2ssc -l mse -e 40 ',shell=True) #spec/spectralcontrast
# subprocess.run('python tl.py -n xception_tl100_stage2hp_mae -m xception -d stage2hp -l mse -e 100 ',shell=True) #hpss
# subprocess.run('python tl.py -n xception_tl100_stage2mc_mae -m xception -d stage2mc -l mse -e 100 ',shell=True) #mfcc/chroma
# subprocess.run('python tl.py -n xception_tl100_stage2sc_mae -m xception -d stage2sc -l mse -e 100 ',shell=True) #spec/chroma
# subprocess.run('python tl.py -n xception_tl100_stage2mm_mae -m xception -d stage2mm -l mse -e 100 ',shell=True) #mel/mfcc
# subprocess.run('python tl.py -n xception_tl100_stage2hm_mae -m xception -d stage2hm -l mse -e 100 ',shell=True) #harmonic/mfcc
# subprocess.run('python tl.py -n xception_tl100_stage2pm_mae -m xception -d stage2pm -l mse -e 100 ',shell=True) #precrasive/mfcc
# subprocess.run('python tl.py -n xception_tl50_stage2MCqt_mse -m xception -d stage2MCqt -l mse -e 50',shell=True) #512/1024/2048


### classy report
# subprocess.run('python classify_report.py -n effiv2l_ts100_stage2_poisson -d stage2 -b 0.88',shell=True)
# subprocess.run('python classify_report.py -n xception_ts100_stage2_poisson -d stage2 -b 0.85',shell=True)

### freq masking
# subprocess.run('python ts.py -n xception_ts50_fm_stage2 -m xception -d stage2',shell=True)
# subprocess.run('python ts.py -n efficientb7_ts50_fm_stage2 -m effib7 -d stage2',shell=True)
# subprocess.run('python ts.py -n effiv2l_ts100_fm_stage2_poisson -m effiv2l -d stage2 -l poisson',shell=True)
# subprocess.run('python ts.py -n effiv2l_ts100_fm_stage2smc_poisson -m effiv2l -d stage2smc -l poisson',shell=True)


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

### resize method
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_bicubic -m xception -d stage2 -l mean_squared_error -e 20 -tt 1',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_bilinear -m xception -d stage2 -l mean_squared_error -e 20 -tt 2',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_hamming -m xception -d stage2 -l mean_squared_error -e 20 -tt 3',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_lanczos -m xception -d stage2 -l mean_squared_error -e 20 -tt 4',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_box -m xception -d stage2 -l mean_squared_error -e 20 -tt 5',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_nearest -m xception -d stage2 -l mean_squared_error -e 20 -tt 6',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_cvcubic -m xception -d stage2 -l mean_squared_error -e 20 -tt 7',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_cvarea -m xception -d stage2 -l mean_squared_error -e 20 -tt 8',shell=True)
# subprocess.run('python tl.py -n xception_tl20_stage2_mse_cvnearest -m xception -d stage2 -l mean_squared_error -e 20 -tt 9',shell=True)