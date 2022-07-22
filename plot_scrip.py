import subprocess
import os

def tsneplot(feature_type):
	### new label
	# subprocess.run('python plot_feature.py -l ok_1 -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ok_1loud -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ok_2 -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ok_3 -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ng_slap -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ng_loud -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ng_restart -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ng_ss -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ng_short -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l ng_electro -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l n2o_pulse -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l bearing_good -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l bearing_bad -f '+feature_type,shell=True)
	## ori label
	# subprocess.run('python plot_feature.py -l ok -f '+feature_type+' -o temp_img_seperate',shell=True)
	# subprocess.run('python plot_feature.py -l ng -f '+feature_type+' -o temp_img_seperate',shell=True)
	# subprocess.run('python plot_feature.py -l ng2ok -f '+feature_type+' -o temp_img_seperate',shell=True)
	### bearing
	# subprocess.run('python plot_feature.py -l bearing_goodN2O -f '+feature_type,shell=True)
	# subprocess.run('python plot_feature.py -l bearing_badNG -f '+feature_type,shell=True)
	
	### tsne & plot
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model inception',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model xception',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model dense121',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model dense201',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model resnet50v2',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model vgg16',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model ix',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model id',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model xd',shell=True)
	# subprocess.run('python tsne_cluster.py -t new -f '+feature_type +' -mode tsne -model ixd',shell=True)

	### stage1
	subprocess.run('python plot_feature.py -l ok -f '+feature_type +' -o stage1ok', shell=True)
	# subprocess.run('python plot_feature.py -l ok -f '+feature_type +' -o stage1arg' ,shell=True)
	# subprocess.run('python plot_feature.py -l outlier -f '+feature_type +' -o temp_img_serial' ,shell=True)

	# subprocess.run('python tsne_cluster.py -t ori -f '+feature_type +' -mode tsne -model xception',shell=True)
	# subprocess.run('python tsne_cluster.py -t ori -f '+feature_type +' -c ac -mode cluster -model xception', shell=True)
	# subprocess.run('python tsne_cluster.py -t ori -f '+feature_type +' -c db -mode cluster -model xception -eps 0.21', shell=True)
	# subprocess.run('python tsne_cluster.py -t ori -f '+feature_type +' -c db -mode cluster -model xception -eps 0.23', shell=True)
	# subprocess.run('python tsne_cluster.py -t ori -f '+feature_type +' -c db -mode cluster -model xception -eps 0.25', shell=True)

### useless features
# tsneplot('spectrum')
# tsneplot('mel')
# tsneplot('chroma')
# tsneplot('spectral')
# tsneplot('tonnetz')
# tsneplot('melMfcc')
# tsneplot('melMfccChroma')
# tsneplot('melSpecMfccChroma')
# tsneplot('specsc')
# tsneplot('spectonnetz')
# tsneplot('specmfccsc')
# tsneplot('specmfcctonnetz')
# tsneplot('test')
# tsneplot('hpss')
# tsneplot('melMfcc')
# tsneplot('specChroma')
# tsneplot('mfccchroma')
# tsneplot('hmfcc')
# tsneplot('pmfcc')
# tsneplot('mfcc')
# tsneplot('spectrogram')
# tsneplot('specMfccChroma')

### useful features
tsneplot('specMfcc')
# tsneplot('melChroma')


# tsneplot('melCT')
# tsneplot('mfccCT')

#################


### cluster & plot

### all
# for cluster_type in ('ac', 'km'):
# 	os.mkdir(cluster_type)
# 	f_report = open(cluster_type+'/'+'cluster_eval_report.txt', 'w')
# 	f_report.write('cluster type: '+cluster_type+'\n')
# 	f_report.write('================'+'\n')
# 	f_report.close()
# 	for feature_name in ('spectrogram', 'specMfcc', 'specChroma', 'specMfccChroma', 'melSpec', 'melChroma'):
# 		for model_name in ('inception', 'xception', 'dense121', 'dense201', 'resnet50', 'resnet50v2', 'vgg16'):
# 				subprocess.run('python tsne_cluster.py -t new -f '+feature_name +' -c '+cluster_type+' -mode cluster -model '+model_name, shell=True)

### single
# os.mkdir('ac')
# f_report = open('ac'+'/'+'cluster_eval_report.txt', 'w')
# f_report.write('cluster type: '+'ac'+'\n')
# f_report.write('================'+'\n')
# f_report.close()
# subprocess.run('python tsne_cluster.py -t new -f specMfcc -c '+'ac'+' -mode cluster -model dense121', shell=True)

### choisen
# os.mkdir('ac')
# f_report = open('ac'+'/'+'cluster_eval_report.txt', 'w')
# f_report.write('cluster type: '+'ac'+'\n')
# f_report.write('================'+'\n')
# f_report.close()
# for feature_name in ('spectrogram', 'specMfcc', 'melChroma'):
# 	for model_name in ('ix', 'xd', 'ixd'):
# 			subprocess.run('python tsne_cluster.py -t new -f '+feature_name +' -c '+'ac'+' -mode cluster -model '+model_name, shell=True)