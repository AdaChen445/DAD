from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import argparse
import pandas as pd
import matplotlib.pyplot as plt

print('[INFO] executing: mk_plot_delta.py')
ap = argparse.ArgumentParser()
ap.add_argument("-n", required=True)
args = vars(ap.parse_args())
model_name = str(args["n"])

model_dir='./log/'+model_name+'/'
plot_dir=model_dir+model_name

df = pd.read_csv(plot_dir+".csv")
df = df.dropna()

anomaly = df.loc[(df['train'] == False), 'anomaly'].to_numpy()
anomaly_score = df.loc[(df['train'] == False), 'anomaly_score'].to_numpy()

# train_normal_AS = df.loc[(df['train'] == True)&(df['anomaly'] == False), 'anomaly_score'].to_numpy()
# train_anomaly_AS = df.loc[(df['train'] == True)&(df['anomaly'] == True), 'anomaly_score'].to_numpy()
test_normal_AS = df.loc[(df['train'] == False)&(df['anomaly'] == False)&(df['ng2ok'] == False), 'anomaly_score'].to_numpy()
test_anomaly_AS = df.loc[(df['train'] == False)&(df['anomaly'] == True), 'anomaly_score'].to_numpy()
test_ng2ok_AS = df.loc[(df['ng2ok'] == True), 'anomaly_score'].to_numpy()


def inverte(x):
	return abs(1-x)

# anomaly_score = inverte(anomaly_score)
# train_normal_AS = inverte(train_normal_AS)
# train_anomaly_AS = inverte(train_anomaly_AS)
# test_normal_AS = inverte(test_normal_AS)
# test_anomaly_AS = inverte(test_anomaly_AS)
# test_ng2ok_AS = inverte(test_ng2ok_AS)



fpr, tpr, _ = roc_curve(anomaly, anomaly_score)
auc_score = str(roc_auc_score(anomaly, anomaly_score))
print('AUC scoe: '+ auc_score)
plt.plot(fpr,tpr)
plt.title('receiver operating characteristic curve')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()
plt.savefig("./"+plot_dir +"_roc_"+ auc_score+ ".png")

precision, recall, thresholds = precision_recall_curve(anomaly, anomaly_score)
fig, ax = plt.subplots()
ax.plot(recall, precision)
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
# plt.show()
plt.savefig("./"+plot_dir +"_pre_recall_"+ auc_score+ ".png")


def normalize(x):
	normalized_x = (x-min(x))/(max(x)-min(x)+1e-9)
	return normalized_x

#test data
# plt.subplots()
# plt.hist(normalize(test_normal_AS), bins=100, alpha=0.5, label='test_ok') #bug
# plt.hist(normalize(test_anomaly_AS), bins=100, alpha=0.5, label='test_ng')
# plt.hist(normalize(test_ng2ok_AS), bins=100, alpha=0.5, label='test_ng2ok')
# # plt.gca().set_yscale("log")
# plt.xlabel("Anomaly Score")
# plt.ylabel("Count")
# plt.title("Distribution Histograms")
# plt.legend(loc='upper right')
# # plt.show()
# plt.savefig("./"+plot_dir +"_test_distribution"+ ".png")

plt.subplots()
plt.hist(normalize(test_normal_AS), bins=100, alpha=0.5, label='test_ok')
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.title("Distribution Histograms")
plt.legend(loc='upper right')
plt.savefig("./"+plot_dir +"_ok"+ ".png")
plt.subplots()
plt.hist(normalize(test_anomaly_AS), bins=100, alpha=0.5, label='test_ng')
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.title("Distribution Histograms")
plt.legend(loc='upper right')
plt.savefig("./"+plot_dir +"_ng"+ ".png")
plt.subplots()
plt.hist(normalize(test_ng2ok_AS), bins=100, alpha=0.5, label='test_ng2ok')
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.title("Distribution Histograms")
plt.legend(loc='upper right')
plt.savefig("./"+plot_dir +"_ng2ok"+ ".png")

# #train data
# plt.subplots()
# plt.hist(normalize(train_normal_AS), bins=100, alpha=0.5, label='train_ok') #bug
# plt.hist(normalize(train_anomaly_AS), bins=100, alpha=0.5, label='train_outlier')
# plt.gca().set_yscale("log")
# plt.xlabel("Anomaly Score")
# plt.ylabel("Count")
# plt.title("Distribution Histograms")
# plt.legend(loc='upper right')
# # plt.show()
# plt.savefig("./"+plot_dir +"_train_distribution"+ ".png")