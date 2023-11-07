import numpy as np
import pandas as pd

columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
           'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'])
features_to_encode = ['protocol_type', 'service', 'flag']
# 此行為扣掉features_to_encode和attack和label
col_feature = (['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'])

# 讀檔
df_train = pd.read_csv('D:/python/AE/KDDTrain+.csv')
df_test = pd.read_csv('D:/python/AE/KDDTest+.csv')

df_train.columns = columns
df_test.columns = columns


label = []
# 將其餘加入每個class最少的子項
# for i in df_train.attack:
#     if i == 'normal':
#         label.append(0)
#     elif i in ['back']:
#         label.append(1)
#     elif i in ['neptune']:
#         label.append(2)
#     elif i in ['pod', 'land', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']:
#         label.append(3)
#     elif i in ['smurf']:
#         label.append(4)
#     elif i in ['teardrop']:
#         label.append(5)
#     elif i in ['ipsweep']:
#         label.append(6)
#     elif i in ['nmap', 'mscan', 'saint']:
#         label.append(7)
#     elif i in ['portsweep']:
#         label.append(8)
#     elif i in ['satan']:
#         label.append(9)
#     elif i in ['warezmaster',  'ftp_write', 'guess_passwd', 'phf', 'multihop', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named', 'imap', 'spy']:
#         label.append(10)
#     elif i in ['warezclient']:
#         label.append(11)
#     elif i in ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']:
#         label.append(12)

# 將其餘刪除
for i in df_train.attack:
    if i == 'normal':
        label.append(0)
    elif i in ['back']:
        label.append(1)
    elif i in ['neptune']:
        label.append(2)
    elif i in ['pod']:
        label.append(3)
    elif i in ['smurf']:
        label.append(4)
    elif i in ['teardrop']:
        label.append(5)
    elif i in ['ipsweep']:
        label.append(6)
    elif i in ['nmap']:
        label.append(7)
    elif i in ['portsweep']:
        label.append(8)
    elif i in ['satan']:
        label.append(9)
    elif i in ['guess_passwd']:
        label.append(10)
    elif i in ['warezclient']:
        label.append(11)
    elif i in ['buffer_overflow']:
        label.append(12)
    else:
        label.append(13)

df_train['label'] = label
df_train.drop('attack', axis=1, inplace=True)
df_train.drop('level', axis=1, inplace=True)

label = []
# for i in df_test.attack:
#     if i == 'normal':
#         label.append(0)
#     elif i in ['neptune', 'smurf', 'pod', 'back', 'teardrop', 'land', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']:
#         label.append(1)
#     elif i in ['satan', 'portsweep', 'ipsweep', 'nmap', 'mscan', 'saint']:
#         label.append(2)
#     elif i in ['warezmaster', 'warezclient', 'ftp_write', 'guess_passwd', 'phf', 'multihop', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named', 'imap', 'spy']:
#         label.append(3)
#     elif i in ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']:
#         label.append(4)
#     else:
#         label.append('nono')

for i in df_test.attack:
    if i == 'normal':
        label.append(0)
    elif i in ['back']:
        label.append(1)
    elif i in ['neptune']:
        label.append(2)
    elif i in ['pod', 'land', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']:
        label.append(3)
    elif i in ['smurf']:
        label.append(4)
    elif i in ['teardrop']:
        label.append(5)
    elif i in ['ipsweep']:
        label.append(6)
    elif i in ['nmap', 'mscan', 'saint']:
        label.append(7)
    elif i in ['portsweep']:
        label.append(8)
    elif i in ['satan']:
        label.append(9)
    elif i in ['warezmaster',  'ftp_write', 'guess_passwd', 'phf', 'multihop', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named', 'imap', 'spy']:
        label.append(10)
    elif i in ['warezclient']:
        label.append(11)
    elif i in ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']:
        label.append(12)

df_test['label'] = label
df_test.drop('attack', axis=1, inplace=True)
df_test.drop('level', axis=1, inplace=True)
# 先各個做one hot
train_1 = pd.get_dummies(df_train[features_to_encode])
test_1 = pd.get_dummies(df_test[features_to_encode])


# test和train不會完全相同，找出不同的值
test_index = np.arange(len(df_test.index))
column_diffs_test_train = list(
    set(train_1 .columns.values)-set(test_1.columns.values))


diff_test = pd.DataFrame(0, index=test_index, columns=column_diffs_test_train)

column_order = train_1.columns.to_list()

# append the new columns
test_encoded_temp = test_1.join(diff_test)

# 填滿NA
test_final = test_encoded_temp[column_order].fillna(0)

#
train = train_1.join(df_train[col_feature])
test = test_final.join(df_test[col_feature])

train.to_csv('D:/python/AE/preKDDtrain_dnn3.csv')
test.to_csv('D:/python/AE/preKDDtest_dnn2.csv')
