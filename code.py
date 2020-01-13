import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statistics as stat
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("train.csv")

self_fill__1 = np.zeros((train_data.shape[0]), float)
self_fill__2 = np.zeros((train_data.shape[0]), float)
self_fill__3 = np.zeros((train_data.shape[0]), float)
self_fill__4 = np.zeros((train_data.shape[0]), float)
self_fill__5 = np.zeros((train_data.shape[0]), float)
self_fill__6 = np.zeros((train_data.shape[0]), float)
self_fill__7 = np.zeros((train_data.shape[0]), float)
self_fill__8 = np.zeros((train_data.shape[0]), float)
self_fill__9 = np.zeros((train_data.shape[0]), float)
self_fill__10 = np.zeros((train_data.shape[0]), float)
self_fill__11 = np.zeros((train_data.shape[0]), float)
train_data['softmax'] = self_fill__1
train_data['tanh'] = self_fill__2
train_data['linear'] = self_fill__3
train_data['dropout'] = self_fill__4
train_data['batchnorm'] = self_fill__5
train_data['flatten'] = self_fill__6
train_data['conv'] = self_fill__7
train_data['selu'] = self_fill__8
train_data['leaky_relu'] = self_fill__9
train_data['relu'] = self_fill__10
train_data['maxpool'] = self_fill__11

self_fill_1 = np.zeros((train_data.shape[0]), float)
self_fill_2 = np.zeros((train_data.shape[0]), float)
self_fill_3 = np.zeros((train_data.shape[0]), float)
self_fill_4 = np.zeros((train_data.shape[0]), float)
self_fill_5 = np.zeros((train_data.shape[0]), float)
self_fill_6 = np.zeros((train_data.shape[0]), float)
self_fill_7 = np.zeros((train_data.shape[0]), float)
self_fill_8 = np.zeros((train_data.shape[0]), float)
self_fill_9 = np.zeros((train_data.shape[0]), float)
self_fill_11 = np.zeros((train_data.shape[0]), float)
self_fill_12 = np.zeros((train_data.shape[0]), float)
self_fill_13 = np.zeros((train_data.shape[0]), float)
self_fill_14 = np.zeros((train_data.shape[0]), float)
self_fill_15 = np.zeros((train_data.shape[0]), float)
self_fill_16 = np.zeros((train_data.shape[0]), float)
self_fill_17 = np.zeros((train_data.shape[0]), float)
self_fill_18 = np.zeros((train_data.shape[0]), float)
self_fill_19 = np.zeros((train_data.shape[0]), float)
train_data['init_params_linear_std'] = self_fill_1
train_data['init_params_linear_mu'] = self_fill_2
train_data['init_params_linear_l2'] = self_fill_3
train_data['init_params_batchnorm_std'] = self_fill_4
train_data['init_params_batchnorm_mu'] = self_fill_5
train_data['init_params_batchnorm_l2'] = self_fill_6
train_data['init_params_conv_std'] = self_fill_7
train_data['init_params_conv_mu'] = self_fill_8
train_data['init_params_conv_l2'] = self_fill_9
train_data['init_params_linear_std_2'] = self_fill_11
train_data['init_params_linear_mu_2'] = self_fill_12
train_data['init_params_linear_l2_2'] = self_fill_13
train_data['init_params_batchnorm_std_2'] = self_fill_14
train_data['init_params_batchnorm_mu_2'] = self_fill_15
train_data['init_params_batchnorm_l2_2'] = self_fill_16
train_data['init_params_conv_std_2'] = self_fill_17
train_data['init_params_conv_mu_2'] = self_fill_18
train_data['init_params_conv_l2_2'] = self_fill_19

my_filter = train_data.isna()

for k in range(train_data.shape[0]):
    if my_filter['arch_and_hp'][k] == False:
        if type(train_data['arch_and_hp'][k]) == str:
            train_data['softmax'][k] = train_data['arch_and_hp'][k].count('softmax')
            train_data['tanh'][k] = train_data['arch_and_hp'][k].count('tanh')
            train_data['linear'][k] = train_data['arch_and_hp'][k].count('linear')
            train_data['dropout'][k] = train_data['arch_and_hp'][k].count('dropout')
            train_data['batchnorm'][k] = train_data['arch_and_hp'][k].count('batchnorm')
            train_data['flatten'][k] = train_data['arch_and_hp'][k].count('flatten')
            train_data['conv'][k] = train_data['arch_and_hp'][k].count('conv')
            train_data['selu'][k] = train_data['arch_and_hp'][k].count('selu')
            train_data['leaky_relu'][k] = train_data['arch_and_hp'][k].count('leaky_relu')
            train_data['relu'][k] = train_data['arch_and_hp'][k].count('relu')
            train_data['maxpool'][k] = train_data['arch_and_hp'][k].count('maxpool')
            
            copy = train_data['arch_and_hp'][k]
            count = 0
            if my_filter['init_params_mu'][k] == True or my_filter['init_params_std'][k] == True or my_filter['init_params_std'][k] == True:
                continue
            while copy.find('linear') != -1 or copy.find('batchnorm') != -1 or copy.find('conv') != -1:
                index = len(copy)
                mine = ''
                if copy.find('linear') != -1 and copy.find('linear') < index:
                    index = copy.find('linear')
                    mine = 'linear'
                if copy.find('batchnorm') != -1 and copy.find('batchnorm') < index:
                    index = copy.find('batchnorm')
                    mine = 'batchnorm'
                if copy.find('conv') != -1 and copy.find('conv') < index:
                    index = copy.find('conv')
                    mine = 'conv'
                string = 'init_params_' + mine
                my_list_1 = [float(i) for i in train_data['init_params_mu'][k][1:-1].split(',')]
                my_list_2 = [float(i) for i in train_data['init_params_std'][k][1:-1].split(',')]
                my_list_3 = [float(i) for i in train_data['init_params_l2'][k][1:-1].split(',')]
                train_data[string + '_mu'][k] += my_list_1[2 * count] 
                train_data[string + '_std'][k] += my_list_2[2 * count]
                train_data[string + '_l2'][k] += my_list_3[2 * count] 
                train_data[string + '_mu_2'][k] += my_list_1[2 * count + 1] 
                train_data[string + '_std_2'][k] += my_list_2[2 * count + 1]
                train_data[string + '_l2_2'][k] += my_list_3[2 * count + 1]
                copy = copy[index + 1 : ]
                count += 1

y1 = np.array(train_data['val_error'])
y2 = np.array(train_data['train_error'])

train_data['init_params_linear_l2']

x1 = np.array(train_data[[
        'init_params_linear_mu', 'init_params_linear_std', 'init_params_linear_l2',
        'init_params_batchnorm_mu', 'init_params_batchnorm_std', 'init_params_batchnorm_l2',
        'init_params_conv_mu', 'init_params_conv_std', 'init_params_conv_l2',
        'init_params_linear_mu_2', 'init_params_linear_std_2', 'init_params_linear_l2_2',
        'init_params_batchnorm_mu_2', 'init_params_batchnorm_std_2', 'init_params_batchnorm_l2_2',
        'init_params_conv_mu_2', 'init_params_conv_std_2', 'init_params_conv_l2_2',
        'epochs', 'number_parameters', 
        'softmax', 'tanh', 'linear', 'dropout', 'batchnorm', 'flatten', 'conv', 'selu', 'leaky_relu', 'relu', 'maxpool',
        'val_accs_0', 'val_accs_10', 'val_accs_20', 'val_accs_30', 
        'val_accs_40', 'val_accs_41', 'val_accs_42', 'val_accs_43', 'val_accs_44',
       'val_accs_45', 'val_accs_46', 'val_accs_47', 'val_accs_48', 'val_accs_49',
        'val_losses_0', 'val_losses_10', 'val_losses_20', 'val_losses_30', 
        'val_losses_40', 'val_losses_41', 'val_losses_42',
       'val_losses_43', 'val_losses_44', 'val_losses_45', 'val_losses_46',
       'val_losses_47', 'val_losses_48', 'val_losses_49']])
x2 = np.array(train_data[[
        'init_params_linear_mu', 'init_params_linear_std', 'init_params_linear_l2',
        'init_params_batchnorm_mu', 'init_params_batchnorm_std', 'init_params_batchnorm_l2',
        'init_params_conv_mu', 'init_params_conv_std', 'init_params_conv_l2',
        'init_params_linear_mu_2', 'init_params_linear_std_2', 'init_params_linear_l2_2',
        'init_params_batchnorm_mu_2', 'init_params_batchnorm_std_2', 'init_params_batchnorm_l2_2',
        'init_params_conv_mu_2', 'init_params_conv_std_2', 'init_params_conv_l2_2',
        'epochs', 'number_parameters', 
        'softmax', 'tanh', 'linear', 'dropout', 'batchnorm', 'flatten', 'conv', 'selu', 'leaky_relu', 'relu', 'maxpool',
        'train_accs_0', 'train_accs_10', 'train_accs_20', 'train_accs_30', 
       'train_accs_40', 'train_accs_41', 'train_accs_42', 'train_accs_43', 'train_accs_44',
       'train_accs_45', 'train_accs_46', 'train_accs_47', 'train_accs_48', 'train_accs_49',
        'train_losses_0', 'train_losses_10', 'train_losses_20', 'train_losses_30',
        'train_losses_40', 'train_losses_41',
       'train_losses_42', 'train_losses_43', 'train_losses_44',
       'train_losses_45', 'train_losses_46', 'train_losses_47',
       'train_losses_48', 'train_losses_49']])

# sum_1 = 0
# sum_2 = 0
# model1 = []
# model2 = []
# model1_score_train = []
model1_score_test = []
# model2_score_train = []
model2_score_test = []
for i in range(200):
    regr1 = linear_model.LinearRegression()
    regr2 = linear_model.LinearRegression()
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.05)
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.05)
    regr1.fit(x1_train, y1_train)
    regr2.fit(x2_train, y2_train)
#     model1.append(regr1)
#     model2.append(regr2)
#     y1_predict_train = regr1.predict(x1_train)
    y1_predict_test = regr1.predict(x1_test)
#     y2_predict_train = regr2.predict(x2_train)
    y2_predict_test = regr2.predict(x2_test)
#     model1_score_train.append(r2_score(y1_predict_train, y1_train))
    model1_score_test.append(r2_score(y1_predict_test, y1_test))
#     model2_score_train.append(r2_score(y2_predict_train, y2_train))
    model2_score_test.append(r2_score(y2_predict_test, y2_test))
# avg1 = np.array(model1_score_test) * 0.2 + np.array(model1_score_train) * 0.8
# avg2 = np.array(model2_score_test) * 0.2 + np.array(model2_score_train) * 0.8
(stat.mean(model1_score_test), stat.mean(model2_score_test))

regr1 = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()
regr1.fit(x1, y1)
regr2.fit(x2, y2)

y1_predict = regr1.predict(x1)
r2_score(y1_predict, y1)

y2_predict = regr2.predict(x2)
r2_score(y2_predict, y2)

test_data = pd.read_csv("test.csv")

self_fill__1 = np.zeros((test_data.shape[0]), float)
self_fill__2 = np.zeros((test_data.shape[0]), float)
self_fill__3 = np.zeros((test_data.shape[0]), float)
self_fill__4 = np.zeros((test_data.shape[0]), float)
self_fill__5 = np.zeros((test_data.shape[0]), float)
self_fill__6 = np.zeros((test_data.shape[0]), float)
self_fill__7 = np.zeros((test_data.shape[0]), float)
self_fill__8 = np.zeros((test_data.shape[0]), float)
self_fill__9 = np.zeros((test_data.shape[0]), float)
self_fill__10 = np.zeros((test_data.shape[0]), float)
self_fill__11 = np.zeros((test_data.shape[0]), float)
test_data['softmax'] = self_fill__1
test_data['tanh'] = self_fill__2
test_data['linear'] = self_fill__3
test_data['dropout'] = self_fill__4
test_data['batchnorm'] = self_fill__5
test_data['flatten'] = self_fill__6
test_data['conv'] = self_fill__7
test_data['selu'] = self_fill__8
test_data['leaky_relu'] = self_fill__9
test_data['relu'] = self_fill__10
test_data['maxpool'] = self_fill__11

self_fill_1 = np.zeros((test_data.shape[0]), float)
self_fill_2 = np.zeros((test_data.shape[0]), float)
self_fill_3 = np.zeros((test_data.shape[0]), float)
self_fill_4 = np.zeros((test_data.shape[0]), float)
self_fill_5 = np.zeros((test_data.shape[0]), float)
self_fill_6 = np.zeros((test_data.shape[0]), float)
test_data['init_params_linear'] = self_fill_3
# train_data['init_params_linear_2'] = self_fill_4
test_data['init_params_batchnorm'] = self_fill_1
# train_data['init_params_batchnorm_2'] = self_fill_2
test_data['init_params_conv'] = self_fill_5
# train_data['init_params_conv_2'] = self_fill_6

my_filter = test_data.isna()

for k in range(test_data.shape[0]):
    if my_filter['arch_and_hp'][k] == False:
        if type(test_data['arch_and_hp'][k]) == str:
            test_data['softmax'][k] = test_data['arch_and_hp'][k].count('softmax')
            test_data['tanh'][k] = test_data['arch_and_hp'][k].count('tanh')
            test_data['linear'][k] = test_data['arch_and_hp'][k].count('linear')
            test_data['dropout'][k] = test_data['arch_and_hp'][k].count('dropout')
            test_data['batchnorm'][k] = test_data['arch_and_hp'][k].count('batchnorm')
            test_data['flatten'][k] = test_data['arch_and_hp'][k].count('flatten')
            test_data['conv'][k] = test_data['arch_and_hp'][k].count('conv')
            test_data['selu'][k] = test_data['arch_and_hp'][k].count('selu')
            test_data['leaky_relu'][k] = test_data['arch_and_hp'][k].count('leaky_relu')
            test_data['relu'][k] = test_data['arch_and_hp'][k].count('relu')
            test_data['maxpool'][k] = test_data['arch_and_hp'][k].count('maxpool')
            
            copy = test_data['arch_and_hp'][k]
            count = 0
            if my_filter['init_params_mu'][k] == True or my_filter['init_params_std'][k] == True or my_filter['init_params_std'][k] == True:
                continue
            while copy.find('linear') != -1 or copy.find('batchnorm') != -1 or copy.find('conv') != -1:
                index = len(copy)
                mine = ''
                if copy.find('linear') != -1 and copy.find('linear') < index:
                    index = copy.find('linear')
                    mine = 'linear'
                if copy.find('batchnorm') != -1 and copy.find('batchnorm') < index:
                    index = copy.find('batchnorm')
                    mine = 'batchnorm'
                if copy.find('conv') != -1 and copy.find('conv') < index:
                    index = copy.find('conv')
                    mine = 'conv'
                string = 'init_params_' + mine
#                 string2 = 'init_params_' + mine + '_2'
                my_list_1 = [float(i) for i in test_data['init_params_mu'][k][1:-1].split(',')]
                my_list_2 = [float(i) for i in test_data['init_params_std'][k][1:-1].split(',')]
                my_list_3 = [float(i) for i in test_data['init_params_l2'][k][1:-1].split(',')]
                test_data[string][k] += (my_list_1[2 * count] + my_list_2[2 * count] + my_list_3[2 * count] + my_list_1[2 * count + 1] + my_list_2[2 * count + 1] + my_list_3[2 * count + 1]) / test_data[mine][k] 
#                 train_data[string2][k] += (my_list_1[2 * count + 1] + my_list_2[2 * count + 1] + my_list_3[2 * count + 1])
                copy = copy[index + 1 : ]
                count += 1

test_x1 = np.array(test_data[[
        'init_params_linear', 'init_params_batchnorm', 'init_params_conv',
        'epochs', 'number_parameters', 
        'softmax', 'tanh', 'linear', 'dropout', 'batchnorm', 'flatten', 'conv', 'selu', 'leaky_relu', 'relu', 'maxpool',
        'val_accs_0', 'val_accs_10', 'val_accs_20', 'val_accs_30', 
        'val_accs_40', 'val_accs_41', 'val_accs_42', 'val_accs_43', 'val_accs_44',
       'val_accs_45', 'val_accs_46', 'val_accs_47', 'val_accs_48', 'val_accs_49',
        'val_losses_0', 'val_losses_10', 'val_losses_20', 'val_losses_30', 
        'val_losses_40', 'val_losses_41', 'val_losses_42',
       'val_losses_43', 'val_losses_44', 'val_losses_45', 'val_losses_46',
       'val_losses_47', 'val_losses_48', 'val_losses_49']])
test_x2 = np.array(test_data[[
        'init_params_linear', 'init_params_batchnorm', 'init_params_conv',
        'epochs', 'number_parameters', 
        'softmax', 'tanh', 'linear', 'dropout', 'batchnorm', 'flatten', 'conv', 'selu', 'leaky_relu', 'relu', 'maxpool',
        'train_accs_0', 'train_accs_10', 'train_accs_20', 'train_accs_30', 
       'train_accs_40', 'train_accs_41', 'train_accs_42', 'train_accs_43', 'train_accs_44',
       'train_accs_45', 'train_accs_46', 'train_accs_47', 'train_accs_48', 'train_accs_49',
        'train_losses_0', 'train_losses_10', 'train_losses_20', 'train_losses_30',
        'train_losses_40', 'train_losses_41',
       'train_losses_42', 'train_losses_43', 'train_losses_44',
       'train_losses_45', 'train_losses_46', 'train_losses_47',
       'train_losses_48', 'train_losses_49']])

test_y1 = regr1.predict(test_x1)
test_y2 = regr2.predict(test_x2)

y_pred = []
for i in range(test_y1.shape[0]):
    y_pred.append(['test_' + str(i) + '_val_error', test_y1[i]])
    y_pred.append(['test_' + str(i) + '_train_error', test_y2[i]])

prediction = pd.DataFrame(y_pred, columns = ['id', 'Predicted'])

prediction.to_csv("result.csv", index=False)
