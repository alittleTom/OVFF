import os
import random

#生成trainList和testList
train_txt = './train_lists.txt'
test_txt = './test_lists.txt'
list_path = 'yourListPath'
train_p = 0.8  #训练数据集占80%

with open(train_txt,'w') as train:
    with open(test_txt,'w') as test:
        pose_id = 0
        for pose in os.listdir(list_path):
            pose_path = os.path.join(list_path,pose)
            subjects = os.listdir(pose_path)
            sub_num = len(subjects)
            sub_train_num = int(sub_num*train_p)

            random.shuffle(subjects)

            sub_train = subjects[0:sub_train_num]
            sub_test = subjects[sub_train_num:]

            for sub in sub_train:
                train.write('list/{}/{}    {}'.format(pose,sub,pose_id))
                train.write('\n')
            for sub in sub_test:
                test.write('list/{}/{}    {}'.format(pose,sub,pose_id))
                test.write('\n')
            pose_id = pose_id+1
