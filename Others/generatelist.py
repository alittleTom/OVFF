import os
# 根据view生成 list
view_path = 'yourDataViews'
list_view = 'yourListPath'

view_num = 20

def Write_txt(file_name, label, view_num, views_context):
    with open(file_name,'w') as f:
        f.write(str(label))
        f.write('\n')
        f.write(str(view_num))
        f.write('\n')
        for view in views_context:
            f.write(view)
            f.write('\n')

pose_id = 0
for pose in os.listdir(view_path):
    pose_path = os.path.join(view_path,pose)
    pose_path_new = os.path.join(list_view, pose)

    if not os.path.exists(pose_path_new):
        os.mkdir(pose_path_new)
    
    views = os.listdir(pose_path)
    model_num = int(len(views)/view_num)
    print(model_num)
    for i in range(model_num):
        model_view = views[i*view_num:i*view_num+view_num]
        views_context = ['view/{}/{}'.format(pose,name) for name in model_view]
        txt_name = '{}_subject_{}.txt'.format(pose,i)
        txt_path = os.path.join(pose_path_new,txt_name)
        Write_txt(txt_path,pose_id,view_num,views_context)
    pose_id = pose_id+1
