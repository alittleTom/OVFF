import vtk
import math
import cv2 as cv
import numpy as np
import os
import shutil

#该脚本用来生成方向不同的数据集

Total_data = 'yourData' #你想要旋转的数据集地址
pose_data = 'transData' #旋转后生成的地址


def tranAndSave(sourceOBJ,newOBJ,axis='X',degree=30):
    objReader = vtk.vtkOBJReader()
    objReader.SetFileName(sourceOBJ)
    objReader.Update()

    #Polydata normal
    polydataNormal = vtk.vtkPolyDataNormals()
    polydataNormal.SetInputConnection(objReader.GetOutputPort())
    polydataNormal.ConsistencyOff()
    polydataNormal.SplittingOff()
    polydataNormal.Update()

    #Polydata mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(polydataNormal.GetOutputPort())

    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.6)
    actor.GetProperty().SetSpecular(0.0)

    trans = vtk.vtkTransform()
    if axis == 'X':
        trans.RotateX(degree)
    elif axis == 'Y':
        trans.RotateY(degree)
    elif axis == 'Z':
        trans.RotateZ(degree)
    else:
        pass

    actor.SetUserTransform(trans)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("obj viewer")

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindow.Render()

    expoter = vtk.vtkOBJExporter()
    expoter.SetFilePrefix(newOBJ)
    expoter.SetInput(renderWindow)
    expoter.Write()


def IsExitClass(classPath):
    if not os.path.exists(classPath):
        os.mkdir(classPath)

if __name__ == '__main__':
    objs = os.listdir(Total_data)
    for obj in objs:
        print(obj)
        obj_name = obj.split('.')[0]
        _class = int(obj_name)%10

        _classPath = os.path.join(pose_data,str(_class*34)) #10个pose,每个pose都会生成33个旋转子类(三个旋转轴，每个旋转11次，每次30度，不能旋转第12次因为这样会一致)加一个未旋转类，所以一共34个子类
        IsExitClass(_classPath)
        obj_Path = os.path.join(Total_data,obj)
        obj_Path_new = os.path.join(_classPath,obj)
        shutil.copyfile(obj_Path,obj_Path_new)

        for i in range(3): #x,y,z三个轴进行旋转
            if i%3 == 0:
                axis = 'X'
            elif i%3 == 1:
                axis = 'Y'
            else:
                axis = 'Z'
            for j in range(1,12): #每个模型在每个轴方向旋转30度，一周正好旋转12次
                _classPath = os.path.join(pose_data,str(_class*34+i*11+j))
                IsExitClass(_classPath)
                obj_Path_new = os.path.join(_classPath,obj_name)
                tranAndSave(obj_Path, obj_Path_new, axis, j*30)

        
