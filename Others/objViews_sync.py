#!/usr/bin/env python
# -*- coding: utf-8 -*-

import vtk
import math
import cv2 as cv
import numpy as np
from vtk.util import numpy_support

phi = (1+math.sqrt(5))/2
vertices = [
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
        
    [0, 1/phi, phi],
    [0, 1/phi, -phi],
    [0, -1/phi, phi],
    [0, -1/phi, -phi],
        
    [phi, 0, 1/phi],
    [phi, 0, -1/phi],
    [-phi, 0, 1/phi],
    [-phi, 0, -1/phi],
        
    [1/phi, phi, 0],
    [-1/phi, phi, 0],
    [1/phi, -phi, 0],
    [-1/phi, -phi, 0]
    ]


def vtkImgToNumpyArray(vtkImageData):
    rows, cols, _ = vtkImageData.GetDimensions()
    scalars = vtkImageData.GetPointData().GetScalars()
    resultingNumpyArray = numpy_support.vtk_to_numpy(scalars)
    resultingNumpyArray = resultingNumpyArray.reshape(cols, rows, -1)
    red, green, blue, alpha = np.dsplit(resultingNumpyArray, resultingNumpyArray.shape[-1])
    resultingNumpyArray = np.stack([blue, green, red, alpha], 2).squeeze()
    resultingNumpyArray = np.flip(resultingNumpyArray, 0)
    return resultingNumpyArray

def resize_image(img, outputSize, minMargin, maxArea):
    nCh = img.shape[2]
    max_len = outputSize * (1 - minMargin)
    max_area = outputSize * outputSize * maxArea

    img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img1, 254, 255, cv.THRESH_BINARY_INV)

    [ys, xs] = np.where(mask > 0)
    y_min = min(ys)
    y_max = max(ys)
    h = y_max - y_min + 1
    x_min = min(xs)
    x_max = max(xs)
    w = x_max - x_min + 1
    scale = min(max_len / max(h, w), math.sqrt(max_area / sum(sum(mask))))
    ii = img[y_min:(y_max + 1), x_min:(x_max + 1), :]
    patch = cv.resize(ii, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    h = patch.shape[0]
    w = patch.shape[1]

    im = np.ones([outputSize, outputSize, nCh], np.uint8)
    im = 255 * im
    loc_start1 = math.floor((outputSize - h) / 2.0)
    loc_start2 = math.floor((outputSize - w) / 2.0)
    xx = np.arange(0, h) + h
    yy = np.arange(0, w) + w
    im[loc_start1:(loc_start1 + h), loc_start2:(loc_start2 + w), :] = patch

    return im


def objToViews_sync(objFloderName, viewFloderName, imageType, objFileName):
    #Obj reader
    objReader = vtk.vtkOBJReader()
    # objReader = vtk.vtkPLYReader()
    flieName = objFloderName + '/' + objFileName
    objReader.SetFileName(flieName)
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

    #需要旋转的话用
    # trans = vtk.vtkTransform()
    # trans.RotateZ(90)
    # trans.RotateX(-90)
    # actor.SetUserTransform(trans)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("obj viewer")

    minMargin = 0.1
    maxArea = 0.3
    outputSize = 227
    for i in range(0,len(vertices)):
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(1.0, 1.0, 1.0)

        camera = vtk.vtkCamera()
        camera.SetPosition(vertices[i])
        camera.SetFocalPoint(0,0,0)
        camera.SetViewAngle(60)
        # camera.Roll(j*90)

        renderer.SetActiveCamera(camera)
        renderer.ResetCameraClippingRange()

        renderWindow.AddRenderer(renderer)
        renderWindow.Render()

        # Window to image filter
        winToImageFilter = vtk.vtkWindowToImageFilter()
        winToImageFilter.SetInput(renderWindow)
        winToImageFilter.SetScale(2)
        winToImageFilter.SetInputBufferTypeToRGBA()
        winToImageFilter.Update()

        img_vtk = winToImageFilter.GetOutput()
        img_np = vtkImgToNumpyArray(img_vtk)

        img_result = resize_image(img_np, outputSize, minMargin, maxArea)
        # viewNum = "%03d"%(i*4+j)
        viewNum = "%03d"%(i)
        filename = viewFloderName + '/' + objFileName[0:len(objFileName)-4] + '_' + viewNum + imageType
        cv.imwrite(filename,img_result)
            




if __name__ == '__main__':
    objToViews('E:/vtkTest','0_f.obj','E:/vtkTest', '.bmp')