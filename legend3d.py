import vtk
import numpy
import os
from PIL import Image, ImageDraw, ImageFont
from vtk.util.numpy_support import vtk_to_numpy

def window2pil_image(window):
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(window)
    w2if.Update()
    numpy_image_data = vtk_to_numpy(w2if.GetOutput().GetPointData().GetArray("ImageScalars"))
    numpy_image_data_reshaped = numpy_image_data.reshape((window.GetSize()[1], window.GetSize()[0],3))
    return Image.fromarray(numpy.flip(numpy_image_data_reshaped,axis=0))

def legend3d(dicom_data,output_directory):
    # contour filter for typical bone density
    contour = vtk.vtkContourFilter()
    contour.SetInputData(dicom_data)
    dicom_data.GetPointData().SetActiveScalars("DICOMImage")
    contour.SetValue(0,500)
    contour.Update()

    # dicom data spatial information
    data_bounds = list(dicom_data.GetBounds())
    data_center = list(dicom_data.GetCenter())
    data_size_x, data_size_y, data_size_z = data_bounds[1]-data_bounds[0], data_bounds[3]-data_bounds[2], data_bounds[5]-data_bounds[4]

    # vtk window and renderer
    window = vtk.vtkRenderWindow()
    renderer = vtk.vtkRenderer()
    window.AddRenderer(renderer)
    window.SetSize(640,480)
    window.OffScreenRenderingOn()

    # background
    renderer.GradientBackgroundOff()
    renderer.SetBackground(0,0,0)

    # vtk mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()

    # vtk actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    camera = renderer.GetActiveCamera()

    # lights
    lightKit = vtk.vtkLightKit()
    lightKit.MaintainLuminanceOff()
    lightKit.SetKeyLightIntensity(1)
    lightKit.SetKeyLightWarmth(0.6)
    lightKit.SetFillLightWarmth(0.5)
    lightKit.SetHeadLightWarmth(0.5)
    lightKit.SetKeyToFillRatio(2)
    lightKit.SetKeyToHeadRatio(4)
    lightKit.SetKeyLightAzimuth(30)
    lightKit.SetKeyLightElevation(45)
    lights = renderer.GetLights()
    lights.InitTraversal()
    lightKit.AddLightsToRenderer(renderer)

    # camera focal point in the middle of data
    camera.SetFocalPoint(data_center)
    camera.SetParallelProjection(True)

    # isometric view rendered to pil image
    max_data_size = max(data_size_x,data_size_y,data_size_z)
    camera.SetPosition([sum(x) for x in zip(data_center, [max_data_size,max_data_size,-max_data_size])])
    camera.SetViewUp([0,0,-1])
    camera.SetParallelScale(max_data_size/1.732)
    window.Render()
    pil_image_iso = window2pil_image(window)

    # view in X+ direction rendered to pil image
    camera.SetPosition([sum(x) for x in zip(data_center, [data_size_x,0,0])])
    camera.SetParallelScale(max(data_size_y,data_size_z)/2)
    window.Render()
    pil_image_x = window2pil_image(window)

    # view in Y+ direction rendered to pil image
    camera.SetPosition([sum(x) for x in zip(data_center, [0,data_size_y,0])])
    camera.SetParallelScale(max(data_size_x,data_size_z)/2)
    window.Render()
    pil_image_y = window2pil_image(window)

    # view in Z+ direction rendered to pil image
    camera.SetPosition([sum(x) for x in zip(data_center, [0,0,-data_size_z])])
    camera.SetViewUp([-1,0,0])
    camera.SetParallelScale(max(data_size_y,data_size_x)/2)
    window.Render()
    pil_image_z = window2pil_image(window)

    # merging images
    legend = Image.new("RGB", (window.GetSize()[0]*2, window.GetSize()[1]*2), (0,0,0))
    legend.paste(pil_image_iso, (0, 0))
    legend.paste(pil_image_x, (window.GetSize()[0], 0))
    legend.paste(pil_image_y, (0, window.GetSize()[1]))
    legend.paste(pil_image_z, (window.GetSize()[0],window.GetSize()[1]))
    # image descriptions
    draw = ImageDraw.Draw(legend)
    draw.text((10,10), "ISO", "magenta",font_size=50)
    draw.text((window.GetSize()[0]+10,10), "X", "magenta",font_size=50)
    draw.text((10,window.GetSize()[1]+10), "Y", "magenta",font_size=50)
    draw.text((window.GetSize()[0]+10,window.GetSize()[1]+10), "Z", "magenta",font_size=50)
    # image saving
    legend.save(os.path.join(output_directory,"legend.png"))
