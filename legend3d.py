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

def legend3d(dicom_data,output_directory,transfer_function,value_range_min,value_range_max):
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
    window.SetSize(1280,1200)
    window.OffScreenRenderingOn()

    # background
    renderer.GradientBackgroundOff()
    renderer.SetBackground(0,0,0)

    # scalar bar text property
    scalarBar_tp = vtk.vtkTextProperty()
    scalarBar_tp.SetFontSize(10)
    scalarBar_tp.SetColor((1,0,1))
    scalarBar_tp.SetFontFamilyToArial()
    scalarBar_tp.BoldOff()
    scalarBar_tp.ItalicOff()
    scalarBar_tp.ShadowOff()

    # scalar bar definition
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(transfer_function)
    scalarBar.SetOrientationToHorizontal()
    scalarBar.SetLabelTextProperty(scalarBar_tp)
    scalarBar.SetLabelFormat("%.1f")
    scalarBar.SetTextPositionToPrecedeScalarBar()
    scalarBar.SetMaximumNumberOfColors(255)
    scalarBar.SetNumberOfLabels(5)
    scalarBar.SetAnnotationLeaderPadding(100)
    scalarBar.SetTextPositionToSucceedScalarBar()
    scalarBar.SetWidth(0.8)
    scalarBar.SetHeight(.05)

    # position of scalar bar defined in window coordinates
    coord = scalarBar.GetPositionCoordinate()
    coord.SetCoordinateSystemToNormalizedViewport()
    coord.SetValue(0.1,0.05)

    # adding the scalar bar actor
    renderer.AddActor2D(scalarBar)

    #rendering the image background with scalarbar
    window.Render()
    pil_image = window2pil_image(window)

    # resetting window size
    window.SetSize(640,480)

    # removing the scalar bar actor
    renderer.RemoveActor2D(scalarBar)

    # vtk mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()

    # vtk actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    camera = renderer.GetActiveCamera()
    
    # ambient occlusion pass
    basicPasses = vtk.vtkRenderStepsPass()
    ssao = vtk.vtkSSAOPass()
    ssao.SetRadius(50)
    ssao.SetBias(1)
    ssao.SetKernelSize(128)
    ssao.BlurOn()
    ssao.SetDelegatePass(basicPasses)
    renderer.SetPass(ssao)

    # lights
    lightKit = vtk.vtkLightKit()
    lightKit.MaintainLuminanceOff()
    lightKit.SetKeyLightIntensity(0.2)
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

    # axes text property
    axes_tp = vtk.vtkTextProperty()
    axes_tp.SetFontSize(50)
    axes_tp.SetColor((1,0,1))
    axes_tp.SetFontFamilyToArial()
    axes_tp.BoldOff()
    axes_tp.ItalicOff()
    axes_tp.ShadowOff()

    # axes
    axes = vtk.vtkCubeAxesActor2D()
    axes.SetInputConnection(contour.GetOutputPort())
    axes.SetInputData(dicom_data)
    axes.SetCamera(camera)
    axes.SetLabelFormat("%6.1f")
    axes.SetFlyModeToOuterEdges()
    axes.SetAxisTitleTextProperty(axes_tp)
    axes.SetAxisLabelTextProperty(axes_tp)
    axes.SetFontFactor(2.5)
    renderer.AddViewProp(axes)

    # isometric view rendered to pil image
    max_data_size = max(data_size_x,data_size_y,data_size_z)
    camera.SetPosition([sum(x) for x in zip(data_center, [max_data_size,max_data_size,-max_data_size])])
    camera.SetViewUp([0,0,-1])
    camera.SetParallelScale(max_data_size/1.25)
    window.Render()
    pil_image_iso = window2pil_image(window)

    # view in X+ direction rendered to pil image
    camera.SetPosition([sum(x) for x in zip(data_center, [data_size_x,0,0])])
    camera.SetParallelScale(max(data_size_y,data_size_z)/1.8)
    axes.SetYAxisVisibility(0)
    window.Render()
    pil_image_x = window2pil_image(window)

    # view in Y+ direction rendered to pil image
    camera.SetPosition([sum(x) for x in zip(data_center, [0,data_size_y,0])])
    camera.SetParallelScale(max(data_size_x,data_size_z)/1.8)
    axes.SetZAxisVisibility(0)
    axes.SetYAxisVisibility(1)
    window.Render()
    pil_image_y = window2pil_image(window)

    # view in Z+ direction rendered to pil image
    camera.SetPosition([sum(x) for x in zip(data_center, [0,0,-data_size_z])])
    camera.SetViewUp([-1,0,0])
    camera.SetParallelScale(max(data_size_y,data_size_x)/1.8)
    axes.SetYAxisVisibility(1)
    axes.SetZAxisVisibility(0)
    window.Render()
    pil_image_z = window2pil_image(window)

    # merging images
    pil_image.paste(pil_image_iso, (0, 0))
    pil_image.paste(pil_image_x, (window.GetSize()[0], 0))
    pil_image.paste(pil_image_y, (0, window.GetSize()[1]))
    pil_image.paste(pil_image_z, (window.GetSize()[0],window.GetSize()[1]))

    # image descriptions
    draw = ImageDraw.Draw(pil_image)
    draw.text((10,10), "ISO", "magenta",font_size=50)
    draw.text((window.GetSize()[0]+10,10), "X", "magenta",font_size=50)
    draw.text((10,window.GetSize()[1]+10), "Y", "magenta",font_size=50)
    draw.text((window.GetSize()[0]+10,window.GetSize()[1]+10), "Z", "magenta",font_size=50)
    
    # image saving
    pil_image.save(os.path.join(output_directory,"legend.png"))
