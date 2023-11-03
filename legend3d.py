import vtk


def legend3d(dicom_data):
    pcut = vtk.vtkContourFilter()
    pcut.SetInputData(dicom_data)
    dicom_data.GetPointData().SetActiveScalars("DICOMImage")
    pcut.SetValue(0,1000)
    pcut.Update()
    print(pcut.GetOutput())

    renderer = vtk.vtkRenderer()
    # window = vtkwin()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1000,1000)
    window.OffScreenRenderingOn()

    renderer.GradientBackgroundOff()
    renderer.SetBackground(0,0,0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(pcut.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)

    ########################################################################	LIGHT

    lightKit = vtk.vtkLightKit()
    lightKit.MaintainLuminanceOff()
    lightKit.SetKeyLightIntensity(1)
    lightKit.SetKeyLightWarmth(0.6)
    lightKit.SetFillLightWarmth(0.5)
    lightKit.SetHeadLightWarmth(0.5)
    lightKit.SetKeyToFillRatio(2)
    lightKit.SetKeyToHeadRatio(4)
    #lightKit.SetKeyToBackRatio(1000.1)
    lightKit.SetKeyLightAzimuth(30)
    lightKit.SetKeyLightElevation(45)
    lights = renderer.GetLights()
    lights.InitTraversal()
    lightKit.AddLightsToRenderer(renderer)

    window.Render()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(window)
    w2if.Update()

    image_writer = vtk.vtkPNGWriter()
    image_writer.SetFileName("legend.ppng")
    image_writer.SetInputData(w2if.GetOutput())
    image_writer.Write()

