# importy
import vtk
import sys
import os
import numpy
import cv2
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image
import argparse
from legend3d import *

parser = argparse.ArgumentParser()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-i','--input', required=True, type=str, help="input directory")
required.add_argument('-o','--output', required=True, type=str, help="output directory")
required.add_argument('-c','--center', required=True, type=int, help="window center")
required.add_argument('-w','--width', required=True, type=int, help="window width")
optional.add_argument('-p',"--palette",type=str,help="color palette")

args = parser.parse_args(["-i/home/maciejm/PHD/PUBLIKACJA_03/DCM/15/","-o/home/maciejm/PHD/PUBLIKACJA_03/IMG","-c100","-w200"])

print(args.center)

# conversion from c and w to min value and max value
value_range_min = args.center - args.width/2
value_range_max = args.center + args.width/2

#dicom reader
dicom_reader = vtk.vtkDICOMImageReader()
dicom_reader.SetDirectoryName(args.input)
dicom_reader.Update()
dicom_data = dicom_reader.GetOutput()
dicom_dims = dicom_data.GetDimensions()
#print(dicom_dims)
# print(dicom_data)

legend3d(dicom_data)


#lookup_table = vtk.vtkLookupTable()
#lookup_table.SetNumberOfTableValues(256)
#lookup_table.SetHueRange(0.667, 0.0)
# lookup_table.SetValueRange(0,256)
#lookup_table.Build()

transfer_function = vtk.vtkColorTransferFunction()
transfer_function.SetColorSpaceToHSV()
transfer_function.HSVWrapOff()
transfer_function.AddRGBSegment(value_range_min-1, 0, 0, 0, value_range_min, 0, 0, 0)
transfer_function.AddRGBSegment(value_range_max, 1, 1, 1, value_range_max+1, 1, 1, 1)
# transfer_function.AddRGBSegment(value_range_min, 0, 0, 1, value_range_max, 1, 0, 0)
transfer_function.Build()



coloring = vtk.vtkImageMapToColors()
# coloring.SetLookupTable(lookup_table)
coloring.SetLookupTable(transfer_function)
coloring.SetOutputFormatToRGB()
coloring.SetInputData(dicom_data)
coloring.Update()

color_data = coloring.GetOutput()
numpy_data = vtk_to_numpy(color_data.GetPointData().GetArray("DICOMImage"))
numpy_data_reshaped = numpy_data.reshape((dicom_dims[2],dicom_dims[0],dicom_dims[1],3))

#cv2.imwrite(os.path.join(ydir,"slice_%s.png"%(str(i).zfill(3))), dicom_np[:,:,i,:])

cv2_data_z = cv2.cvtColor(numpy.flip(numpy_data_reshaped[100,:,:,:],axis=0),cv2.COLOR_BGR2RGB)
cv2.imwrite("obraz.png", cv2_data_z)

pil_data = Image.fromarray(numpy.flip(numpy_data_reshaped[100,:,:,:],axis=0))
pil_data.save("pil.png")

sys.exit()
for i in range(512):
    #cv2.imwrite(os.path.join(xdir,"slice_%s.png"%(str(i).zfill(3))), dicom_np[:,i,:,:])
    #cv2.imwrite(os.path.join(ydir,"slice_%s.png"%(str(i).zfill(3))), dicom_np[:,:,i,:])
    if i < z_layers:
        cv2.imwrite(os.path.join(zdir,"slice_%s.png"%(str(z_layers-i).zfill(3))), numpy.flip(dicom_np[i,:,:,:],axis=0))