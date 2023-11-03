# importy

"""
A script for converting DICOM format data from CT scans into images. 
The use of the VTK (Visualization Toolkit) library makes it possible to
select any color scale and also to obtain additional data in the form of,
for example, a legend containing three-dimensional visualizations of the converted data.

"""

import vtk
import os
import numpy
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
optional.add_argument('-p',"--palette",type=str,help="color palette, bw - black-white (default), br - blue-red rainbow, cw - cool-warm",default="bw")

# args = parser.parse_args(["-i/home/maciejm/PHD/PUBLIKACJA_03/DCM/15/","-o/home/maciejm/PHD/PUBLIKACJA_03/IMG","-c100","-w100"])
# args = parser.parse_args(["-i/home/maciejm/PHD/PUBLIKACJA_03/DCM/15/","-oIMG","-c100","-w200"])
# args = parser.parse_args(["-i/home/maciejm/PHD/PUBLIKACJA_03/DCM/00000003/","-o/home/maciejm/PHD/PUBLIKACJA_03/IMG","-c100","-w500"])
args = parser.parse_args()

# creating the output directory
output_directory = os.path.abspath(args.output)
os.makedirs(output_directory, exist_ok=True)

# conversion from c and w to min value and max value
value_range_min = args.center - args.width/2
value_range_max = args.center + args.width/2

# dicom reader
dicom_reader = vtk.vtkDICOMImageReader()
dicom_reader.SetDirectoryName(args.input)
dicom_reader.Update()
dicom_data = dicom_reader.GetOutput()
dicom_dims = dicom_data.GetDimensions()
dicom_bounds = dicom_data.GetBounds()
dicom_spacing = list(dicom_data.GetSpacing())

# dicom data information
print("\ndicom data resolution   : ",dicom_dims)
print("dicom data spacing      : ",dicom_spacing)
print("dicom data X dimensions : ",dicom_bounds[0],"-",dicom_bounds[1])
print("dicom data Y dimensions : ",dicom_bounds[2],"-",dicom_bounds[3])
print("dicom data Z dimensions : ",dicom_bounds[4],"-",dicom_bounds[5])

# creating an image with 3d representation of dicom data
legend3d(dicom_data,output_directory)

# vtk color transfer function
transfer_function = vtk.vtkColorTransferFunction()
transfer_function.SetColorSpaceToHSV()
transfer_function.HSVWrapOff()
if args.palette == "cw":    # cold-warm
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 1, (value_range_max+value_range_min)/2, 1, 1, 1)
    transfer_function.AddRGBSegment((value_range_max+value_range_min)/2, 1, 1, 1, value_range_max, 1, 0, 0)
elif args.palette == "br":  # blue-red
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 1, value_range_max, 1, 0, 0)
else:   # black-white
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 0, value_range_max, 1, 1, 1)
transfer_function.Build()

# conversion from dicom data values to RGB color
coloring = vtk.vtkImageMapToColors()
coloring.SetLookupTable(transfer_function)
coloring.SetOutputFormatToRGB()
coloring.SetInputData(dicom_data)
coloring.Update()
color_data = coloring.GetOutput()

# conversion from vtk array to numpy array and numpy array reshaping
numpy_data = vtk_to_numpy(color_data.GetPointData().GetArray("DICOMImage"))
numpy_data_reshaped = numpy_data.reshape((dicom_dims[2],dicom_dims[1],dicom_dims[0],3))

# scales of nonuniform axes
scales = [round((x/min(dicom_spacing)),5) for x in dicom_spacing]

# saving images in X axis (Y-Z plane)
x_directory = os.path.join(output_directory,"X")
os.makedirs(x_directory, exist_ok=True)
for i in range(numpy_data_reshaped.shape[1]):
    pil_data = Image.fromarray(numpy.flip(numpy_data_reshaped[:,i,:,:],axis=1))
    if max(scales[1],scales[2]) > 1:
        pil_data = pil_data.resize((int(round(pil_data.size[0]*scales[1],0)), int(round(pil_data.size[1]*scales[2],0))), Image.Resampling.LANCZOS)
    pil_data.save(os.path.join(x_directory,"slice_%s.png"%(str(i).zfill(4))))
print("\nsaved %i images from X axis"%numpy_data_reshaped.shape[1])

# saving images in Y axis (X-Z plane)
y_directory = os.path.join(output_directory,"Y")
os.makedirs(y_directory, exist_ok=True)
for i in range(numpy_data_reshaped.shape[2]):
    pil_data = Image.fromarray(numpy.flip(numpy_data_reshaped[:,:,i,:],axis=1))
    if max(scales[0],scales[2]) > 1:
        pil_data = pil_data.resize((int(round(pil_data.size[0]*scales[0],0)), int(round(pil_data.size[1]*scales[2],0))), Image.Resampling.LANCZOS)
    pil_data.save(os.path.join(y_directory,"slice_%s.png"%(str(i).zfill(4))))
print("saved %i images from Y axis"%numpy_data_reshaped.shape[2])

# saving images in Z axis (X-Y plane)
z_directory = os.path.join(output_directory,"Z")
os.makedirs(z_directory, exist_ok=True)
for i in range(numpy_data_reshaped.shape[0]):
    pil_data = Image.fromarray(numpy.flip(numpy.flip(numpy_data_reshaped[i,:,:,:],axis=0),axis=1))
    if max(scales[0],scales[1]) > 1:
        pil_data = pil_data.resize((int(round(pil_data.size[0]*scales[0],0)), int(round(pil_data.size[1]*scales[1],0))), Image.Resampling.LANCZOS)
    pil_data.save(os.path.join(z_directory,"slice_%s.png"%(str(i).zfill(4))))
print("saved %i images from Z axis"%numpy_data_reshaped.shape[0])

#cv2_data_z = cv2.cvtColor(numpy.flip(numpy_data_reshaped[100,:,:,:],axis=0),cv2.COLOR_BGR2RGB)
#cv2.imwrite("obraz.png", cv2_data_z)