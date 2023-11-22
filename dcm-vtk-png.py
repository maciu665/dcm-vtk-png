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
optional.add_argument('-p',"--palette",type=str,help="color palette, bw - black-white (default), cw - cool-warm, br - blue-red rainbow, bwnl - black-white bilinear, bwba - black-white with below and above colors, bwsel - black-white with range selection, bands - arbitrary bands",default="bw")

args = parser.parse_args(["-i/home/maciejm/PHD/PUBLIKACJA_03/DCM/15/","-o/home/maciejm/PHD/PUBLIKACJA_03/IMG","-c0","-w150","-pcw"])
# args = parser.parse_args(["-i/home/maciejm/PHD/PUBLIKACJA_03/DCM/17/","-oIMG","-c0","-w150"])
# args = parser.parse_args(["-i/home/maciejm/PHD/PUBLIKACJA_03/DCM/00000003/","-o/home/maciejm/PHD/PUBLIKACJA_03/IMG","-c100","-w500"])
# args = parser.parse_args()

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
# dicom data in the form of vtkImageData object
dicom_data = dicom_reader.GetOutput()
# dicom data resolution
dicom_dims = dicom_data.GetDimensions()
# dicom data size (X,Y and Z ranges)
dicom_bounds = dicom_data.GetBounds()
# dicom data spacing
dicom_spacing = list(dicom_data.GetSpacing())

# dicom data information
print("\ndicom data resolution   : ",dicom_dims)
print("dicom data spacing      : ",dicom_spacing)
print("dicom data X dimensions : ",dicom_bounds[0],"-",dicom_bounds[1])
print("dicom data Y dimensions : ",dicom_bounds[2],"-",dicom_bounds[3])
print("dicom data Z dimensions : ",dicom_bounds[4],"-",dicom_bounds[5])

# vtk color transfer function
transfer_function = vtk.vtkColorTransferFunction()
transfer_function.SetColorSpaceToHSV()
transfer_function.HSVWrapOff()

if args.palette == "cw":    # cold-warm, from blue to white to red
    transfer_function.SetColorSpaceToRGB()
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 1, (value_range_max+value_range_min)/2, 1, 1, 1)
    transfer_function.AddRGBSegment((value_range_max+value_range_min)/2, 1, 1, 1, value_range_max, 1, 0, 0)
elif args.palette == "br":  # blue-red, rainbow-like transition thanks to HSV color space
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 1, value_range_max, 1, 0, 0)
elif args.palette == "bwnl":  # black-white, bilinear
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 0, value_range_max-(args.width*0.85), 0.5, 0.5, 0.5)
    transfer_function.AddRGBSegment(value_range_max-(args.width*0.85), 0.5, 0.5, 0.5, value_range_max, 1, 1, 1)
elif args.palette == "bwba":  # black-white, with blue and red below and above range
    transfer_function.AddRGBSegment(value_range_min-1, 0, 0, 1, value_range_min, 0, 0, 1)
    transfer_function.AddRGBSegment(value_range_max, 1, 0, 0, value_range_max+1, 1, 0, 0)
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 0, value_range_max, 1, 1, 1)
elif args.palette == "bwsel":  # black-white, with middle 10% of values colored magenta
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 0, value_range_min+(args.width*0.45), 0.45, 0.45, 0.45)
    transfer_function.AddRGBSegment(value_range_min+(args.width*0.55)-0.001, 1, 0, 1, value_range_max, 0.55, 0.55, 0.55)
    transfer_function.AddRGBSegment(value_range_min+(args.width*0.45), 0.45, 0.45, 0.45, value_range_min+(args.width*0.45)+0.001, 1, 0, 1)
    transfer_function.AddRGBSegment(value_range_min+(args.width*0.55), 0.55, 0.55, 0.55, value_range_max, 1, 1, 1)
elif args.palette == "bands":  # arbitrary color bands
    transfer_function.AddRGBSegment(value_range_min, 0, 0.45, 0.45, value_range_min+(args.width*0.2), 0.0, 0.45, 0.45)
    transfer_function.AddRGBSegment(value_range_min+(args.width*0.2)+0.001, 1, 0, 1, value_range_min+(args.width*0.4), 1, 0, 1)
    transfer_function.AddRGBSegment(value_range_min+(args.width*0.4)+0.001, 1, 1, 0, value_range_min+(args.width*0.6), 1, 1, 0)
    transfer_function.AddRGBSegment(value_range_min+(args.width*0.6)+0.001, 0, 0.3, 0, value_range_min+(args.width*0.8), 0, 0.3, 0)
    transfer_function.AddRGBSegment(value_range_min+(args.width*0.8)+0.001, 1, 0.7, 1, value_range_max, 1, 0.7, 1)
else:   # bw, black-white
    transfer_function.AddRGBSegment(value_range_min, 0, 0, 0, value_range_max, 1, 1, 1)

transfer_function.Build()

# creating an image with 3d representation of dicom data
legend3d(dicom_data,output_directory,transfer_function,value_range_min,value_range_max)

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

###
pil_data = Image.fromarray(numpy.flip(numpy_data_reshaped[148,:,:,:],axis=0))
pil_data.save("slice_172-%s.png"%args.palette)

# sys.exit()
###

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
