from vtk.util import numpy_support
import vtk
import numpy as np
import SimpleITK as sitk

def savevtk(data, spacing, filename):
	# vtkImageData is the vtk image volume type
	imdata = vtk.vtkImageData()
	# this is where the conversion happens
	depthArray = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)

	# fill the vtk image data object
	imdata.SetDimensions(data.shape)
	imdata.SetSpacing(spacing)
	imdata.SetOrigin([0,0,0])
	imdata.GetPointData().SetScalars(depthArray)

	# f.ex. save it as mhd file
	writer = vtk.vtkMetaImageWriter()
	writer.SetFileName(filename)
	writer.SetInputData(imdata)
	writer.Write()


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from matplotlib import cm
	# Load the MHD file using SimpleITK
	image = sitk.ReadImage('./tj2_uniform_eps/tj2_200ueV.mhd')

	# Get the image data as a numpy array
	image_array = sitk.GetArrayFromImage(image)
	print(image_array.shape)

	line_data = -image_array[30, 40, :] + 1.13
	axis = np.linspace(0, 60, 61)

	plt.plot(axis, line_data)
	plt.show()

	#data_slice = image_array[:, 40, :]
	#plt.imshow(data_slice, cmap="jet")
	#plt.colorbar()
	#plt.show()
