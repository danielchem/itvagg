import numpy as np
from matplotlib import pyplot as plt
import scipy.stats 
import argparse
parser = argparse.ArgumentParser(description='My app description')
parser.add_argument('-i', '--input', help='Path to input file')
args = parser.parse_args()

'''def ploting_distribution(infile):
	data = numpy.loadtxt(infile)
	a = data[:,1]     
	hist, bins = numpy.histogram(a, bins=10, density=True)
	# print bins
	bin_centers = (bins[1:]+bins[:-1])*0.5
	pyplot.plot(bin_centers, hist)
	pyplot.show()
	'''
def ploting_distribution(infile):
	data = np.loadtxt(infile)
	a = data[:,1]
	b = data[:,2]
	# test values for the bw_method option ('None' is the default value)
	# bw_values =  [None, 0.1, 0.01]
	bw_values =  [None]
	# generate a list of kde estimators for each bw
	kde1 = [scipy.stats.gaussian_kde(a,bw_method=bw) for bw in bw_values]
	kde2 = [scipy.stats.gaussian_kde(b,bw_method=bw) for bw in bw_values]
	# plot (normalized) histogram of the data
	plt.hist(a, 50, normed=1, facecolor='green', alpha=0.5);
	
	# plot density estimates
	t_range = np.linspace(-2,13,700)
	for i, bw in enumerate(bw_values):
			# plt.plot(t_range,kde1[i](t_range),lw=2, label='bw_com1-com2 = '+str(bw))
			# plt.plot(t_range,kde2[i](t_range),lw=2, label='bw_com1-collidingpoint = '+str(bw))
			plt.plot(t_range,kde1[i](t_range),lw=2, label='between two centers of mass')
			plt.plot(t_range,kde2[i](t_range),lw=2, label='between one center of mass and the colliding point')

	plt.xlabel('Distance')
	plt.ylabel('Probability')
	plt.title('Distribution of Aggregates interction distance\n '+str(infile))
	plt.xlim(0,10)
	plt.legend(loc='best')
	plt.show()

#---------------please use execute code in the terminal: "python plot-PDF.py -i agg133-result7.dat"

ploting_distribution(args.input)