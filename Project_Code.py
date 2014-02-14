import numpy
import os
import pyfits
import sys
import pyrap.tables
import argparse

def handleparser():
  #Parses in variables from the command line and provides variable
  #usage advice via the --help and -h commands.
  global args
  parser = argparse.ArgumentParser()
  parser.add_argument("ms_name", help="File name of input measurement set.")
  parser.add_argument("output_fits", help="File name of output .fits file.")
  parser.add_argument("--tigger", help="Specify whether or not Tigger should be called", action="store_true")
  parser.add_argument("--clean", help="Remove all .fits from folder before running.", action="store_true")
  parser.add_argument("--npix", help="Specify image pixel width.", default=1024, type=int)
  parser.add_argument("--cellsize", help="Specify image cell size.", default=1, type=float)
  parser.add_argument("--mode", help="Specify mode.", default='channel')
  parser.add_argument("--weight", help="Specify weighting.", default='natural')
  parser.add_argument("--data", help="Specify data column.", default="CORRECTED_DATA", type=str)
  parser.add_argument("--binwidth", help="Specify the number of time slices per image.", default=100, type=int)
  args = parser.parse_args()  
  
def handletime():
  #Populates the list of times. Duplicate time values are removed. 
  global timelis
  mstab = pyrap.tables.table(args.ms_name)
  timetab = mstab.getcol("TIME")
  timelis = sorted(set(timetab))
  return
  
def handleimage():   
  #Handles image generation by generating .fits files for each time
  #slice/group and incorporates them into one multidimensional array. 
  #Also calls Tigger to display the images.
  imgcount = len(timelis)//args.binwidth
  dataarr = numpy.zeros((imgcount,1,args.npix,args.npix))
  if args.clean:
    os.system("rm *.fits")
  else:
    os.system("rm chunk*.fits")
  for i in range(imgcount):
    os.system("lwimager ms={0} fits=chunk{1}.fits npix={2} cellsize={3}arcsec prefervelocity=False data={4} select='TIME>={5} && TIME<={6}' mode={7} weight={8}".format(args.ms_name,i,args.npix,args.cellsize,args.data,timelis[args.binwidth*i],timelis[args.binwidth*(i+1)-1],args.mode,args.weight))    
    tmp = pyfits.open("chunk{0}.fits".format(i))
    tmpdata = tmp[0].data
    dataarr[i,0,...,...] = tmpdata
 
  tmp = pyfits.open("chunk0.fits")
  tmp[0].header['NAXIS4']=(imgcount)
  newfile = pyfits.PrimaryHDU(dataarr,tmp[0].header) 
  newfile.writeto("{}.fits".format(args.output_fits))
  if args.tigger:
    os.system("tigger {}.fits".format(args.output_fits))
  return

def main():
  #Calls other functions for the sake of neatness and conciseness.
  handleparser()
  handletime()
  handleimage()

main()