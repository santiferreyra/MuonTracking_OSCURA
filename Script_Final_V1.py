import matplotlib.pyplot as plt
import concurrent.futures
import mysql.connector
import pandas as pd
import numpy as np
import time 
import sys
import os 
import json
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from astropy.io import fits
from pprint import pprint

####o run: python3 256chTool.py Multiplexed_LTA_Output_Image.fz

###If you want to make a save data in MySQL:
def DatabaseMySQL():
    MyDB = mysql.connector.connect(
        host="YourLocalHost", #
        port="YourPort",
        user="YourUser",
        passwd="YourPassword",
        auth_plugin="mysql_native_password")
    MyCursor = MyDB.cursor()
    MyCursor.execute("CREATE DATABASE DatabaseName")
    ###Database connection created and data table creation
    MyDB = mysql.connector.connect(
        host = "YourLocalHost",
        port = "YourPort",
        user = "YourUser",
        passwd = "YourPassword",
        database = "DatabaseName",
        auth_plugin = "mysql_native_password")
    print(MyDB)
    MyCursor = MyDB.cursor()
    MyCursor.execute("CREATE TABLE TableName(id INT AUTO_INCREMENT PRIMARY KEY, gain VARCHAR(255), date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")

###To save data in the table created in the database
def SaveData(values):
    ###Saved in rows of the ccd number and in columns of the corresponding image
    MyDB = mysql.connector.connect(
        host = "YourLocalHost",
        port = "YourPassword",
        user = "YourUser",
        passwd = "YourPassword",
        database = "DatabaseName",
        auth_plugin = "mysql_native_password")
    MyCursor = MyDB.cursor()
    sql = "INSERT INTO TableName (gain) VALUES (%s)"
    for v in values:            
        MyCursor.execute(sql, (v[0],))
    MyDB.commit()

###Functions in charge of calculating the multinomial distribution by searching for the peak at zero and at one.
###The parameters are defined: mu (mean), sigma(standard deviation), A(amplitude)
def Gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)
def Bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return Gauss(x,mu1,sigma1,A1)+Gauss(x,mu2,sigma2,A2)

###Extra function in charge of printing a PDF with the CCD peaks for user viewing.
def Save_Multi_Image(filename):
   pp = PdfPages(filename)
   fig_nums = plt.get_fignums()
   figs = [plt.figure(n) for n in fig_nums]
   for fig in figs:
      fig.savefig(pp, format='pdf')
   pp.close()

###Function in charge of calculating the gain of the CCD by means of the distribution of the peaks
def GainSingleCCD(img_CCD16):
    Data = (img_CCD16).flatten()
    Data = np.delete(Data, np.where(Data == 0)) #Eliminate from zero due to statistical problems
    PeakRange = list(range(-250,400)) ###Range in which the first and second zeros are expected to be seen, vary if looking for other peaks.
    #fig = plt.figure()
    Y,X,_ = plt.hist(Data, bins = PeakRange)            
    try:
        X = (X[1:]+X[:-1])/2 
        ###The first and second expected peaks are performed
        ###The search for other peaks can also be performed by changing the expected value.
        Expected = (0, .6, 800, 300, .8, 250) #mu1,sigma1,A1,mu2,sigma2,A2
        params, cov = curve_fit(Bimodal, X, Y, Expected)
        ###String conversion for saving in database, also possible as integer/float.
        Gain = [str(params[3]-params[0])] 
        return Gain
    except:
        ###In case of not being able to graph, it may be a problem with the image itself or with an incorrect value in the expected parameters.
        print("Can't Fit Model") 
        return[-1] ###Return value for an error

###Function in charge of demultiplexing
def GetSingleCCDImage(hdul,LTA_channel,ColInit,NCOL,NAXIS2,NSAMP,CCDNCOL): 
    MuxedImage=hdul[LTA_channel].data
    LastCol=ColInit+(NCOL-1)*NSAMP 
    indexCol=list(range(ColInit,LastCol,NSAMP)) 
    DeMuxedImage=MuxedImage[:, indexCol]
    ###The offset is also analyzed using the mean criterion.
    #OffsetRange=list( range( int(CCDNCOL/2),NCOL)) 
    #OffsetImage=(MuxedImage[:,OffsetRange]).flatten()
    #OffsetImage = np.delete(OffsetImage, np.where(OffsetImage == 0))
    #PeakRange = list(range(-200,200))
    #Y,X,_ = plt.hist(OffsetImage, bins = PeakRange)
    #X = (X[1:]+X[:-1])/2 
    #Expected = (0, .6, 800) #mu1,sigma1,A1
    #params, cov = curve_fit(Gauss, X, Y, Expected)
    #Offset = params[0] 
    for p in range(NAXIS2):
        Offset=np.mean(DeMuxedImage[p,int(CCDNCOL/2):NCOL])
        DeMuxedImage[p,:]=DeMuxedImage[p,:]-Offset
    return DeMuxedImage 

###Start of the code:
if __name__=='__main__':
    Start_Time = time.time() ###Variable for time measurement
    inputFile = str(sys.argv[1])
    baseName=os.path.splitext(inputFile)[0]
    Primaryhdu_MCM = fits.PrimaryHDU() # Create primary HDU without data
    HDU_list_MCM = fits.HDUList([Primaryhdu_MCM]) #Create HDU list
    gain_list=[] 
    hdulist = fits.open(inputFile)
    HDU_list_MCM[0].header=hdulist[0].header 
    NAXIS1=int(hdulist[4].header['NAXIS1']) #Size X
    NAXIS2=int(hdulist[4].header['NAXIS2']) #Size Y
    NSAMP=int(hdulist[4].header['NSAMP']) #CCD's
    ANSAMP=int(hdulist[4].header['ANSAMP']) 
    NCOL=int(hdulist[4].header['NCOL']) 
    CCDNCOL=int(hdulist[4].header['CCDNCOL'])
    Scidata = hdulist[4].data
    LTA_channel=4
    CCDinMCM=16 
    PartialImageData=[] ###List used to save the images and obtain the profits corresponding to the CCDs
    CCD_Order=[12,8,4,0,13,9,5,1,14,10,6,2,15,11,7,3]
    for N in range(int(NSAMP/CCDinMCM)): ###Generates N images in arrays of 16   
        for CCD in range(CCDinMCM): ###It traverses in the number of ccd per MCM 
            PartialImage=GetSingleCCDImage(hdulist,LTA_channel,CCD_Order[CCD]+CCDinMCM*N,NCOL,NAXIS2,NSAMP,CCDNCOL)
            PartialImageData.append(PartialImage)
            PartialImage_HDU=fits.ImageHDU(PartialImage)
            PartialImage_HDU.header=hdulist[LTA_channel].header 
            PartialImage_HDU.header.set('NSAMP',ANSAMP)
            HDU_list_MCM.append(PartialImage_HDU)
        ###Process of saving the new partial images
        Directory_Demux="Demuxed_"+baseName+"/"
        if not os.path.exists(Directory_Demux):
            os.makedirs(Directory_Demux)
        SaveName=str(Directory_Demux+"MCM"+str(N+1)+"_Demuxed_"+baseName+"_PROC.fits") 
        HDU_list_MCM.writeto(SaveName,overwrite=True)
        HDU_list_MCM.clear()
    print("All MCM Done!")
    ###The process of obtaining the gain for NSAMP CCD is executed in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        Result = executor.map(GainSingleCCD,PartialImageData)
    AllGain=list(Result) ###List in which all gains corresponding to the processed image are stored
    ###There are two ways to store earnings data:
    #If you want to save the image in a .json format:
    #CCD_AllGain = [[AllGain[_], datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")] for _ in range(len(AllGain))]
    #with open('CCD_AllGain', 'a+') as f: json.dump(CCD_AllGain, f)  
    #If you want to save the image in a database, as an example of MySQL
    #DatabaseMySQL()
    #SaveData(AllGain)        
    PartialImageData.clear()    
    hdulist.close()
    #Filename = directorio_demux=directorio_guardado+"Histogram112.pdf"
    #Save_Multi_Image(Filename)
    HDU_list_MCM.close()
    End_Time = time.time()
    Elapsed_time = End_Time - Start_Time
    print('Execution time:', Elapsed_time, 'seconds')
