import matplotlib.pyplot as plt
import numpy as np
import time 
import os
import sys 
from scipy.optimize import curve_fit
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import concurrent.futures
from pprint import pprint
############################################################################################
def Gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)
def Bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return Gauss(x,mu1,sigma1,A1)+Gauss(x,mu2,sigma2,A2)
############################################################################################
def PDF_GainAllCCD(filename):
   PDFGain = PdfPages(filename)
   Fig_Nums = plt.get_fignums()
   Figs = [plt.figure(n) for n in Fig_Nums]
   for fig in Figs:
      fig.savefig(PDFGain, format='pdf')
   PDFGain.close()
############################################################################################
def CalcGain(img_CCD16,NCOL,CCDNCOL,NAXIS2,CCD_Number,args):
        #####Zona de evaluacion 
    #Zona de overscan de interes NCOL-int(NCOL-ccdncol/2)):NCOL
    #limitx=int(CCDNCOL/2)
    #limity=int(NAXIS2/3)
    #DataCCD=(img_CCD16[0:limity,(limitx-45):(limitx+50)]).flatten()
    DataCCD = (img_CCD16).flatten()
    RangePlot = list(range(-200,500))
    fig = plt.figure()
    Y,X,_ = plt.hist(DataCCD,bins=RangePlot)            
    try:
        X = (X[1:]+X[:-1])/2 
        ExpectedCurveFit = (0, 0.8, 200, 300, 0.6, 20)
        params, cov = curve_fit(Bimodal, X, Y, ExpectedCurveFit)
        GainCCD = [str(params[3]-params[0])]
        #############################################
        if args==str("1") or args==str("3") or args==str("3"):
            if (params[0]>-20 and params[0]<20 and params[3]>150 and params[3]<800 and params[2]>0 and params[5]>0):
                return [GainCCD[0],params[0],params[3]]
            else:
                return [-1,params[0],params[3],params[2],params[5]]
        #############################################
        if args==str("3"):
            X_Line = range(-200, 450, 1)
            Y_Line1 = Gauss(X_Line, params[0], params[1], params[2])
            Y_Line2 = Gauss(X_Line,params[3], params[4], params[5])
            plt.plot(X_Line, Y_Line1, '--', color='red')
            plt.plot(X_Line, Y_Line2, '--', color='green')
            if (params[0]>-20 and params[0]<20 and params[3]>150 and params[3]<800 and params[2]>0 and params[5]>0):
                plt.text(0.6,0.7,f"CCD_Number:{CCD_Number}\nGain={GainCCD[0]}\nmu1: {params[0]} \nsigma1: {params[1]} \nA1: {params[2]} \nmu2: {params[3]} \nsigma2: {params[4]} \nA2: {params[5]}".format(GainCCD),size=7,transform=plt.gca().transAxes)
            else:
                if (params[0]<-20 or params[0]>20):
                    plt.text(0.6,0.7,f"CCD_Number:{CCD_Number}\nError: Primer pico desfasado\nmu1: {params[0]} \nmu2: {params[3]}".format(GainCCD),size=7,transform=plt.gca().transAxes)
                if (params[3]<150 or params[3]<800):
                    plt.text(0.6,0.7,f"CCD_Number:{CCD_Number}\nError: Segundo pico desfasado\nmu1: {params[0]}\nmu2: {params[3]}".format(GainCCD),size=7,transform=plt.gca().transAxes)
                if (params[2]<0 or params[5]<0):
                    plt.text(0.6,0.7,f"CCD_Number:{CCD_Number}\nError: Amplitud incorrecta\nA1: {params[2]}\nA2: {params[5]}".format(GainCCD),size=7,transform=plt.gca().transAxes)
        #############################################
    except:
        #plt.text(0.6,0.7,f"CCD_Number:{CCD_Number}\nError: Can't Fit Model",size=7,transform=plt.gca().transAxes)
        print("Error: Can't Fit Model")
        return [-2]
############################################################################################
def GetSingleCCDImage(hdul,LTA_channel,ColInit,NCOL,tamy,ccdncol,NSAMP): 
    MuxedImage=hdul[LTA_channel].data
    LastCol=ColInit+(NCOL-1)*NSAMP+1
    indexCol=list(range(ColInit,LastCol,NSAMP))
    DeMuxedImage=np.array(MuxedImage[:, indexCol],dtype='f')
    for p in range(tamy):
        Offset=np.mean(DeMuxedImage[p,(NCOL-int(NCOL-ccdncol/2)):NCOL])
        DeMuxedImage[p,:]=DeMuxedImage[p,:]-Offset
    return DeMuxedImage 
############################################################################################
if __name__ == '__main__': 
    StartTime = time.time() 
    args = sys.argv[1:]
    inputFile = str(args[0]) #inputFile = str(sys.argv[1])
    baseName=os.path.splitext(inputFile)[0]
    hdulist = fits.open(inputFile)
    NAXIS1=int(hdulist[4].header['NAXIS1']) #Size X
    NAXIS2=int(hdulist[4].header['NAXIS2']) #Size Y
    NSAMP=int(hdulist[4].header['NSAMP']) 
    NCOL=int(hdulist[4].header['NCOL'])
    CCDNCOL=int(hdulist[4].header['CCDNCOL'])
    scidata = hdulist[4].data
    tamxpimg=int(NAXIS1/NSAMP)
    LTA_channel=4
    CCDinMCM=16 
    CSV = []
    List_CCDNumber_Gains = []
    Map=[1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]
    #############################################
    if args[1]==str("1") or args[1]==str("2") or args[1]==str("3"):
        for N in range(int(NSAMP/CCDinMCM)):     
            Primaryhdu_MCM = fits.PrimaryHDU(header=hdulist[0].header)
            img_CCD16=fits.HDUList([Primaryhdu_MCM]) 
            for CCD in Map:                                
                img_parcial=GetSingleCCDImage(hdulist,LTA_channel,CCD-1+CCDinMCM*N,NCOL,NAXIS2,CCDNCOL,NSAMP)
                img_name="CCD "+str(CCD)+" in MCM "+str(N+1)
                image_hdu=fits.ImageHDU(data=img_parcial, header=hdulist[LTA_channel].header, name=img_name)
                image_hdu.verify('silentfix')
                ############################################# 
                if args[1]==str("2") or args[1]==str("3"):
                    CCD_Number=N*16+CCD+1
                    GainCCD=CalcGain(img_parcial,NCOL,CCDNCOL,NAXIS2,CCD_Number,args[1]) 
                    print("CCD ",CCD_Number," ",GainCCD)
                    List_CCDNumber_Gains.extend([[CCD_Number,GainCCD[0]]])
                    CSV.extend(List_CCDNumber_Gains)
                    List_CCDNumber_Gains.clear()
                #############################################
                img_CCD16.append(image_hdu)
            Directory_Demux="Demuxed_"+baseName+"/"
            if not os.path.exists(Directory_Demux):
                os.makedirs(Directory_Demux)
            SaveName=str(Directory_Demux+"PROC_MCM"+str(N+1)+"_Demuxed_"+baseName+".fits") 
            img_CCD16.writeto(SaveName,overwrite=True)
            img_CCD16.clear()
            print((N+1)/(NSAMP/16)*100,"% Done...")
        hdulist.close()
    #############################################
    if args[1]==str("3"):
        filename = Directory_Demux+"Histogram_Full"+baseName+"params.pdf" 
        PDF_GainAllCCD(filename)
        np.savetxt("GAIN_CDD's.csv", CSV, delimiter =",",fmt ='% s')
    #############################################
    EndTime = time.time()
    Elapsed_time = EndTime - StartTime
    print('Execution time:', Elapsed_time, 'seconds')