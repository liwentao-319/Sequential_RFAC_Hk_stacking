
import re
import numpy as np
from obspy import read
from obspy.taup import TauPyModel
from obspy.core import Stream
from acc.core import spectral_whitening,remove_response
from acc.processing import _autocorrelation
from acc.stack import linear_stack,pws_stack
from glob import glob 
from math import floor
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import math
from scipy import signal
from scipy.fft import fft, ifft

list_colornames = ['orange','gold','cyan','skyblue','blueviolet','violet','pink']

def dereverberation_filter(r0, t0, freqs):

    """
    resonance removal filter
    reference: Yu et al., 2014; Zhang and Olugboji 2021;
    written by Wentao Li
    Date: 2024/06/04
    """
    complex_numbers = -2*1j*np.pi*freqs*t0
    return 1+r0*np.exp(complex_numbers)

def remove_resonance(rdata,dt,r0,t0):
    """
    remove resonance in time series, rdata with sampling interval to dt
    the frequency of resonance is 1/t0, and the attenuation of each peak is r0
    assuming the length of rdata is even
    written by Wentao Li
    Date: 2024/06/04

    """
    #number points in frequency domain
    n = len(rdata)
    if (n%2)!=0:
        rdata = np.insert(rdata,-1,0)
        n += 1
    ##transform from time-domain to frequency domain and get its positive frequency part
    spec_rdata = fft(rdata)
    ##frequency interval
    df = 1/n/dt
    ##calculate the corresponding frequencies
    freqs = df*np.concatenate([np.arange(n//2),-1*np.flip(np.arange(1,n//2+1))])
    ##design dereverberation filter
    filter = dereverberation_filter(r0,t0,freqs)
    spec_rdata_ = spec_rdata*filter

    return np.real(ifft(spec_rdata_))

def measure_t0_r0(rdata,dt):
    npts = len(rdata)
    autocorr = np.correlate(rdata,rdata,mode='full')[npts:]
    index_t0 = np.argmin(autocorr)
    t0 = index_t0*dt
    r0 = -1*autocorr[index_t0]/autocorr[0]
    return t0, r0





class Hk_searching_RF_2layer_Yu(object):
    """
    Discription   : 2layer H-k stacking of Yu et al 2015
    Author        : Wentao Li
    Time          : Jul 6th, 2023
    Version       : 1.0

    """
    def __init__(self, stationname, data_st=[], dataparam={}, priorparam={}, weights={}, phasestackparams=[0.04,0.08,0.002,0.004,2]):
        """
        dataset [list] : data used to Hk stacking, a list of obspy Streams
        dataparam [dic] : Directory contain data information, such as sampling interval (delta), time length before zeto time(pretimes), length used to Hk stacking (length)
        priorparam [dic] : Directory contain prior parameters for Hk stacking
        weigths [list] : list of list to specify weights of seismic phases in each layer
        phasestackparams [list]: parameter set  of phase-weighting stack before Hk stacking
                                  [ray_param_min, ray_param_max, delta, width, phasestack_power]
        """
        ## 
        self.stationname = stationname
        ## each seismic phase will match a index in the dataset
        self.data_st = data_st
        self.Dataset = []
        self.ray_params = []

        self.bin_nums = []
        ##check the Stack_dataset has the same ray parameter distribution

        self.Ndataset = len(self.data_st)
        if self.Ndataset==0:
            print("Configure Error: No stack data input, quit!")
            raise
        ##resolve dataparam
        self.delta = dataparam['delta']
        self.pretimes = dataparam['pretimes']
        self.lengths = dataparam['lengths']
        self.norm_types = dataparam['norm_type'] ##-1 or 1
        self.plotlengths = dataparam['plotlengths']

        if len(self.pretimes)!= self.Ndataset or len(self.lengths)!= self.Ndataset or len(self.norm_types)!= self.Ndataset or len(self.plotlengths)!= self.Ndataset:
            print("Configure Error: length of pretimes, lengths or norm_types is not euqal to Ndataset")
            raise
        ##preprocessing, including trim data, phase stacking, normalization and phase weighting stacking
        self._trim_resample_normalize_trace()


        if phasestackparams!=None:
            self._phase_stacking(phasestackparams)
        else:
            nevent = len(self.data_st[0])
            for i in range(self.Ndataset):
                if len(self.data_st[i])!=nevent:
                    print("Configure Error: not equal number of events in different datasets, quit!")
                    raise
            Nst = len(self.data_st)
            for j in range(Nst):
                self.Dataset.append([])
            for i in range(nevent):
                ray_param = self.data_st[0][i].stats.sac['user0']
                for j in range(self.Ndataset):
                    if not math.isclose(self.data_st[j][i].stats.sac['user0'],ray_param,rel_tol=1e-5):
                        print(f"Configure Error: the ray parameters doesn't match between dataset 0 and {j} for event {i}")
                        raise

                    self.Dataset[j].append(self.data_st[j][i].data)
                self.ray_params.append(ray_param)
            for j in range(Nst):
                self.Dataset[j] = np.stack(self.Dataset[j],axis=1)        
        ##preprocessing for Dereverberation
        self.ray_params_DeRe = []
        self.t0ss_DeRe = []
        self.tpbsss_DeRe = []
        self.Dataset_DeRe = []
        for j in range(Nst):
            dataset_arr = self.Dataset[j]
            
            npts, ntrace = dataset_arr.shape
            t0s = []
            tpbss = []
            r0s = []
            for i in range(ntrace):
                data_ = dataset_arr[:,i]
                index_tpbs = np.argmax(data_)
                tpbs = index_tpbs*self.delta
                ##measure t0, r0 and tpbs
                t0,r0 = measure_t0_r0(data_,self.delta)
                t0s.append(t0)
                r0s.append(r0)
                tpbss.append(tpbs)

            dataset_arr_DeRe = []
            t0s_DeRe = []
            tpbss_DeRe = []
            medium_t0 = np.median(np.array(t0s))
            print("mean_t0",medium_t0)
            for i in range(ntrace):
                t0 = t0s[i]                
                if abs(t0-medium_t0)>0.5:
                    continue
                ##remove resonance
                data_ = dataset_arr[:,i]
                r0 = r0s[i]
                dataset_arr_DeRe.append(remove_resonance(data_, self.delta, r0, t0))
                self.ray_params_DeRe.append(self.ray_params[i])
                t0s_DeRe.append(t0)
                tpbss_DeRe.append(tpbss[i])
            self.Dataset_DeRe.append(np.stack(dataset_arr_DeRe,axis=1))
            self.t0ss_DeRe.append(t0s_DeRe)
            self.tpbsss_DeRe.append(tpbss_DeRe)
        ##resolve priorparam
        self.Nlayer = priorparam['Nlayer']
        self.Vps = priorparam['Vps']
        self.Hs_range = priorparam['Hs']
        self.Ks_range = priorparam['Ks']
     
        ##check the parameter length
        if len(self.Vps)!=self.Nlayer or len(self.Hs_range)!=self.Nlayer or len(self.Ks_range)!=self.Nlayer :
            print("Configure Error: length of Vps, Hs or Ks is not euqal to Nlayer")
            raise

        ##resolve weights of seismic phases
        self.weights = weights

        ##check weights
        if len(self.weights)!=self.Nlayer:
            print("Configure Error: the number of weights doesn't match the number of layers")
            raise
        for i in range(self.Nlayer):
            if len(self.weights[i])!=3:
                print(f"Configure Error: no enough weigths for the {i}th layer")
                raise

        ## initialize other parameters
        self.optimal_Hs = []
        self.optimal_Ks = []
        self.std_Hs = []
        self.std_Ks = []
        self.Hk_amps = []
        self.bootstrap_randoms_Hs = []
        self.bootstrap_randoms_Ks = []

    def _trim_resample_normalize_trace(self):
        ndataset = len(self.data_st)
        for i in range(ndataset):
            ntrace = len(self.data_st[i])
            pretime = self.pretimes[i]
            length = self.lengths[i]
            norm_type = self.norm_types[i]
            for j in range(ntrace):
                starttime = self.data_st[i][j].stats.starttime
                if self.delta!=self.data_st[i][j].stats.delta:
                    self.data_st[i][j].resample(1/self.delta)
                self.data_st[i][j].trim(starttime+pretime,starttime+pretime+length)
                norm = max(norm_type*self.data_st[i][j].data)
                # norm = max([0.001, norm])
                self.data_st[i][j].data = self.data_st[i][j].data/norm
                # print(len(self.data_st[i][j].data))
                
    def _phase_stacking(self, phasestackparams):
        ray_para_min, ray_para_max, ray_para_delta, ray_param_width, power = phasestackparams
        ray_params  = np.arange(ray_para_min,ray_para_max+ray_para_delta,ray_para_delta)
        data_streams = self.data_st
        Nst = len(data_streams)
        for j in range(Nst):
            self.Dataset.append([])
        for ray_param in ray_params:
            ray_param_bin_min = ray_param-ray_param_width/2.0
            ray_param_bin_max = ray_param+ray_param_width/2.0
            temp_stack_list = []
            for j in range(Nst):
                Stream_stack=Stream()
                Nrecord = len(data_streams[j])
                Nbin = 0
                for i in range(Nrecord):
                    ray_param_cur = data_streams[j][i].stats.sac['user0']
                    if ray_param_cur>ray_param_bin_min and ray_param_cur<=ray_param_bin_max:
                        Stream_stack.append(data_streams[j][i])
                        Nbin +=1
                if len(Stream_stack)!=0:
                    tr_stack = pws_stack(Stream_stack,power=power)
                    tr_stack.stats.sac['User0']=ray_param
                    temp_stack_list.append(tr_stack.data)
            if len(temp_stack_list)==Nst:
                for j in range(Nst):
                    self.Dataset[j].append(temp_stack_list[j])
                self.ray_params.append(ray_param)
                self.bin_nums.append(Nbin)

        for j in range(Nst):
            self.Dataset[j] = np.stack(self.Dataset[j],axis=1)

    def do_searching(self):
        self.optimal_Hs, self.optimal_Ks, self.Hk_amps = self.searching(self.ray_params_DeRe, self.Dataset_DeRe, self.t0ss_DeRe, self.tpbsss_DeRe)

    def searching(self, ray_params, Dataset, t0ss, tpbsss):
        """
        kernal of sequetial Hk stacking
        Writen by Wentao Li
        Datetime: 2024/06/13
        """
        ray_params = np.array(ray_params)
        sampling = 1/self.delta
        Nlayer = self.Nlayer
        ##intialize records for optimal thickness and Vp/Vs ratio of each layer
        Hs_op = []
        Ks_op = []
        Amps = []
        t0s = np.array(t0ss[0])
        tpbss = np.array(tpbsss[0])
        stack_data = Dataset[0]
        #prior Vp
        Vpc = self.Vps[0]
        ##searching domain
        Hmin,Hmax,nH = self.Hs_range[0]
        Kmin,Kmax,nK = self.Ks_range[0]
        hh=np.linspace(Hmin,Hmax,nH)
        kk=np.linspace(Kmin,Kmax,nK)
        H,K=np.meshgrid(hh,kk)
        H=H[:,:,None]
        K=K[:,:,None]
        ##initialize the stacking
        Amp = np.zeros([nK,nH])
        ##load seismic phase
        w1,w2,w3 = self.weights[0]
        
        ttt1 = H*((K**2/Vpc**2-ray_params**2)**(1/2) - (1/Vpc**2-ray_params**2)**(1/2)) + tpbss[None,None,:]

        ttt2 = H*((K**2/Vpc**2-ray_params**2)**(1/2) + (1/Vpc**2-ray_params**2)**(1/2)) + t0s[None,None,:] - tpbss[None,None,:]

        ttt3 = H*(2*(K**2/Vpc**2-ray_params**2)**(1/2)) + t0s[None,None,:] 

        
        Ampc = w1*self.phasetime2amp(stack_data, ttt1, sampling) + w2*self.phasetime2amp(stack_data, ttt2, sampling) + \
                                                    w3*self.phasetime2amp(stack_data, ttt3, sampling)

        ##find the maximum amp point
        index_max=Ampc.argmax()
        ik=floor(index_max/nH)
        ih=index_max%nH
        hhc = hh[ih]
        kkc = kk[ik]
        Hs_op.append(hhc)
        Ks_op.append(kkc)
        Amps.append(Ampc)

        Vps = self.Vps[1]
        ##searching domain
        Hmin,Hmax,nH = self.Hs_range[1]
        Kmin,Kmax,nK = self.Ks_range[1]
        hh=np.linspace(Hmin,Hmax,nH)
        kk=np.linspace(Kmin,Kmax,nK)
        H,K=np.meshgrid(hh,kk)
        H=H[:,:,None]
        K=K[:,:,None]
        ##initialize the stacking
        Amp = np.zeros([nK,nH])
        ##load seismic phase
        w1,w2,w3 = self.weights[1]

        ttt1 = H*((K**2/Vps**2-ray_params**2)**(1/2) - (1/Vps**2-ray_params**2)**(1/2))

        ttt2_c = hhc*((kkc**2/Vpc**2-ray_params**2)**(1/2) + (1/Vpc**2-ray_params**2)**(1/2))

        ttt2 = H*((K**2/Vps**2-ray_params**2)**(1/2) + (1/Vps**2-ray_params**2)**(1/2)) + ttt2_c[None,None,:]

        ttt3_c = 2*hhc*(kkc**2/Vpc**2-ray_params**2)**(1/2)

        ttt3 = 2*H*(K**2/Vps**2-ray_params**2)**(1/2) + ttt3_c[None,None,:]

        
        Amp = w1*self.phasetime2amp(stack_data, ttt1, sampling) + w2*self.phasetime2amp(stack_data, ttt2, sampling) + \
                                                    w3*self.phasetime2amp(stack_data, ttt3, sampling)
        ##find the maximum amp point
        index_max=Amp.argmax()
        ik=floor(index_max/nH)
        ih=index_max%nH
        hhs = hh[ih]
        kks = kk[ik]
        Hs_op.append(hhs)
        Ks_op.append(kks)
        Amps.append(Amp)

        return Hs_op, Ks_op, Amps


    def hk_bootstrap_estimate(self,Nb,nthread=None):

        ##sharing variables
        ntrace = len(self.ray_params_DeRe)

        if nthread != None:
            rgn = np.random.default_rng(12345)
            random_indexes = []
            for i in range(Nb):
                random_indexes.append(rgn.choice(ntrace,ntrace,replace=True))

            ##allocate jobs
            args_pool = []
            nwork = int(Nb/nthread)
            for i in range(nthread-1):
                args_pool.append(random_indexes[i*nwork:(i+1)*nwork])
            i=i+1
            args_pool.append(random_indexes[i*nwork:])
            ##apply parallel processing
            pool = Pool(processes=nthread)
            results = []
            for i in range(nthread):
                results.append(pool.apply_async(self.Do_bootstrap,args=(args_pool[i],)))

            pool.close()
            pool.join()
            ##collect results
            Hs_randoms = []
            Ks_randoms = []
            for j in range(self.Nlayer):
                Hs_randoms.append([])
                Ks_randoms.append([])  
            
            for result in results:
                Hs_randoms_batch, Ks_randoms_batch = result.get()
                for i in range(len(Hs_randoms_batch[0])):
                    for j in range(self.Nlayer):
                        Hs_randoms[j].append(Hs_randoms_batch[j][i])
                        Ks_randoms[j].append(Ks_randoms_batch[j][i])

            for j in range(self.Nlayer):
                Hs_randoms[j] = np.array(Hs_randoms[j])
                Ks_randoms[j] = np.array(Ks_randoms[j])

            self.bootstrap_randoms_Hs = Hs_randoms
            self.bootstrap_randoms_Ks = Ks_randoms
            self.std_Hs = [np.std(Hs_randoms[j]) for j in range(self.Nlayer)]
            self.std_Ks = [np.std(Ks_randoms[j]) for j in range(self.Nlayer)]

        else:

            rgn = np.random.default_rng(12345)
            random_indexes = []
            for i in range(Nb):
                random_indexes.append(rgn.choice(ntrace,ntrace,replace=True)) 
            Hs_randoms, Ks_randoms = self.Do_bootstrap(random_indexes)
            
            self.bootstrap_randoms_Hs = Hs_randoms
            self.bootstrap_randoms_Ks = Ks_randoms
            self.std_Hs = [np.std(Hs_randoms[i] for i in range(self.Nlayer))]
            self.std_Ks = [np.std(Ks_randoms[i] for i in range(self.Nlayer))]

    def Do_bootstrap(self,indexes):
        Hs_ops = []
        Ks_ops = []
        for j in range(self.Nlayer):
            Hs_ops.append([])
            Ks_ops.append([])

        for index in indexes:
            random_Dataset = []
            random_t0ss = []
            random_tpbsss = []
            for i in range(self.Ndataset):
                random_Dataset.append(self.Dataset_DeRe[i][:,index])
                random_t0ss.append([self.t0ss_DeRe[i][iii] for iii in index])
                random_tpbsss.append([self.tpbsss_DeRe[i][iii] for iii in index])               
            random_ray_params = [self.ray_params_DeRe[iii] for iii in index]

            Hs_op, Ks_op, _= self.searching(random_ray_params,random_Dataset,random_t0ss,random_tpbsss)
            for j in range(self.Nlayer):
                Hs_ops[j].append(Hs_op[j])
                Ks_ops[j].append(Ks_op[j])

        return Hs_ops, Ks_ops

    def save_result_to_files(self,savedir):
        stationname = self.stationname
        savefile = f"{savedir}/{stationname}_Hk_result.dat"
        with open(savefile, 'w') as fd:
            fd.write("vp   H    K    Hstd    Kstd \n")
            #for sedimentary layer
            for j in range(self.Nlayer):
                vp = self.Vps[j]
                H_op = self.optimal_Hs[j]
                K_op = self.optimal_Ks[j]
                H_std = self.std_Hs[j]
                K_std = self.std_Ks[j]
                fd.write(f"{vp:3.2f} {H_op:<4.2f} {K_op:4.3f} {H_std:4.2f} {K_std:4.3f} \n")

        # if len(self.bootstrap_randoms_Hs) != 0 :
        #     Hs_randoms = np.stack(self.bootstrap_randoms_Hs,axis=1)
        #     np.save(f"{savedir}/{stationname}_bootstrap_Hs.npy", Hs_randoms)
        # if len(self.bootstrap_randoms_Ks) != 0 :
        #     Ks_randoms = np.stack(self.bootstrap_randoms_Ks,axis=1)
        #     np.save(f"{savedir}/{stationname}_bootstrap_Ks.npy", Ks_randoms)
        
        # if len(self.Hk_amps) != 0:
        #     Hk_amps = np.stack(self.Hk_amps,axis=2)
        #     np.save(f"{savedir}/{stationname}_Hk_amps.npy", Hk_amps)
            
        # if len(self.Dataset) != 0:
        #     for i in range(self.Ndataset):
        #         np.savez(f"{savedir}/{stationname}_dataset{i+1}.npz", self.Dataset[i])

    def save_result_to_figs(self,savedir,pic_format='png'):
        plt.rcParams.update({"font.size":7})
        stationname = self.stationname
        sampling = 1/self.delta
        if len(self.Hk_amps)!= 0 :
            fig = plt.figure(1,figsize=(6*2,5),tight_layout=True)
            ax1=fig.add_subplot(121)
            amp = self.Hk_amps[0]
            Hmin,Hmax,nH = self.Hs_range[0]
            Kmin,Kmax,nK = self.Ks_range[0]
            hh=np.linspace(Hmin,Hmax,nH)
            kk=np.linspace(Kmin,Kmax,nK)
            Hs_op = self.optimal_Hs[0]
            Ks_op = self.optimal_Ks[0]
            if len(self.std_Hs)==0:
                Hs_std = 0.0
                Ks_std = 0.0
            else:
                Hs_std = self.std_Hs[0]
                Ks_std = self.std_Ks[0]

            ax1.contourf(hh,kk,amp,levels=25)
            ax1.plot([Hs_op-3*Hs_std,Hs_op+3*Hs_std],[Ks_op,Ks_op],lw=1,c='black')
            ax1.plot([Hs_op,Hs_op],[Ks_op-3*Ks_std,Ks_op+3*Ks_std],lw=1,c='black')
            ax1.set_ylim(Kmin,Kmax)
            ax1.set_xlim(Hmin,Hmax)
            ax1.set_title(f'crust layer \n' + r'Thickness:{0:4.2f}$\pm${2:4.2f} Vp/Vs:{1:4.3f}$\pm${3:4.3f}'.\
                            format(Hs_op,Ks_op,3*Hs_std,3*Ks_std))
            ax1.set_ylabel('Vp/Vs')
            ax1.set_xlabel('Thickness [km]')
            ax1.minorticks_on()

            ax2=fig.add_subplot(122)
            amp = self.Hk_amps[1]
            Hmin,Hmax,nH = self.Hs_range[1]
            Kmin,Kmax,nK = self.Ks_range[1]
            hh=np.linspace(Hmin,Hmax,nH)
            kk=np.linspace(Kmin,Kmax,nK)
            Hs_op = self.optimal_Hs[1]
            Ks_op = self.optimal_Ks[1]
            if len(self.std_Hs)==0:
                Hs_std = 0.0
                Ks_std = 0.0
            else:
                Hs_std = self.std_Hs[1]
                Ks_std = self.std_Ks[1]

            ax2.contourf(hh,kk,amp,levels=25)
            ax2.plot([Hs_op-3*Hs_std,Hs_op+3*Hs_std],[Ks_op,Ks_op],lw=1,c='black')
            ax2.plot([Hs_op,Hs_op],[Ks_op-3*Ks_std,Ks_op+3*Ks_std],lw=1,c='black')
            ax2.set_ylim(Kmin,Kmax)
            ax2.set_xlim(Hmin,Hmax)
            ax2.set_title(f'sedimentary layer \n' + r'Thickness:{0:4.2f}$\pm${2:4.2f} Vp/Vs:{1:4.3f}$\pm${3:4.3f}'.\
                           format(Hs_op,Ks_op,3*Hs_std,3*Ks_std))
            ax2.set_ylabel('Vp/Vs')
            ax2.set_xlabel('Thickness [km]')
            ax2.minorticks_on()

            plt.savefig(f"{savedir}/{stationname}_Hk_results.{pic_format}",dpi=900)
            plt.close('all')

        if len(self.Dataset) != 0:

            ##for raw RF
            ray_params = np.array(self.ray_params)
            sampling = 1/self.delta
            stack_data = self.Dataset[0]
            fig, ax = plt.subplots(1,1,figsize=(5,5),tight_layout=True)

            self.plot_RF_rayp(ax,stack_data,ray_params,sampling,0.003,0,30)

            id_phase = 0
            Hc = self.optimal_Hs[0]
            Kc = self.optimal_Ks[0]
            Hs = self.optimal_Hs[1]
            Ks = self.optimal_Ks[1]
            Vpc = self.Vps[0]
            Vps = self.Vps[1]

            ##sediments
            t_Pbs = Hs*((Ks**2/Vps**2-ray_params**2)**(1/2)-(1/Vps**2-ray_params**2)**(1/2))
            ax.plot(ray_params,t_Pbs,lw=1, ls='--', color=list_colornames[id_phase], label="Pbs")
            id_phase +=1
            t_PpPbs = Hs*((Ks**2/Vps**2-ray_params**2)**(1/2)+(1/Vps**2-ray_params**2)**(1/2))
            ax.plot(ray_params,t_PpPbs,lw=1, ls='--', color=list_colornames[id_phase], label="PpPbs")
            id_phase +=1
            t_PpSbs = 2*Hs*(Ks**2/Vps**2-ray_params**2)**(1/2)
            ax.plot(ray_params,t_PpSbs,lw=1, ls='--', color=list_colornames[id_phase], label="PpSbs")
            id_phase +=1
            t_PsSbs = Hs*(3*Ks**2/Vps**2-ray_params**2-(1/Vps**2-ray_params**2)**(1/2))**(1/2)
            ax.plot(ray_params,t_PsSbs,lw=1, ls='--', color=list_colornames[id_phase], label="PsSbs")
            id_phase +=1
            ##crust
            t_Pms = Hc*((Kc**2/Vpc**2-ray_params**2)**(1/2)-(1/Vpc**2-ray_params**2)**(1/2)) + t_Pbs
            ax.plot(ray_params,t_Pms,lw=1, ls='--', color=list_colornames[id_phase], label="Pbs")
            id_phase +=1
            t_PpPms = Hc*((Kc**2/Vpc**2-ray_params**2)**(1/2)+(1/Vpc**2-ray_params**2)**(1/2))+ t_PpPbs
            ax.plot(ray_params,t_PpPms,lw=1, ls='--', color=list_colornames[id_phase], label="PpPbs")
            id_phase +=1
            t_PpSms = 2*Hc*(Kc**2/Vpc**2-ray_params**2)**(1/2) + t_PpSbs
            ax.plot(ray_params,t_PpSms,lw=1, ls='--', color=list_colornames[id_phase], label="PpSbs")

            ax.set_xlim(0.035,0.085)
            ax.set_ylim(0,self.plotlengths[0])
            ax.invert_yaxis()
            ax.set_title(f"RF")
            ax.legend(loc='lower left')

            plt.tight_layout()
            plt.savefig(f"{savedir}/{stationname}_RF_phasefit.{pic_format}",dpi=900)
            plt.close('all')


            ##for Rede
            ray_params = np.array(self.ray_params_DeRe)
            sampling = 1/self.delta
            stack_data = self.Dataset_DeRe[0]
            fig, ax = plt.subplots(1,1,figsize=(5,5),tight_layout=True)

            self.plot_RF_rayp(ax,stack_data,ray_params,sampling,0.003,0,30)

            id_phase = 0
            Hc = self.optimal_Hs[0]
            Kc = self.optimal_Ks[0]
            Hs = self.optimal_Hs[1]
            Ks = self.optimal_Ks[1]
            Vpc = self.Vps[0]
            Vps = self.Vps[1]

            ##sediments
            t_Pbs = Hs*((Ks**2/Vps**2-ray_params**2)**(1/2)-(1/Vps**2-ray_params**2)**(1/2))
            ax.plot(ray_params,t_Pbs,lw=1, ls='--', color=list_colornames[id_phase], label="Pbs")
            id_phase +=1
            t_PpPbs = Hs*((Ks**2/Vps**2-ray_params**2)**(1/2)+(1/Vps**2-ray_params**2)**(1/2))
            ax.plot(ray_params,t_PpPbs,lw=1, ls='--', color=list_colornames[id_phase], label="PpPbs")
            id_phase +=1
            t_PpSbs = 2*Hs*(Ks**2/Vps**2-ray_params**2)**(1/2)
            ax.plot(ray_params,t_PpSbs,lw=1, ls='--', color=list_colornames[id_phase], label="PpSbs")
            id_phase +=1
            t_PsSbs = Hs*(3*Ks**2/Vps**2-ray_params**2-(1/Vps**2-ray_params**2)**(1/2))**(1/2)
            ax.plot(ray_params,t_PsSbs,lw=1, ls='--', color=list_colornames[id_phase], label="PsSbs")
            id_phase +=1
            ##crust
            t_Pms = Hc*((Kc**2/Vpc**2-ray_params**2)**(1/2)-(1/Vpc**2-ray_params**2)**(1/2)) + t_Pbs
            ax.plot(ray_params,t_Pms,lw=1, ls='--', color=list_colornames[id_phase], label="Pbs")
            id_phase +=1
            t_PpPms = Hc*((Kc**2/Vpc**2-ray_params**2)**(1/2)+(1/Vpc**2-ray_params**2)**(1/2))+ t_PpPbs
            ax.plot(ray_params,t_PpPms,lw=1, ls='--', color=list_colornames[id_phase], label="PpPbs")
            id_phase +=1
            t_PpSms = 2*Hc*(Kc**2/Vpc**2-ray_params**2)**(1/2) + t_PpSbs
            ax.plot(ray_params,t_PpSms,lw=1, ls='--', color=list_colornames[id_phase], label="PpSbs")

            ax.set_xlim(0.035,0.085)
            ax.set_ylim(0,self.plotlengths[0])
            ax.invert_yaxis()
            ax.set_title(f"RF")
            ax.legend(loc='lower left')

            plt.tight_layout()
            plt.savefig(f"{savedir}/{stationname}_RFDeRe_phasefit.{pic_format}",dpi=900)
            plt.close('all')
        # if len(self.bootstrap_randoms_Hs)!=0 and len(self.bootstrap_randoms_Ks)!=0:
        #     ##sediments
        #     for j in range(self.Nlayer):
        #         Hs_randoms = self.bootstrap_randoms_Hs[j]
        #         Ks_randoms = self.bootstrap_randoms_Ks[j]
        #         fig = plt.figure(figsize=(6,6))
        #         gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
        #                             left=0.1, right=0.9, bottom=0.1, top=0.9,
        #                             wspace=0.05, hspace=0.05)
        #         ax = fig.add_subplot(gs[1, 0])
        #         ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        #         ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        #         self.scatter_hist(Ks_randoms, Hs_randoms, ax, ax_histx, ax_histy, 0.005,0.02)
        #         ax.set_xlabel('Vp/Vs')
        #         ax.set_ylabel('Thickness [km]')
        #         fig.suptitle(f"Bootstrap of {j}th layer ")
        #         plt.savefig(f"{savedir}/{stationname}_layer{j}_bootstrap.pdf",dpi=900)
        #         plt.close('all')

    @staticmethod
    def phasetime2amp(Ss,t,samp):
        shapet=t.shape
        nrecord=shapet[-1]
        amp_sum=np.zeros(shapet[:-1])
        for i in range(nrecord):
            tt=t[:,:,i]
            tt=tt.reshape(-1)
            tt=np.floor(tt*samp)###return value type oof np.floor is float 
            tt=tt.astype(np.int64)
            ttt=np.ix_(tt,[i])
            aa=Ss[ttt]
            aa=aa.reshape(shapet[:-1])
            amp_sum=amp_sum+aa
        return amp_sum
    
    @staticmethod
    def plot_RF_rayp(ax,Ss,rays,sampling,scale,t0=None,t1=None,):
        delta = 1.0/sampling
        if rays.shape[0] != Ss.shape[1]:
            return -1
        for i,rayp in enumerate(rays):
            data = Ss[:,i]
            data = data*scale
            times = np.arange(len(data))*delta
            if t0!=None and t1!=None:
                index = np.logical_and(times>t0, times<t1)
                times = times[index]
                data = data[index]
            ax.plot(data+rayp,times,lw=0.01,c='k')
            data_fill = data.copy()
            data_fill[data_fill>0]=0
            ax.fill_betweenx(times,data_fill+rayp,rayp,color='red',lw=0.05)
            data_fill = data.copy()
            data_fill[data_fill<0]=0
            ax.fill_betweenx(times,data_fill+rayp,rayp,color='blue',lw=0.05)
        ax.grid(axis='y',lw=0.5,ls='--',color='gray',alpha=0.5)
        ax.set_xlabel('ray_param [s/km]')
        ax.set_ylabel('Time [s]')

    @staticmethod
    def scatter_hist(x, y, ax, ax_histx, ax_histy, binwidthx, binwidthy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y, color='blue')

        # now determine nice limits by hand:
        xmax = x.max(); xmin = x.min()
        midx = (int((xmax+xmin)/binwidthx/2)) * binwidthx
        limx = (int((xmax-xmin)/binwidthx/2) + 1) * binwidthx

        ymax = y.max(); ymin = y.min()
        midy= (int((ymax+ymin)/binwidthy/2)) * binwidthy
        limy = (int((ymax-ymin)/binwidthy/2) + 1) * binwidthy

        bins_x = np.arange(-limx+midx, limx+midx+binwidthx, binwidthx)
        bins_y = np.arange(-limy+midy, limy+midy+binwidthy, binwidthy)
        ax_histx.hist(x, bins=bins_x)
        ax_histy.hist(y, bins=bins_y, orientation='horizontal')
