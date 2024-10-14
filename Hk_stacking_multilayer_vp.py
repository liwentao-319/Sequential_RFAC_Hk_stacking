
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


list_colornames = ['orange','gold','cyan','skyblue','blueviolet','violet','pink']


class Hk_searching_RFAC_multilayer(object):
    """
    Discription   : Modified multilayer H-k stacking
    Author        : Wentao Li
    Time          : Jul 6th, 2023
    Version       : 1.0

    """
    def __init__(self, stationname, data_st=[], dataparam={}, priorparam={}, phaseparam={}, phasestackparams=[0.04,0.08,0.002,0.004,2]):
        """
        dataset [list] : data used to Hk stacking, a list of obspy Streams
        dataparam [dic] : Directory contain data information, such as sampling interval (delta), time length before zeto time(pretimes), length used to Hk stacking (length)
        priorparam [dic] : Directory contain prior parameters for Hk stacking
        phaseparam [list] : list of list to specify seismic phases in each layer
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
        ##resolve priorparam
        self.Nlayer = priorparam['Nlayer']
        self.Vps = priorparam['Vps']
        self.Hs_range = priorparam['Hs']
        self.Ks_range = priorparam['Ks']
     
        ##check the parameter length
        if len(self.Vps)!=self.Nlayer or len(self.Hs_range)!=self.Nlayer or len(self.Ks_range)!=self.Nlayer :
            print("Configure Error: length of Vps, Hs or Ks is not euqal to Nlayer")
            raise

        ##resolve seismic phases
        self.phases = phaseparam

        ##check seismic phases
        if len(self.phases)!=self.Nlayer:
            print("Configure Error: the number of seismic phases groups doesn't match the number of layers")
            raise
        for i in range(self.Nlayer):
            if len(self.phases[i])<2:
                print(f"Configure Error: no enough seismic phases for the {i}th layer")
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
                self.data_st[i][j].resample(1/self.delta)
                self.data_st[i][j].trim(starttime+pretime,starttime+pretime+length)
                norm = max(norm_type*self.data_st[i][j].data)
                # norm = max([0.001, norm])
                self.data_st[i][j].data = self.data_st[i][j].data/norm
                
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
        self.optimal_Hs, self.optimal_Ks, self.Hk_amps = self.searching(self.ray_params, self.Dataset)

    def searching(self, ray_params, Dataset):
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
        for n in range(Nlayer):
            #prior Vp
            Vp = self.Vps[n]
            ##searching domain
            Hmin,Hmax,nH = self.Hs_range[n]
            Kmin,Kmax,nK = self.Ks_range[n]
            hh=np.linspace(Hmin,Hmax,nH)
            kk=np.linspace(Kmin,Kmax,nK)
            H,K=np.meshgrid(hh,kk)
            H=H[:,:,None]
            K=K[:,:,None]
            ##initialize the stacking
            Amp = np.zeros([nK,nH])
            ##load seismic phase
            phases = self.phases[n]
            nphase = len(phases)
            for m in range(nphase):
                weight, data_index, nts, ntp, other_ntpts  = phases[m]
                # print(data_index,len(Dataset))
                stack_data = Dataset[data_index]
                ttt = H*(nts*(K**2/Vp**2-ray_params**2)**(1/2)+ntp*(1/Vp**2-ray_params**2)**(1/2))
                nlayer_phase = int(len(other_ntpts)/3)
                for k in range(nlayer_phase):
                    klayer, nts, ntp = other_ntpts[3*k:3*k+3]
                    H_kk = Hs_op[klayer]
                    K_kk = Ks_op[klayer]
                    Vpk = self.Vps[klayer]
                    ttt_kk = H_kk*(nts*(K_kk**2/Vpk**2-ray_params**2)**(1/2)+ntp*(1/Vpk**2-ray_params**2)**(1/2))
                    ttt = ttt + ttt_kk[None,None,:]                                    
                Amp = Amp + weight*self.phasetime2amp(stack_data, ttt, sampling)
            ##find the maximum amp point
            index_max=Amp.argmax()
            ik=floor(index_max/nH)
            ih=index_max%nH
            Hs_op.append(hh[ih])
            Ks_op.append(kk[ik])
            Amps.append(Amp)

        return Hs_op, Ks_op, Amps


    def hk_bootstrap_estimate(self,Nb,nthread=None):

        ##sharing variables
        ntrace = len(self.ray_params)

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
            for i in range(self.Ndataset):
                random_Dataset.append(self.Dataset[i][:,index])
            random_ray_params = [self.ray_params[iii] for iii in index]
            Hs_op, Ks_op, _= self.searching(random_ray_params,random_Dataset)
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

    def save_result_to_figs(self,savedir):
        stationname = self.stationname
        sampling = 1/self.delta
        if len(self.Hk_amps)!= 0 :
            for j in range(self.Nlayer):
                fig = plt.figure(1,figsize=(5,6),tight_layout=True)
                ax1=fig.add_subplot(111)
                amp = self.Hk_amps[j]
                Hmin,Hmax,nH = self.Hs_range[j]
                Kmin,Kmax,nK = self.Ks_range[j]
                hh=np.linspace(Hmin,Hmax,nH)
                kk=np.linspace(Kmin,Kmax,nK)
                Hs_op = self.optimal_Hs[j]
                Ks_op = self.optimal_Ks[j]
                if len(self.std_Hs)==0:
                    Hs_std = 0.0
                    Ks_std = 0.0
                else:
                    Hs_std = self.std_Hs[j]
                    Ks_std = self.std_Ks[j]

                ax1.contourf(hh,kk,amp,levels=25)
                ax1.plot([Hs_op-3*Hs_std,Hs_op+3*Hs_std],[Ks_op,Ks_op],lw=1,c='black')
                ax1.plot([Hs_op,Hs_op],[Ks_op-3*Ks_std,Ks_op+3*Ks_std],lw=1,c='black')
                ax1.set_title(f'{j}th layer \n' + r'Thickness:{0:4.2f}$\pm${2:4.2f} Vp/Vs:{1:4.3f}$\pm${3:4.3f}'.\
                              format(Hs_op,Ks_op,3*Hs_std,3*Ks_std, j))
                ax1.set_ylabel('Vp/Vs')
                ax1.set_xlabel('Thickness [km]')
                ax1.minorticks_on()
                plt.savefig(f"{savedir}/{stationname}_layer{j}_Hk_results.pdf",dpi=900)
                plt.close('all')

        if len(self.Dataset) != 0:
            ray_params = np.array(self.ray_params)
            sampling = 1/self.delta
            for j in range(self.Ndataset):
                stack_data = self.Dataset[j]
                fig, ax = plt.subplots(1,1,figsize=(4,4),tight_layout=True)

                self.plot_RF_rayp(ax,stack_data,ray_params,sampling,0.002,0,30)
                id_phase = 0
                for n in range(self.Nlayer):
                    phases = self.phases[n]
                    nphase = len(phases)
                    Vp = self.Vps[n]
                    for m in range(nphase):
                        weight, data_index, nts, ntp, other_ntpts  = phases[m]
                        if data_index!=j:
                            continue
                        H_n = self.optimal_Hs[n]
                        K_n = self.optimal_Ks[n]
                        t = H_n*(nts*(K_n**2/Vp**2-ray_params**2)**(1/2)+ntp*(1/Vp**2-ray_params**2)**(1/2))
                        ##add traveling times in previous searched layers
                        nlayer_phase = int(len(other_ntpts)/3)
                        for k in range(nlayer_phase):
                            klayer, nts, ntp = other_ntpts[3*k:3*k+3]
                            H_klayer = self.optimal_Hs[klayer]
                            K_klayer = self.optimal_Ks[klayer]
                            Vpk = self.Vps[k]
                            t = t + H_klayer*(nts*(K_klayer**2/Vpk**2-ray_params**2)**(1/2)+ntp*(1/Vpk**2-ray_params**2)**(1/2))
                        ax.plot(ray_params,t,lw=3, ls='--', color=list_colornames[id_phase], label=f"n{n}:phase{m}")
                        id_phase +=1
                
                ax.set_ylim(0,self.plotlengths[j])
                ax.invert_yaxis()
                ax.set_title(f" {j}th dataset")
                ax.legend(loc='lower left')

                plt.tight_layout()
                plt.savefig(f"{savedir}/{stationname}_dataset{j}_phasefit.pdf",dpi=900)
                plt.close('all')

        if len(self.bootstrap_randoms_Hs)!=0 and len(self.bootstrap_randoms_Ks)!=0:
            ##sediments
            for j in range(self.Nlayer):
                Hs_randoms = self.bootstrap_randoms_Hs[j]
                Ks_randoms = self.bootstrap_randoms_Ks[j]
                fig = plt.figure(figsize=(6,6))
                gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                                    wspace=0.05, hspace=0.05)
                ax = fig.add_subplot(gs[1, 0])
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
                self.scatter_hist(Ks_randoms, Hs_randoms, ax, ax_histx, ax_histy, 0.005,0.02)
                ax.set_xlabel('Vp/Vs')
                ax.set_ylabel('Thickness [km]')
                fig.suptitle(f"Bootstrap of {j}th layer ")
                plt.savefig(f"{savedir}/{stationname}_layer{j}_bootstrap.pdf",dpi=900)
                plt.close('all')

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
            ax.plot(data+rayp,times,lw=0.5,c='k')
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
