"""
Code adapted from Brooke Russell (https://github.com/russellphysics/2x2_n_xs/tree/main) 
for extracting truth info from MiniRun4 and making proton selection cuts.
"""
import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
import json
import os
import sys
sys.path.append("plots/")

m_p=938.27208816 # [MeV/c^2]
m_n=939.56542052 # [MeV/c^2]
c_light=29.9792 # [cm/ns]
cuts={
    'none':'No Cuts',\
    'single_nu':r'Single LAr $\nu$',\
    'single_proton':'Single Proton',
    'nu_p_match':r'$\nu$-p Match',\
    'single_particle':r'Single Visible Primary',\
    'MIP_track':r'Single MIP Primary',\
    'MINERvA_track':r'Track Matched to MINER$\nu$A'
}
pdg_label={
    3122:r'$\Lambda$',
    3222:r'$\Sigma^+$',
    3212:r'$\Sigma^0$',
    3112:r'$\Sigma^-$',
    2112:'n', 
    2212:'p',
    22:r'$\gamma$',\
    -211:r'$\pi^-$',
    211:r'$\pi^+$',
    11:'e$^-$',
    -11:'e$^+$',\
    13:r'$\mu^-$',
    -13:r'$\mu^+$',
    111:'$\pi^0$',
    321:'K$^+$',\
    130:r'K$^0_L$',
    321:r'K$^+$',
    -321:r'K$^-$'
}

def get_unique_spills(sim_h5):
    return np.unique(sim_h5['trajectories']['event_id'])

def get_spill_data(sim_h5, spill_id):
    ghdr_spill_mask = sim_h5['mc_hdr'][:]['event_id']==spill_id
    gstack_spill_mask = sim_h5['mc_stack'][:]['event_id']==spill_id
    traj_spill_mask = sim_h5['trajectories'][:]['event_id']==spill_id
    vert_spill_mask = sim_h5['vertices'][:]['event_id']==spill_id
    seg_spill_mask = sim_h5['segments'][:]['event_id']==spill_id
    
    ghdr = sim_h5['mc_hdr'][ghdr_spill_mask]
    gstack = sim_h5['mc_stack'][gstack_spill_mask]
    traj = sim_h5['trajectories'][traj_spill_mask]
    vert = sim_h5['vertices'][vert_spill_mask]
    seg = sim_h5['segments'][seg_spill_mask]
    
    return ghdr, gstack, traj, vert, seg

def tpc_bounds(i):
    active_tpc_widths=[30.6, 130., 64.] # cm
    tpcs_relative_to_module=[[-15.7,0.,0.],[15.7,0.,0.]]
    modules_relative_to_2x2=[
        [-33.5,0.,-33.5],
        [33.5,0.,-33.5],
        [-33.5,0.,33.5],
        [33.5,0.,33.5]
    ]
    detector_center=[0.,-268,1300]
    tpc_bounds=np.array([-active_tpc_widths[i]/2., active_tpc_widths[i]/2.])
    tpc_bounds_relative_to_2x2=[]
    for tpc in tpcs_relative_to_module:
        tpc_bound_relative_to_module = tpc_bounds + tpc[i]
        for module in modules_relative_to_2x2:
            bound = tpc_bound_relative_to_module + module[i]
            tpc_bounds_relative_to_2x2.append(bound)
    bounds_relative_to_NDhall = np.array(tpc_bounds_relative_to_2x2) + detector_center[i]
    return np.unique(bounds_relative_to_NDhall, axis=0)

def fiducialized_vertex(vert_pos):
    flag=False; x_drift_flag=False; y_vertical_flag=False; z_beam_flag=False
    for i in range(3):
        for i_bounds, bounds in enumerate(tpc_bounds(i)):
            if vert_pos[i]>bounds[0] and vert_pos[i]<bounds[1]:
                if i==0: x_drift_flag=True; break
                if i==1: y_vertical_flag=True
                if i==2: z_beam_flag=True
    if x_drift_flag==True and y_vertical_flag==True and z_beam_flag==True:
        flag=True
    return flag

def total_edep_charged_e(traj_id, traj, seg):
    seg_id_mask=seg['traj_id']==traj_id
    total_e=0.
    for sg in seg[seg_id_mask]: total_e+=sg['dE']
    return total_e

def total_edep_length(traj_id, traj, seg):
    seg_id_mask=seg['traj_id']==traj_id
    length=0.
    for sg in seg[seg_id_mask]: 
        length+=np.sqrt((sg['x_start']-sg['x_end'])**2+
                        (sg['y_start']-sg['y_end'])**2+
                        (sg['z_start']-sg['z_end'])**2
                       )
    return length

def three_momentum(pxyz):
    return float(np.sqrt(pxyz[0]**2+pxyz[1]**2+pxyz[2]**2))

def tuple_key_to_string(d):
    out={}
    for key in d.keys():
        string_key=""
        max_length=len(key)
        for i in range(max_length):
            if i<len(key)-1: string_key+=str(key[i])+"-"
            else: string_key+=str(key[i])
        out[string_key]=d[key]
    return out

def save_dict_to_json(d, name, if_tuple):
    with open(name+".json","w") as outfile:
        if if_tuple==True:
            updated_d = tuple_key_to_string(d)
            json.dump(updated_d, outfile, indent=4)
        else:
            json.dump(d, outfile, indent=4)

def np_array_of_array_to_flat_list(a):
    b = list(a)
    return [list(c)[0] for c in b]

def process_file(sim_h5):
    unique_spill = get_unique_spills(sim_h5)
    d=dict()
    for spill_id in unique_spill:
        ghdr, gstack, traj, vert, seg = get_spill_data(sim_h5, spill_id)
        traj_proton_mask = traj['pdg_id']==2212
        proton_traj = traj[traj_proton_mask]
    
        for pt in proton_traj:
        
            # REQUIRE proton contained in 2x2 active LAr
            proton_start=pt['xyz_start']
            if fiducialized_vertex(proton_start)==False: continue
            if fiducialized_vertex(pt['xyz_end'])==False: continue
        
            # is nu vertex contained in 2x2 active LAr?
            vert_mask = vert['vertex_id']==pt['vertex_id']
            nu_vert = vert[vert_mask]
            vert_loc = [nu_vert['x_vert'],nu_vert['y_vert'],nu_vert['z_vert']]
            vert_loc = np_array_of_array_to_flat_list(vert_loc)
            lar_fv = 1
            if fiducialized_vertex(vert_loc)==False: lar_fv = 0
        
            # Find proton parent PDG 
            parent_mask = (traj['traj_id']==pt['parent_id'])
            if sum(parent_mask) == 0: continue
            parent_pdg=traj[parent_mask]['pdg_id']
            if pt['parent_id']==-1:
                ghdr_mask=ghdr['vertex_id']==pt['vertex_id']
                parent_pdg=ghdr[ghdr_mask]['nu_pdg']
        
            # Find proton grandparent PDG 
            grandparent_mask = (traj['traj_id']==traj[parent_mask]['parent_id'])
            grandparent_trackid = traj[grandparent_mask]['traj_id']
            grandparent_pdg = traj[grandparent_mask]['pdg_id']
            if grandparent_trackid.size>0:
                if grandparent_trackid==-1:
                    ghdr_mask=ghdr['vertex_id']==pt['vertex_id']
                    grandparent_pdg=ghdr[ghdr_mask]['nu_pdg']
            grandparent_pdg=list(grandparent_pdg)
            if len(grandparent_pdg)==0: grandparent_pdg=[0]
            
            if parent_pdg[0] not in [12,14,16,-12,-14,-16]:
                parent_total_energy = float(list(traj[parent_mask]['E_start'])[0])
                parent_length = float(total_edep_length(traj[parent_mask]['traj_id'], traj, seg))
                parent_start_momentum = float(three_momentum(traj[parent_mask]['pxyz_start'][0]))
                parent_end_momentum = float(three_momentum(traj[parent_mask]['pxyz_end'][0]))
            else:
                parent_total_energy = float(-1) 
                parent_length = float(-1)
                parent_start_momentum = float(-1)
                parent_end_momentum = float(-1)
                
            gstack_mask = gstack['vertex_id']==pt['vertex_id']
            gstack_traj_id = gstack[gstack_mask]['traj_id']
            primary_length=[]; primary_pdg=[]
            for gid in gstack_traj_id:
                primary_mask = traj['traj_id']==gid
                primary_start = traj[primary_mask]['xyz_start']
                primary_end = traj[primary_mask]['xyz_end']
                p_pdg = traj[primary_mask]['pdg_id']
                if len(p_pdg)==1:
                    primary_pdg.append(int(p_pdg[0]))
                    dis = np.sqrt( (primary_start[0][0]-primary_end[0][0])**2+
                                   (primary_start[0][1]-primary_end[0][1])**2+
                                   (primary_start[0][2]-primary_end[0][2])**2)
                    primary_length.append(float(dis))
                      
            p_start = proton_start.tolist()
            p_vtx = []
            for i in p_start: p_vtx.append(float(i))
            
            nu_vtx=[]
            for i in vert_loc: nu_vtx.append(float(i))
            d[(spill_id, pt['vertex_id'], pt['traj_id'])]=dict(
                lar_fv=int(lar_fv),
                
                p_vtx=p_vtx,
                nu_vtx=nu_vtx,
                
                proton_total_energy = float(pt['E_start']),
                proton_vis_energy = float(total_edep_charged_e(pt['traj_id'], traj, seg)),
                proton_length = float(total_edep_length(pt['traj_id'], traj, seg)),
                proton_start_momentum = float(three_momentum(pt['pxyz_start'])),
                proton_end_momentum = float(three_momentum(pt['pxyz_end'])),
                
                parent_total_energy = parent_total_energy, 
                parent_length = parent_length, 
                parent_start_momentum = parent_start_momentum, 
                parent_end_momentum = parent_end_momentum, 
                
                nu_proton_dt = float(pt['t_start']) - float(nu_vert['t_vert'][0]),
                nu_proton_distance = float(np.sqrt( (proton_start[0]-vert_loc[0])**2+
                                          (proton_start[1]-vert_loc[1])**2+
                                          (proton_start[2]-vert_loc[2])**2 )),
                
                parent_pdg=int(parent_pdg[0]),
                grandparent_pdg=int(grandparent_pdg[0]),
                primary_pdg=primary_pdg,                
                primary_length=primary_length
            )
    return d

def tpc_vertex(vert_pos):
    temp=[]
    for i in range(3): temp.append(tpc_bounds(i).tolist())
    tpc_fv={}
    for i in range(8): tpc_fv[i]=False
    tpc=0
    enclosed=False
    for x in range(4):
        for y in range(1):
            for z in range(2):
                if vert_pos[0]>temp[0][x][0] and vert_pos[0]<temp[0][x][1] and\
                   vert_pos[1]>temp[1][y][0] and vert_pos[1]<temp[1][y][1] and\
                   vert_pos[2]>temp[2][z][0] and vert_pos[2]<temp[2][z][1]:
                    tpc_fv[tpc]=True
                    return tpc_fv
                tpc+=1
    return tpc_fv

def files_processed(processed_files, total_files=1023, \
                    production_pot=1e19, target_pot=2.5e19):
    return target_pot/((processed_files*production_pot)/total_files)

def find_lar_nu_spill_multiplicity(d):
    out={}
    for k in d.keys():
        spill=k.split("-")[0]
        vertex=k.split("-")[1]
        track=k.split("-")[2]
        if spill not in out.keys(): out[spill]=[0,-1]
        if d[k]['lar_fv']==1: out[spill][0]+=1; out[spill][1]=vertex                
    return out

def find_proton_spill_multiplicity(d):
    out={}
    for k in d.keys():
        spill=k.split("-")[0]
        vertex=k.split("-")[1]
        track=k.split("-")[2]
        if spill not in out.keys(): out[spill]=0
        if d[k]['lar_fv']==1: out[spill]+=1
    return out

def single_track_primaries(primaries):
    tracks=0
    tracks+=primaries.count(11) # e -
    tracks+=primaries.count(-11) # e +
    tracks+=primaries.count(13) # mu -
    tracks+=primaries.count(-13) # mu +
    tracks+=primaries.count(22) # gamma
    tracks+=primaries.count(111) # pi0
    tracks+=primaries.count(130) # K 0 L
    tracks+=primaries.count(211) # pi +
    tracks+=primaries.count(-211) # pi -
    tracks+=primaries.count(221) # eta
    tracks+=primaries.count(310) # K 0 S
    tracks+=primaries.count(311) # K 0
    tracks+=primaries.count(-311) # K 0 
    tracks+=primaries.count(321) # K +
    tracks+=primaries.count(-321) # K -
    tracks+=primaries.count(411) # D +
    tracks+=primaries.count(-411) # D -
    tracks+=primaries.count(421) # D 0
    tracks+=primaries.count(2212) # p
    tracks+=primaries.count(3122) # lambda
    tracks+=primaries.count(3222) # sigma +
    tracks+=primaries.count(3212) # sigma 0
    tracks+=primaries.count(3112) # sigma -    
    return tracks

def find_reference_track(pdg, lengths):
    tracks=[]
    l=[]
    for p in range(len(pdg)):
        if pdg[p] in [12,14,16,-12,-14,-16]: continue
        elif pdg[p] > 100000: continue
        elif pdg[p]==2112: continue
        else: tracks.append(pdg[p]); l.append(lengths[p])
    return tracks, l
    
def fill_dict(calc, cut, parent_pdg, grandparent_pdg, \
              reco_ke, true_ke, tof, \
              nu_proton_distance, proton_length, parent_length):
    if parent_pdg==2112 and grandparent_pdg==0:
        calc[cut]['initscat_frac_diff'].append( (reco_ke-true_ke)/true_ke )
        calc[cut]['initscat_tof'].append( tof )
        calc[cut]['initscat_dis'].append( nu_proton_distance )
        calc[cut]['initscat_length'].append( proton_length )
        calc[cut]['initscat_plength'].append( parent_length )
        calc[cut]['initscat_nke'].append( true_ke )
    elif parent_pdg==2112 and grandparent_pdg==2112:
        calc[cut]['rescat_frac_diff'].append( (reco_ke-true_ke)/true_ke )
        calc[cut]['rescat_tof'].append( tof )
        calc[cut]['rescat_dis'].append( nu_proton_distance )
        calc[cut]['rescat_length'].append( proton_length )
        calc[cut]['rescat_plength'].append( parent_length )
        calc[cut]['rescat_nke'].append( true_ke )
    elif parent_pdg in [12,14,16]:
        calc[cut]['nu_frac_diff'].append( (reco_ke-true_ke)/true_ke )
        calc[cut]['nu_tof'].append( tof )
        calc[cut]['nu_dis'].append( nu_proton_distance )
        calc[cut]['nu_length'].append( proton_length )
        calc[cut]['nu_plength'].append( parent_length )
    else:
        calc[cut]['other_frac_diff'].append( (reco_ke-true_ke)/true_ke )
        calc[cut]['other_tof'].append( tof )
        calc[cut]['other_dis'].append( nu_proton_distance )
        calc[cut]['other_length'].append( proton_length )
        calc[cut]['other_plength'].append( parent_length )
    return

def cut_variation(calc, file_ctr):
    out={}
    for k in calc.keys():
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,8))

        scale_factor=files_processed(file_ctr)
        total=len(calc[k]['initscat_tof'])+len(calc[k]['rescat_tof'])+len(calc[k]['nu_tof'])+len(calc[k]['other_tof'])

        initscat=(len(calc[k]['initscat_tof'])/total)*100
        w_initscat=[scale_factor]*len(calc[k]['initscat_tof'])
        n_initscat=len(calc[k]['initscat_tof'])*scale_factor

        rescat=(len(calc[k]['rescat_tof'])/total)*100
        w_rescat=[scale_factor]*len(calc[k]['rescat_tof'])
        n_rescat=len(calc[k]['rescat_tof'])*scale_factor

        nu=(len(calc[k]['nu_tof'])/total)*100
        w_nu=[scale_factor]*len(calc[k]['nu_tof'])
        n_nu=len(calc[k]['nu_tof'])*scale_factor

        other=(len(calc[k]['other_tof'])/total)*100
        w_other=[scale_factor]*len(calc[k]['other_tof'])
        n_other=len(calc[k]['other_tof'])*scale_factor

        out[k]=[[initscat,rescat,nu,other],
                [n_initscat,n_rescat,n_nu,n_other]]

        bins=np.linspace(0,40,41)
        ax[0][0].hist(calc[k]['initscat_tof'], bins=bins, \
                      weights=w_initscat, histtype='step', linewidth=2, \
                   label=r'n progenitor, 1st scatter ({:.1f}%)'.format(initscat))
        ax[0][0].hist(calc[k]['rescat_tof'], bins=bins, \
                      weights=w_rescat, histtype='step', linewidth=2, \
                   label=r'n progenitor, rescatter ({:.1f}%)'.format(rescat))
        ax[0][0].hist(calc[k]['nu_tof'], bins=bins, \
                      weights=w_nu, histtype='step', linewidth=2,\
                   label=r'$\nu$ progenitor ({:.1f}%)'.format(nu))
        ax[0][0].hist(calc[k]['other_tof'], bins=bins, \
                      weights=w_other, histtype='step', linewidth=2, \
                   label='Other progenitor ({:.1f}%)'.format(other))        
        ax[0][0].set_xlabel('TOF [ns]')
        ax[0][0].set_ylabel('Event Count / ns')
        ax[0][0].set_yscale('log')
        ax[0][0].set_xlim(0,40)
        ax[0][0].legend(loc='upper right')

        bins=np.linspace(0,200,41)
        ax[0][1].hist(calc[k]['initscat_dis'], bins=bins, \
                      weights=w_initscat, histtype='step', linewidth=2)
        ax[0][1].hist(calc[k]['rescat_dis'], bins=bins, \
                      weights=w_rescat, histtype='step', linewidth=2)
        ax[0][1].hist(calc[k]['nu_dis'], bins=bins, \
                      weights=w_nu, histtype='step', linewidth=2)
        ax[0][1].hist(calc[k]['other_dis'], bins=bins, \
                      weights=w_other, histtype='step', linewidth=2)
        ax[0][1].set_xlabel(r'$\nu$-to-p Distance [cm]')
        ax[0][1].set_ylabel('Event Count / 5 cm')
        ax[0][1].set_yscale('log')
        ax[0][1].set_xlim(0,200)
    
        bins=np.linspace(-1,1,51)
        ax[0][2].hist(calc[k]['initscat_frac_diff'], bins=bins, \
                      weights=w_initscat, histtype='step', \
                   linewidth=2)
        ax[0][2].hist(calc[k]['rescat_frac_diff'], bins=bins, \
                      weights=w_rescat, histtype='step', \
                   linewidth=2)
        ax[0][2].hist(calc[k]['nu_frac_diff'], bins=bins, \
                      weights=w_nu, histtype='step', \
                   linewidth=2)
        ax[0][2].hist(calc[k]['other_frac_diff'], bins=bins, \
                      weights=w_other, histtype='step', \
                   linewidth=2)
        ax[0][2].set_xlim(-1,1)
        ax[0][2].set_xlabel(r'(T$_{reco}$-T$_{true}$)/T$_{true}$')
        ax[0][2].set_ylabel('Event Count')
        ax[0][2].set_yscale('log')

        bins=np.linspace(0,100,21)
        ax[1][0].hist(calc[k]['initscat_length'], bins=bins, \
                      weights=w_initscat, histtype='step', linewidth=2)
        ax[1][0].hist(calc[k]['rescat_length'], bins=bins, \
                      weights=w_rescat, histtype='step', linewidth=2)
        ax[1][0].hist(calc[k]['nu_length'], bins=bins, \
                      weights=w_nu, histtype='step', linewidth=2)
        ax[1][0].hist(calc[k]['other_length'], bins=bins, \
                      weights=w_other, histtype='step', linewidth=2)
        ax[1][0].set_xlabel(r'Proton Track Length [cm]')
        ax[1][0].set_ylabel('Event Count / 5 cm')
        ax[1][0].set_yscale('log')
        ax[1][0].set_xlim(0,100)

        bins=np.linspace(0,1000,21)
        ax[1][1].hist(calc[k]['initscat_plength'], bins=bins, \
                      weights=w_initscat, histtype='step', linewidth=2)
        ax[1][1].hist(calc[k]['rescat_plength'], bins=bins, \
                      weights=w_rescat, histtype='step', linewidth=2)
        ax[1][1].hist(calc[k]['nu_plength'], bins=bins, \
                      weights=w_nu, histtype='step', linewidth=2)
        ax[1][1].hist(calc[k]['other_plength'], bins=bins, \
                      weights=w_other, histtype='step', linewidth=2)
        ax[1][1].set_xlabel(r'Parent Track Length [cm]')
        ax[1][1].set_ylabel('Event Count / 50 cm')
        ax[1][1].set_yscale('log')
        ax[1][1].set_xlim(0,1000)

        bins=np.linspace(0,1000,41)
        ax[1][2].hist(calc[k]['initscat_nke'], bins=bins, \
                      weights=w_initscat, histtype='step', linewidth=2)
        ax[1][2].hist(calc[k]['rescat_nke'], bins=bins, \
                      weights=w_rescat, histtype='step', linewidth=2)
        ax[1][2].hist(calc[k]['nu_nke'], bins=bins, \
                      weights=[scale_factor]*len(calc[k]['nu_nke']), \
                      histtype='step', linewidth=2)
        ax[1][2].hist(calc[k]['other_nke'], bins=bins, \
                      weights=[scale_factor]*len(calc[k]['other_nke']), \
                      histtype='step', linewidth=2)
        ax[1][2].set_xlabel(r'T$_n$ [MeV]')
        ax[1][2].set_ylabel('Event Count / 25 MeV')
        ax[1][2].set_yscale('log')
        ax[1][2].set_xlim(0,1000)
    
        fig.tight_layout()
        fig.suptitle(cuts[k])
        plt.savefig('plots/minirun4_statistics_1.png')
        plt.show()
    return out

def piechart_single_vis_particle_at_vertex(primary_single_track):
    fig, ax = plt.subplots(figsize=(6,6))
    pst_set = set(primary_single_track)
    pst_count = [(p, primary_single_track.count(p)) for p in pst_set]
    pst_fraction = [100*(i[1]/len(primary_single_track)) for i in pst_count]
    pst_label=[pdg_label[i[0]] for i in pst_count]
    ax.pie(pst_fraction, labels=pst_label, autopct='%1.1f%%')
    ax.set_title(r'Single Visible Particle at $\nu$ Vertex')
    plt.tight_layout()
    plt.savefig('plots/minirun4_piechart_vis_particle.png')
    plt.show()

def piechart_mip_reference_proton_parent(mrpp):
    fig, ax = plt.subplots(figsize=(6,6))
    mrpp_set = set(mrpp)
    print(mrpp_set)
    mrpp_count = [(p, mrpp.count(p)) for p in mrpp_set]
    mrpp_fraction = [100*(i[1]/len(mrpp)) for i in mrpp_count]
    mrpp_label=[pdg_label[i[0]] for i in mrpp_count]
    ax.pie(mrpp_fraction, labels=mrpp_label, autopct='%1.1f%%')
    ax.set_title(r'Proton Progenitor'+'\n'+r'Provided Single MIP at $\nu$ Vertex')
    plt.tight_layout()
    plt.savefig('plots/minirun4_piechart_single_mip.png')
    plt.show()    
    
def sample_fraction(d):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot([cuts[k] for k in d.keys()],[d[k][0][0]/100. for k in d.keys()],\
            'o--', label='n progenitor, 1st scatter')
    ax.plot([cuts[k] for k in d.keys()],[d[k][0][1]/100. for k in d.keys()],\
            'o--', label='n progenitor, rescatter')
    ax.plot([cuts[k] for k in d.keys()],[d[k][0][2]/100. for k in d.keys()],\
            'o--', label=r'$\nu$ progenitor')
    ax.plot([cuts[k] for k in d.keys()],[d[k][0][3]/100. for k in d.keys()],\
            'o--', label=r'Other progenitor')
    ax.set_ylabel('Sample Fraction')
    ax.set_ylim(0,1)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig("plots/minirun4_sample_fraction.png")
    plt.show()

def sample_event_count(d):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot([cuts[k] for k in d.keys()],[d[k][1][0] for k in d.keys()],\
            'o--', label='n progenitor, 1st scatter')
    ax.plot([cuts[k] for k in d.keys()],[d[k][1][1] for k in d.keys()],\
            'o--', label='n progenitor, rescatter')
    ax.plot([cuts[k] for k in d.keys()],[d[k][1][2] for k in d.keys()],\
            'o--', label=r'$\nu$ progenitor')
    ax.plot([cuts[k] for k in d.keys()],[d[k][1][3] for k in d.keys()],\
            'o--', label=r'Other progenitor')
    ax.set_ylabel('Sample Event Count')
    ax.set_yscale('log')
    ax.set_title('2.5E19 POT ME RHC NuMI')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/minirun4_event_count.png')
    plt.show()

def analysis_selection_tof(location_tof, file_ctr):
    fig, ax = plt.subplots(figsize=(6,4))
    scale_factor=files_processed(file_ctr)
    bins=np.linspace(0,15,16)
    for key in location_tof.keys():
        ax.hist(location_tof[key], bins=bins, \
                weights=[scale_factor]*len(location_tof[key]), \
                label=key+' ({:.0f} events)'.format(scale_factor*len(location_tof[key])), \
                alpha=0.5, linewidth=2)
    ax.set_xlabel('TOF [ns]')
    ax.set_ylabel('Event Count / ns')
    ax.set_xlim(0,15)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/minirun4_tof.png')
    plt.show()
    

def analysis_selection_proton_length(selected_proton_track_length, file_ctr):
    a ={'initscat':'n progenitor, 1st scatter',
        'rescat':'n progenitor, rescatter',
        'nu':r'$\nu$ progenitor',
        'other':'Other progenitor'}
    fig, ax = plt.subplots(figsize=(6,4))
    scale_factor=files_processed(file_ctr)
    bins=np.linspace(0,70,36)
    for key in selected_proton_track_length.keys():
        ax.hist(selected_proton_track_length[key], bins=bins, \
                weights=[scale_factor]*len(selected_proton_track_length[key]), \
                label=a[key]+' ({:.0f} events)'.format(scale_factor*len(selected_proton_track_length[key])), \
                alpha=0.5, linewidth=2)
    ax.set_xlabel('Selected Proton Track Length [cm]')
    ax.set_ylabel('Event Count / 2 cm')
    ax.set_xlim(0,70)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/minirun4_proton_track_length.png')
    plt.show()
      

def analysis_selection_neutron_true_ke(selected_neutron_true_ke, file_ctr):
    a ={'initscat':'n progenitor, 1st scatter',
        'rescat':'n progenitor, rescatter',
        'nu':r'$\nu$ progenitor',
        'other':'Other progenitor'}
    fig, ax = plt.subplots(figsize=(6,4))
    scale_factor=files_processed(file_ctr)
    bins=np.linspace(0,800,31)
    for key in selected_neutron_true_ke.keys():
        ax.hist(selected_neutron_true_ke[key], bins=bins, \
                weights=[scale_factor]*len(selected_neutron_true_ke[key]), \
                label=a[key]+' ({:.0f} events)'.format(scale_factor*len(selected_neutron_true_ke[key])), \
                alpha=0.5, linewidth=2)
    ax.set_xlabel('Selected Neutron True KE [MeV]')
    ax.set_ylabel('Event Count / 25 MeV')
    ax.set_xlim(0,800)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/minirun4_neutron_true_ke.png')
    plt.show()
    
    
def mip_reference_length_dist(mip_reference_length, file_ctr):
    a ={'initscat':'n progenitor, 1st scatter',
        'rescat':'n progenitor, rescatter',
        'nu':r'$\nu$ progenitor',
        'other':'Other progenitor'}
    scale_factor = files_processed(file_ctr)
    bins=np.linspace(0,2500,101)
    fig, ax = plt.subplots(figsize=(8,6))
    for k in mip_reference_length.keys():
        ax.hist(mip_reference_length[k], bins=bins, \
                weights=[scale_factor]*len(mip_reference_length[k]), \
                label=a[k], histtype='step', linewidth=2)
    ax.set_xlabel('MIP Primary Track Length [cm]')
    ax.set_ylabel('Event Count / 25 cm')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/minirun4_mip_track_length.png')
    plt.show()
    
def location(tpc_p, tpc_v):
    p=-1; v=-1
    for key in tpc_p.keys():
        if tpc_p[key]==True: p=key
    for key in tpc_v.keys():
        if tpc_v[key]==True: v=key
    if p==v: return 'same TPC'
    if p==1 and v==2: return 'same module'
    if p==2 and v==1: return 'same module'
    if p==3 and v==4: return 'same module'
    if p==4 and v==3: return 'same module'
    if p==5 and v==6: return 'same module'
    if p==6 and v==5: return 'same module'
    if p==7 and v==8: return 'same module'
    if p==8 and v==7: return 'same module'
    return 'different module'   