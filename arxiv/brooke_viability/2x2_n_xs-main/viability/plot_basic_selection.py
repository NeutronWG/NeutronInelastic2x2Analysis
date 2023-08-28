import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import json

m_p=938.27208816 # [MeV/c^2]
m_n=939.56542052 # [MeV/c^2]
c_light=29.9792 # [cm/ns]
cuts={'none':'No Cuts',\
      'single_nu':r'Single LAr $\nu$',\
      'single_proton':'Single Proton',
      'nu_p_match':r'$\nu$-p Match',\
      'single_particle':r'Single Visible Primary',\
      'MIP_track':r'Single MIP Primary',\
      'MINERvA_track':r'Track Matched to MINER$\nu$A'}
pdg_label={3122:r'$\Lambda$',2112:'n', 2212:'p',22:r'$\gamma$',\
           -211:r'$\pi^-$',211:r'$\pi^+$',11:'e$^-$',-11:'e$^+$',\
           13:r'$\mu^-$',-13:r'$\mu^+$',111:'$\pi^0$',321:'K$^+$',\
           130:r'K$^0_L$',321:r'K$^+$',-321:r'K$^-$'}


def tpc_bounds(i):
    active_tpc_widths = [30.6, 130., 64.] # cm                                  
    tpcs_relative_to_module = [[-15.7,0.,0.], [15.7, 0., 0.]]
    modules_relative_to_2x2= [[-33.5,0.,-33.5],
                              [33.5,0.,-33.5],
                              [-33.5,0.,33.5],
                              [33.5,0.,33.5]]
    detector_center = [0.,52.25,0.]
    tpc_bounds = np.array([-active_tpc_widths[i]/2., active_tpc_widths[i]/2.])
    tpc_bounds_relative_to_2x2 = []
    for tpc in tpcs_relative_to_module:
        tpc_bound_relative_to_module = tpc_bounds + tpc[i]
        for module in modules_relative_to_2x2:
            bound = tpc_bound_relative_to_module + module[i]
            tpc_bounds_relative_to_2x2.append(bound)

    bounds_relative_to_NDhall = np.array(tpc_bounds_relative_to_2x2) + detector_center[i]

    return np.unique(bounds_relative_to_NDhall, axis = 0)


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
    
#        fig.tight_layout()
        fig.suptitle(cuts[k])
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
    fig.tight_layout()
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
    plt.show()
    fig.tight_layout()


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
    plt.show()
    fig.tight_layout()    


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
    plt.show()
    fig.tight_layout()    

    
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
    


def main(file_dir):
    calc=dict()
    for c in cuts.keys():
        calc[c]=dict(
            initscat_frac_diff=[], rescat_frac_diff=[], nu_frac_diff=[], other_frac_diff=[],
            initscat_tof=[], rescat_tof=[], nu_tof=[], other_tof=[],
            initscat_dis=[], rescat_dis=[], nu_dis=[], other_dis=[],
            initscat_length=[], rescat_length=[], nu_length=[], other_length=[],
            initscat_plength=[], rescat_plength=[], nu_plength=[], other_plength=[],
            initscat_nke=[], rescat_nke=[], nu_nke=[], other_nke=[]
        )
    primary_single_track=[]
    mip_reference_proton_parent=[]
    mip_reference_length={'initscat':[],'rescat':[],'nu':[],'other':[]}
    location_tof={'same TPC':[],'same module':[],'different module':[]}
    file_ctr=0
    selected_proton_track_length={'initscat':[],'rescat':[],'nu':[],'other':[]}
    selected_neutron_true_ke={'initscat':[],'rescat':[],'nu':[],'other':[]}

    for filename in glob.glob(file_dir+'*.json'):
        with open(filename) as input_file: d = json.load(input_file)
        file_ctr+=1
        spill_lar_multiplicity = find_lar_nu_spill_multiplicity(d)
        proton_spill_multiplicity = find_proton_spill_multiplicity(d)
        for k in d.keys():
            spill=k.split("-")[0]
            vertex=k.split("-")[1]
            
            temp_tof = d[k]['nu_proton_dt']*1e3
            temp_dis = d[k]['nu_proton_distance']
            if temp_tof==0: gamma=-10.
            else: gamma=1/np.sqrt(1-(temp_dis/(temp_tof*c_light))**2)
            reco_ke = (gamma-1)*m_n
            true_ke = d[k]['parent_total_energy']-m_n
            proton_length = d[k]['proton_length']
            parent_length=d[k]['parent_length']
            parent_pdg = d[k]['parent_pdg']
            grandparent_pdg = d[k]['grandparent_pdg']

            if parent_pdg==2112: parent_length=temp_dis

            # no cuts
            fill_dict(calc, 'none', parent_pdg, grandparent_pdg, \
                      reco_ke, true_ke, temp_tof, \
                      temp_dis, proton_length, parent_length)

            # single nu in LAr active volume
            if spill_lar_multiplicity[spill][0]!=1: continue
            fill_dict(calc, 'single_nu', parent_pdg, grandparent_pdg, \
                      reco_ke, true_ke, temp_tof, \
                      temp_dis, proton_length, parent_length)

            # single proton in LAr active volume
            if proton_spill_multiplicity[spill]!=1: continue
            fill_dict(calc, 'single_proton', parent_pdg, grandparent_pdg, \
                      reco_ke, true_ke, temp_tof, \
                      temp_dis, proton_length, parent_length)                

            # proton matched to nu
            if spill_lar_multiplicity[spill][1]!=vertex: continue
            fill_dict(calc, 'nu_p_match', parent_pdg, grandparent_pdg, \
                      reco_ke, true_ke, temp_tof, \
                      temp_dis, proton_length, parent_length)                    

            # single primary particle
            tracks = single_track_primaries(d[k]['primary_pdg'])

            if tracks!=1: continue
            reference_track, reference_length = find_reference_track(d[k]['primary_pdg'],
                                                                     d[k]['primary_length'])
            for rf in reference_track: primary_single_track.append(rf)
            fill_dict(calc, 'single_particle', parent_pdg, grandparent_pdg, \
                      reco_ke, true_ke, temp_tof, \
                      temp_dis, proton_length, parent_length)
            if len(reference_track)>1:
                print('ERROR! more than one reference track found')
                print(reference_track)
                continue

            # MIP primary track
            if reference_track[0] not in [211,-211,13,-13]: continue
            mip_reference_proton_parent.append(parent_pdg)
            fill_dict(calc, 'MIP_track', parent_pdg, grandparent_pdg, \
                      reco_ke, true_ke, temp_tof, \
                      temp_dis, proton_length, parent_length)

            if parent_pdg==2112 and grandparent_pdg==0:
                mip_reference_length['initscat'].append(reference_length[0])
            elif parent_pdg==2112 and grandparent_pdg==2112:
                mip_reference_length['rescat'].append(reference_length[0])
            elif parent_pdg in [12,14,16]:
                mip_reference_length['nu'].append(reference_length[0])
            else:
                mip_reference_length['other'].append(reference_length[0])

            # MINERvA matched MIP track
            if reference_length[0]<1000.: continue
            fill_dict(calc, 'MINERvA_track', parent_pdg, grandparent_pdg, \
                      reco_ke, true_ke, temp_tof, \
                      temp_dis, proton_length, parent_length)

            residence = location(tpc_vertex(d[k]['p_vtx']), tpc_vertex(d[k]['nu_vtx']))
            location_tof[residence].append(temp_tof)

            if parent_pdg==2112 and grandparent_pdg==0:
                selected_proton_track_length['initscat'].append(proton_length)
                selected_neutron_true_ke['initscat'].append(true_ke)
            elif parent_pdg==2112 and grandparent_pdg==2112:
                selected_proton_track_length['rescat'].append(proton_length)
                selected_neutron_true_ke['rescat'].append(true_ke)
            elif parent_pdg in [12,14,16]:
                selected_proton_track_length['nu'].append(proton_length)
                selected_neutron_true_ke['nu'].append(true_ke)
            else:
                selected_proton_track_length['other'].append(proton_length)
                selected_neutron_true_ke['other'].append(true_ke)

    piechart_single_vis_particle_at_vertex(primary_single_track)
    piechart_mip_reference_proton_parent(mip_reference_proton_parent)
    mip_reference_length_dist(mip_reference_length, file_ctr)
    sf = cut_variation(calc, file_ctr)
    sample_fraction(sf)
    sample_event_count(sf)
    analysis_selection_tof(location_tof, file_ctr)
    analysis_selection_proton_length(selected_proton_track_length, file_ctr)
    analysis_selection_neutron_true_ke(selected_neutron_true_ke, file_ctr)
            


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', \
                        default='/home/russell/DUNE/2x2/neutron_xs_2x2/data/n_tof_iv/', \
                        type=str, help='''File(s) directory''')
    args = parser.parse_args()
    main(**vars(args))




