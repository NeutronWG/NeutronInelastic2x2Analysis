"""
Script for running analysis on MiniRun4 truth level
info.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import json
import os
import sys
sys.path.append("src/")

from minirun_4 import *

def main(file_dir):

    if not os.path.isdir('selection_data/'):
        os.makedirs('selection_data/')
    if not os.path.isdir('plots/'):
        os.makedirs('plots/')

    for filename in glob.glob(file_dir+'/*.LARNDSIM.h5'):
        file_no = filename.split(".")[-3]
        sim_h5=h5py.File(filename,'r')
        d = process_file(sim_h5)
        save_dict_to_json(d, "selection_data/n_tof_"+file_no, True)

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

    for filename in glob.glob('selection_data/*.json'):
        print(filename)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', \
                        default='', \
                        type=str, help='''File(s) directory''')
    args = parser.parse_args()

    main(**vars(args))