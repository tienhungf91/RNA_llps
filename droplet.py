#!/usr/bin/env python

from simtk.openmm import app
import simtk.openmm as omm
from simtk import unit
from numpy import diag
import itertools as it
import time
import sys
import argparse
from math import sqrt, acos, atan2, ceil
import re

def prev_and_next(iterable):
    prevs, items, nexts = it.tee(iterable, 3)
    prevs = it.chain([None], prevs)
    nexts = it.chain(it.islice(nexts, 1, None), [None])
    return it.izip(prevs, items, nexts)

def atm_index(res):
    #return res.atoms[0].index
    for atom in res.atoms():
        return atom.index


def build_by_seq(seq, number, box_size, forcefield):
    bl = 3.41*unit.angstrom
    name_map = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'U': 'URA'}

    ite = int(number**(1./3))
    print ite
    distance = box_size / ite
    num_add = ite**3
    topo = app.Topology()
    positions = []
    def get_atom(res):
        for atom in res.atoms():
            return atom

    Nrepeat = 1
    #### process sequence

    if seq.find('(') > -1 and seq.find(')') > -1 and seq.split(")", 1)[1].isdigit():
        Nrepeat = int(seq.split(")", 1)[1])
        newseq = re.search('\((.*)\)', seq)
        seq = newseq.group(1)
        #print seq, Nrepeat

    for xshift in range(ite):
        for yshift in range(ite):
            for zshift in range(ite):
                chain = topo.addChain()
                atoms = []
                curr_idx = -1
                for it in range(Nrepeat):
                    for i, resSymbol in enumerate(seq):
                        symbol = name_map[resSymbol]
                        if (it == 0 and i == 0) or (it == Nrepeat - 1 and i == len(seq) - 1):
                            symbol = symbol + "T"

                        res = topo.addResidue(symbol, chain)
                        atom = forcefield._templates[symbol].atoms[0]
                        atoms.append(topo.addAtom(atom.name, forcefield._atomTypes[atom.type].element, res))
                        curr_idx += 1
                        positions.append([curr_idx*bl + xshift*distance, curr_idx*bl + yshift*distance, curr_idx*bl + zshift*distance])

                for prev, item, nxt in prev_and_next(chain.residues()):
                    if prev != None:
                        topo.addBond(get_atom(prev), get_atom(item))

    return topo, positions

###################################################################
def AllAtom2CoarseGrain(pdb, forcefield):
    name_map = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'U': 'URA'}
    cg_topo = app.Topology()
    chain = cg_topo.addChain('X')
    beads = []
    cg_positions = []
    bead_indx = -1
    prevbead = None

    for i, aa_res in enumerate(pdb.topology.residues()):
        resname = name_map[aa_res.name]
        if i == 0 or i == pdb.topology.getNumResidues() - 1:
            resname += "T"

        #print "Res %s %s" % (resname, aa_res.id)
        cg_res = cg_topo.addResidue(resname, chain, aa_res.id)

        count = 0
        bead_x = 0.*unit.angstrom
        bead_y = 0.*unit.angstrom
        bead_z = 0.*unit.angstrom

        for aa_atom in aa_res.atoms():
            #print aa_atom.name
            if "'" in aa_atom.name and (not "H" in aa_atom.name):
                count += 1
                bead_x += pdb.positions[aa_atom.index][0]
                bead_y += pdb.positions[aa_atom.index][1]
                bead_z += pdb.positions[aa_atom.index][2]
            else:
                continue

        bead_x /= count
        bead_y /= count
        bead_z /= count

        bead_indx += 1
        element = None
        beadname = None

        if "ADE" in cg_res.name:
            element = forcefield._atomTypes["0"].element
            beadname = "A"
        elif "GUA" in cg_res.name:
            element = forcefield._atomTypes["1"].element
            beadname = "G"
        elif "CYT" in cg_res.name:
            element = forcefield._atomTypes["2"].element
            beadname = "C"
        elif "URA" in cg_res.name:
            element = forcefield._atomTypes["3"].element
            beadname = "U"

        beads.append(cg_topo.addAtom(beadname, element, cg_res))
        cg_positions.append([bead_x, bead_y, bead_z])

        if prevbead != None:
            cg_topo.addBond(beads[bead_indx], beads[prevbead])

        prevbead = bead_indx

    return cg_topo, cg_positions

KELVIN_TO_KT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB / unit.kilocalorie_per_mole
#print KELVIN_TO_KT

parser = argparse.ArgumentParser(description='Coarse-grained simulation using OpenMM')
parser.add_argument('-p','--pdb', type=str, help='pdb structure')
parser.add_argument('-f','--sequence', type=str, help='input structure')
parser.add_argument('-C','--RNA_conc', type=float, default='10.',
                    help='RNA concentration (microM) [10.0]')
parser.add_argument('-K','--monovalent_concentration', type=float, default='100.',
                    help='Monovalent concentration (mM) [100.0]')
#parser.add_argument('-v','--box_size', type=float, default='80.',
#                    help='Box length (A) [80.0]')
parser.add_argument('-c','--cutoff', type=float, default='30.',
                    help='Electrostatic cutoff (A) [30.0]')
parser.add_argument('-H','--hbond_energy', type=float, default='2.22',
                    help='Hbond strength (kcal/mol) [2.22]')
parser.add_argument('-b','--hbond_file', type=str,
                    help='file storing tertiary Hbond')
parser.add_argument('-T','--temperature', type=float, default='20.',
                    help='Temperature (oC) [20.0]')
parser.add_argument('-t','--traj', type=str, default='md.dcd',
                    help='trajectory output')
parser.add_argument('-e','--energy', type=str, default='energy.out',
                    help='energy decomposition')
parser.add_argument('-o','--output', type=str, default='md.out',
                    help='status and energy output')
parser.add_argument('-x','--frequency', type=int, default='10000',
                    help='output and restart frequency')
parser.add_argument('-n','--step', type=long, default='10000',
                    help='Number of step [10000]')
parser.add_argument('-R','--restart', action='store_true',
                    help='flag to restart simulation')
parser.add_argument('-k','--chkpoint', type=str, default='checkpoint.xml',
                    help='initial xml state')
parser.add_argument('-r','--res_file', type=str, default='checkpnt.chk',
                    help='checkpoint file for restart')
args = parser.parse_args()

class simu:    ### structure to group all simulation parameter
    box = 0.
    temp = 0.
    Kconc = 0.
    Nstep = 0
    cutoff = 0.
    epsilon = 0.
    b = 4.38178046 * unit.angstrom / unit.elementary_charge
    restart = False
#    list = None ### list cannot be initialized here!!

#simu.list = []
#simu.box = args.box_size * unit.angstrom
Hbond_Uhb = args.hbond_energy*unit.kilocalorie_per_mole
simu.temp = (args.temperature + 273.15)*unit.kelvin
simu.Nstep = args.step
simu.cutoff = args.cutoff*unit.angstrom
simu.Kconc = args.monovalent_concentration
simu.restart = args.restart

T_unitless = simu.temp * KELVIN_TO_KT
simu.epsilon = 296.0736276 - 619.2813716 * T_unitless + 531.2826741 * T_unitless**2 - 180.0369914 * T_unitless**3;
#simu.l_Bjerrum = 1./(simu.epsilon * unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * simu.temp)
simu.l_Bjerrum = 332.0637*unit.angstroms / simu.epsilon
print "Bjerrum length  ", simu.l_Bjerrum / T_unitless
simu.Q = simu.b * T_unitless * unit.elementary_charge**2 / simu.l_Bjerrum
print "Phosphate charge   ", -simu.Q
simu.kappa = unit.sqrt (4*3.14159 * simu.l_Bjerrum * 2*simu.Kconc*6.022e-7 / (T_unitless * unit.angstrom**3))
print "kappa   ", simu.kappa

forcefield = app.ForceField('rna_cg1.xml')
topology = None
positions = None

if args.pdb != None:
    print "Reading PDB file ..."
    pdb = app.PDBFile(args.pdb)
    topology, positions = AllAtom2CoarseGrain(pdb, forcefield)
elif args.sequence != None:
    print "Building from sequence %s ..." % args.sequence
    #N_RNA = args.RNA_conc * 6.022e-10 * args.box_size**3
    #N_RNA_added = (int(N_RNA**(1./3)))**3
    #N_RNA_added = 27
    N_RNA_added = 64
    #real_conc = N_RNA_added / (6.022e-10 * args.box_size**3)
    real_conc = args.RNA_conc
    simu.box = (N_RNA_added / (real_conc * 6.022e-10))**(1./3) * unit.angstrom
    print "Box size    %f A" % (simu.box/unit.angstrom)
    print "Numbers added   %d ----> %f microM" % (N_RNA_added, real_conc)
    topology, positions = build_by_seq(args.sequence, N_RNA_added, simu.box, forcefield)
else:
    print "Need at least structure or sequence !!!"
    sys.exit()

topology.setPeriodicBoxVectors([[simu.box.value_in_unit(unit.nanometers),0,0], [0,simu.box.value_in_unit(unit.nanometers),0], [0,0,simu.box.value_in_unit(unit.nanometers)]])

system = forcefield.createSystem(topology)

########## bond force
bondforce = omm.HarmonicBondForce()
for bond in topology.bonds():
    bondforce.addBond(bond[0].index, bond[1].index, 5.9*unit.angstroms, 15.0*unit.kilocalorie_per_mole/(unit.angstrom**2))

bondforce.setUsesPeriodicBoundaryConditions(True)
bondforce.setForceGroup(0)
system.addForce(bondforce)

######### angle force
angleforce = omm.HarmonicAngleForce()

for chain in topology.chains():
    for prev, item, nxt in prev_and_next(chain.residues()):
        if prev == None or nxt == None:
            continue

        angleforce.addAngle(prev.index, item.index, nxt.index, 2.618*unit.radian, 10.0*unit.kilocalorie_per_mole/(unit.radians**2))

angleforce.setUsesPeriodicBoundaryConditions(True)
angleforce.setForceGroup(1)
system.addForce(angleforce)

######## WCA force
WCA_cutoff = 10.*unit.angstroms
energy_function =  'step(sig-r) * ep * ((R6 - 2)*R6 + 1);'
energy_function += 'R6=(sig/r)^6;'

WCAforce = omm.CustomNonbondedForce(energy_function)
WCAforce.addGlobalParameter('ep',  2.*unit.kilocalorie_per_mole)
WCAforce.addGlobalParameter('sig', WCA_cutoff)

for atom in topology.atoms():
    WCAforce.addParticle([])

for bond in topology.bonds():
    WCAforce.addExclusion(bond[0].index, bond[1].index)

for chain in topology.chains():
    for prev, item, nxt in prev_and_next(chain.residues()):
        if prev == None or nxt == None:
            continue
        WCAforce.addExclusion(atm_index(prev), atm_index(nxt))

WCAforce.setCutoffDistance(WCA_cutoff)
WCAforce.setForceGroup(2)
WCAforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffPeriodic)
system.addForce(WCAforce)

######## Debye-Huckel
#DHforce = omm.CustomNonbondedForce("scale*exp(-kappa*r)/r")
#DHforce.addGlobalParameter("scale", simu.l_Bjerrum * simu.Q**2 * unit.kilocalorie_per_mole / unit.elementary_charge**2)
#DHforce.addGlobalParameter("kappa", simu.kappa)
#
#for atom in topology.atoms():
#    DHforce.addParticle([])
#
#for bond in topology.bonds():
#    DHforce.addExclusion(bond[0].index, bond[1].index)
#
#for chain in topology.chains():
#    for prev, item, nxt in prev_and_next(chain.residues()):
#        if prev == None or nxt == None:
#            continue
#        DHforce.addExclusion(atm_index(prev), atm_index(nxt))
#
#DHforce.setCutoffDistance(simu.cutoff)
#DHforce.setForceGroup(3)
#DHforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffPeriodic)
#system.addForce(DHforce)

###### Hbond
energy_function =  "- kr*(distance(a1, d1) - r0)^2"
energy_function += "- kt*(angle(a1, d1, d2) - theta1)^2"
energy_function += "- kt*(angle(d1, a1, a2) - theta1)^2"
energy_function += "- kt*(angle(a1, d1, d3) - theta2)^2"
energy_function += "- kt*(angle(d1, a1, a3) - theta2)^2"
energy_function += "- kp*(1. + cos(dihedral(d2, d1, a1, a2) + phi1))"
energy_function += "- kp*(1. + cos(dihedral(d3, d1, a1, a3) + phi2))"

energy_functionAU = "2 * Uhb * exp(" + energy_function + ")"

bondlength = 1.38*unit.nanometers
kr = 3./unit.angstrom**2
kt = 1.5/unit.radian**2
kp = 0.5
theta1 = 1.8326*unit.radians
theta2 = 0.9425*unit.radians
phi1 = 1.8326*unit.radians
phi2 = 1.1345*unit.radians
cutoff = 1.8*unit.nanometers

HbAUforce = omm.CustomHbondForce(energy_functionAU)
HbAUforce.addGlobalParameter('Uhb', -Hbond_Uhb)
HbAUforce.addGlobalParameter('r0', bondlength)
HbAUforce.addGlobalParameter('kr', kr)
HbAUforce.addGlobalParameter('kt', kt)
HbAUforce.addGlobalParameter('kp', kp)
HbAUforce.addGlobalParameter('theta1', theta1)
HbAUforce.addGlobalParameter('theta2', theta2)
HbAUforce.addGlobalParameter('phi1', phi1)
HbAUforce.addGlobalParameter('phi2', phi2)
HbAUforce.setCutoffDistance(cutoff)
HbAUforce.setNonbondedMethod(omm.CustomHbondForce.CutoffPeriodic)
HbAUforce.setForceGroup(3)
#HbAUforce.usesPeriodicBoundaryConditions()

energy_functionGC = "3 * Uhb * exp(" + energy_function + ")"

HbGCforce = omm.CustomHbondForce(energy_functionGC)
HbGCforce.addGlobalParameter('Uhb', -Hbond_Uhb)
HbGCforce.addGlobalParameter('r0', bondlength)
HbGCforce.addGlobalParameter('kr', kr)
HbGCforce.addGlobalParameter('kt', kt)
HbGCforce.addGlobalParameter('kp', kp)
HbGCforce.addGlobalParameter('theta1', theta1)
HbGCforce.addGlobalParameter('theta2', theta2)
HbGCforce.addGlobalParameter('phi1', phi1)
HbGCforce.addGlobalParameter('phi2', phi2)
HbGCforce.setCutoffDistance(cutoff)
HbGCforce.setNonbondedMethod(omm.CustomHbondForce.CutoffPeriodic)
HbGCforce.setForceGroup(4)
#HbGCforce.usesPeriodicBoundaryConditions()

HbGUforce = omm.CustomHbondForce(energy_functionAU)
HbGUforce.addGlobalParameter('Uhb', -Hbond_Uhb)
HbGUforce.addGlobalParameter('r0', bondlength)
HbGUforce.addGlobalParameter('kr', kr)
HbGUforce.addGlobalParameter('kt', kt)
HbGUforce.addGlobalParameter('kp', kp)
HbGUforce.addGlobalParameter('theta1', theta1)
HbGUforce.addGlobalParameter('theta2', theta2)
HbGUforce.addGlobalParameter('phi1', phi1)
HbGUforce.addGlobalParameter('phi2', phi2)
HbGUforce.setCutoffDistance(cutoff)
HbGUforce.setNonbondedMethod(omm.CustomHbondForce.CutoffPeriodic)
HbGUforce.setForceGroup(5)
#HbGUforce.usesPeriodicBoundaryConditions()

totalforcegroup = 5

list_donorGC = []
list_acceptorGC = []
list_donorAU = []
list_acceptorAU = []
list_donorGU = []
list_acceptorGU = []

for chain in topology.chains():
    for prev, item, nxt in prev_and_next(chain.residues()):
        if prev == None or nxt == None:
            continue
        if "ADE" in item.name:
            HbAUforce.addDonor(atm_index(item), atm_index(prev), atm_index(nxt), [])
            list_donorAU.append(item.index)
        elif "GUA" in item.name:
            HbGCforce.addDonor(atm_index(item), atm_index(prev), atm_index(nxt), [])
            HbGUforce.addDonor(atm_index(item), atm_index(prev), atm_index(nxt), [])
            list_donorGC.append(item.index)
            list_donorGU.append(item.index)
        elif "CYT" in item.name:
            HbGCforce.addAcceptor(atm_index(item), atm_index(prev), atm_index(nxt), [])
            list_acceptorGC.append(item.index)
        elif "URA" in item.name:
            HbAUforce.addAcceptor(atm_index(item), atm_index(prev), atm_index(nxt), [])
            HbGUforce.addAcceptor(atm_index(item), atm_index(prev), atm_index(nxt), [])
            list_acceptorAU.append(item.index)
            list_acceptorGU.append(item.index)

#def check_same_chain(id1, id2, topology):
#    check1 = 0
#    check2 = 0
#    for chain in topology.chains():
#        for res in chain.residues():
#            if res.index == id1:
#                check1 = 1
#            if res.index == id2:
#                check2 = 1
#        if (check1 or check2):
#            break
#    return (check1 and check2)

same_chain_list = []
for chain in topology.chains():
    for res1 in chain.residues():
        connect_list = []
        for res2 in chain.residues():
            if res1.index == res2.index:
                continue
            connect_list.append(res2.index)
        same_chain_list.append(connect_list)

#print same_chain_list

print ("Initializing HB bonds:")
if (HbAUforce.getNumDonors() > 0 and HbAUforce.getNumAcceptors() > 0):
    for ind1, res1 in enumerate(list_donorAU):
        for ind2, res2 in enumerate(list_acceptorAU):
            if res2 in same_chain_list[res1] and abs(res1-res2) < 5:
                HbAUforce.addExclusion(ind1, ind2)
    print ("   A-U:  %d A,   %d U" % (HbAUforce.getNumDonors(), HbAUforce.getNumAcceptors()))
    print ("         %d exclusion" % HbAUforce.getNumExclusions())
    system.addForce(HbAUforce)

if (HbGCforce.getNumDonors() > 0 and HbGCforce.getNumAcceptors() > 0):
    for ind1, res1 in enumerate(list_donorGC):
        for ind2, res2 in enumerate(list_acceptorGC):
            if res2 in same_chain_list[res1] and abs(res1-res2) < 5:
                HbGCforce.addExclusion(ind1, ind2)
    print ("   G-C:  %d G,   %d C" % (HbGCforce.getNumDonors(), HbGCforce.getNumAcceptors()))
    print ("         %d exclusion" % HbGCforce.getNumExclusions())
    system.addForce(HbGCforce)

if (HbGUforce.getNumDonors() > 0 and HbGUforce.getNumAcceptors() > 0):
    for ind1, res1 in enumerate(list_donorGU):
        for ind2, res2 in enumerate(list_acceptorGU):
            if res2 in same_chain_list[res1] and abs(res1-res2) < 5:
                HbGUforce.addExclusion(ind1, ind2)
    print ("   G-U:  %d G,   %d U" % (HbGUforce.getNumDonors(), HbGUforce.getNumAcceptors()))
    print ("         %d exclusion" % HbGUforce.getNumExclusions())
    system.addForce(HbGUforce)

########## Tertiary Hbond
#if args.pdb != None and args.hbond_file != None:
#    def get_res_from_index(idx, topology):
#        for res in topology.residues():
#            if res.id == idx:
#                return res
#
#    def compute_angle(i, j, k, positions):
#        x1 = positions[i][0] - positions[j][0]
#        y1 = positions[i][1] - positions[j][1]
#        z1 = positions[i][2] - positions[j][2]
#
#        x2 = positions[k][0] - positions[j][0]
#        y2 = positions[k][1] - positions[j][1]
#        z2 = positions[k][2] - positions[j][2]
#
#        norm1 = x1**2 + y1**2 + z1**2
#        norm2 = x2**2 + y2**2 + z2**2
#
#        norm12 = (norm1*norm2).sqrt()
#        return acos((x1*x2 + y1*y2 + z1*z2) / norm12)
#
#    def compute_dihedral(i, j, k, l, positions):
#        x1 = positions[j][0] - positions[i][0]
#        y1 = positions[j][1] - positions[i][1]
#        z1 = positions[j][2] - positions[i][2]
#
#        x2 = positions[k][0] - positions[j][0]
#        y2 = positions[k][1] - positions[j][1]
#        z2 = positions[k][2] - positions[j][2]
#
#        x3 = positions[l][0] - positions[k][0]
#        y3 = positions[l][1] - positions[k][1]
#        z3 = positions[l][2] - positions[k][2]
#
#        def cross_product(a, b, c, x, y, z):
#            m = b*z - c*y
#            n = c*x - a*z
#            o = a*y - b*x
#            return m, n, o
#
#        a, b, c = cross_product(x1, y1, z1, x2, y2, z2)
#        x, y, z = cross_product(x2, y2, z2, x3, y3, z3)
#
#        cosine = (a*x + b*y + c*z) / unit.angstrom**4
#        sine = (x1*x + y1*y + z1*z) * (x2**2 + y2**2 + z2**2).sqrt() / unit.angstrom**4
#        return atan2 (sine, cosine)
#
#
#    energy_function =  "- kr*(distance(p1, p4) - r0)^2"
#    energy_function += "- kt*(angle(p1, p4, p5) - theta1)^2"
#    energy_function += "- kt*(angle(p4, p1, p2) - theta2)^2"
#    energy_function += "- kt*(angle(p1, p4, p6) - theta3)^2"
#    energy_function += "- kt*(angle(p4, p1, p3) - theta4)^2"
#    energy_function += "- kp*(1. + cos(dihedral(p5, p4, p1, p2) + phi1))"
#    energy_function += "- kp*(1. + cos(dihedral(p6, p4, p1, p3) + phi2))"
#    energy_function = "Nbond * Uhb * exp(" + energy_function + ")"
#
#    tertHbforce = omm.CustomCompoundBondForce(6, energy_function)
#    tertHbforce.addGlobalParameter('Uhb', -Hbond_Uhb)
#    tertHbforce.addGlobalParameter('kr', 3./unit.angstrom**2)
#    tertHbforce.addGlobalParameter('kt', 1.5/unit.radian**2)
#    tertHbforce.addGlobalParameter('kp', 0.5)
#    tertHbforce.addPerBondParameter('r0')
#    tertHbforce.addPerBondParameter('theta1')
#    tertHbforce.addPerBondParameter('theta2')
#    tertHbforce.addPerBondParameter('theta3')
#    tertHbforce.addPerBondParameter('theta4')
#    tertHbforce.addPerBondParameter('phi1')
#    tertHbforce.addPerBondParameter('phi2')
#    tertHbforce.addPerBondParameter('Nbond')
#
#    with open(args.hbond_file) as f:
#        print "Adding tertiary Hbonds ..."
#        for line in f:
#            columns = line.split()
#            res1 = get_res_from_index(columns[0], topology)
#            res2 = get_res_from_index(columns[1], topology)
#            print "Adding bond between %d - %d" %(res1.index, res2.index)
#
#            a1 = res1.index
#            a2 = res1.index + 1
#            a3 = res1.index - 1
#
#            d1 = res2.index
#            d2 = res2.index + 1
#            d3 = res2.index - 1
#
#            dx = positions[a1][0] - positions[d1][0]
#            dy = positions[a1][1] - positions[d1][1]
#            dz = positions[a1][2] - positions[d1][2]
#            r0 = (dx**2 + dy**2 + dz**2).sqrt()
#
#            theta1 = compute_angle(a1, d1, d2, positions)
#            theta2 = compute_angle(d1, a1, a2, positions)
#            theta3 = compute_angle(a1, d1, d3, positions)
#            theta4 = compute_angle(d1, a1, a3, positions)
#
#            phi1 = 3.1415926 - compute_dihedral(d2, d1, a1, a2, positions)
#            phi2 = 3.1415926 - compute_dihedral(d3, d1, a1, a3, positions)
#
#            #print acceptor2, acceptor1, donor1, donor2, r0, theta1*180/3.1415926, theta2*180/3.1415926, 180. - phi*180/3.1415926
#            #print positions[acceptor1][0], positions[acceptor1][1], positions[acceptor1][2]
#            #print positions[donor1][0], positions[donor1][1], positions[donor1][2]
#
#            tertHbforce.addBond([a1, a2, a3, d1, d2, d3], [r0, theta1*unit.radian, theta2*unit.radian, theta3*unit.radian, theta4*unit.radian, phi1*unit.radian, phi2*unit.radian, int(columns[2])])
#
#        tertHbforce.setUsesPeriodicBoundaryConditions(True)
#        totalforcegroup += 1
#        tertHbforce.setForceGroup(totalforcegroup)
#        system.addForce(tertHbforce)

########## Simulation ############
class EnergyReporter(object):
    def __init__ (self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__ (self):
        self._out.close()

    def describeNextReport(self, simulation):
        step = self._reportInterval - simulation.currentStep%self._reportInterval
        return (step, False, False, False, True)
        #return (step, position, velocity, force, energy)

    def report(self, simulation, state):
        energy = []
        self._out.write(str(simulation.currentStep))
        for i in range(totalforcegroup + 1):
            state = simulation.context.getState(getEnergy=True, groups=2**i)
            energy = state.getPotentialEnergy() / unit.kilocalorie_per_mole
            self._out.write("  " + str(energy))
        self._out.write("\n")

integrator = omm.LangevinIntegrator(simu.temp, 0.5/unit.picosecond, 50*unit.femtoseconds)
#platform = omm.Platform.getPlatformByName('CUDA')
#properties = {'CudaPrecision': 'mixed'}

#simulation = app.Simulation(topology, system, integrator, platform)
#simulation = app.Simulation(topology, system, integrator, platform, properties)
simulation = app.Simulation(topology, system, integrator)

if simu.restart == False:
    #simulation.context.setPositions(positions)
    simulation.loadState(args.chkpoint)
    boxvector = diag([simu.box/unit.angstrom for i in range(3)]) * unit.angstrom
    simulation.context.setPeriodicBoxVectors(*boxvector)
    print simulation.usesPeriodicBoundaryConditions()

    positions = simulation.context.getState(getPositions=True).getPositions()
    newpost = []
    for pos in positions:
        #pos[0] = pos[0] * simu.box / (150*unit.nanometers)
        #pos[1] = pos[1] * simu.box / (150*unit.nanometers)
        #pos[2] = pos[2] * simu.box / (150*unit.nanometers)

        newpost.append([pos[0]*simu.box/(150*unit.nanometers), pos[1]*simu.box/(150*unit.nanometers), pos[2]*simu.box/(150*unit.nanometers)])

    #simulation.context.setPositions(positions)
    simulation.context.setPositions(newpost)
    #print "Initial energy   %f   kcal/mol" % (simulation.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalorie_per_mole)

    print('Minimizing ...')
    simulation.minimizeEnergy(1*unit.kilocalorie_per_mole, 10000)

    #state = simulation.context.getState(getPositions=True)
    #app.PDBFile.writeFile(topology, state.getPositions(), open("input.pdb", "w"), keepIds=True)

    simulation.context.setVelocitiesToTemperature(simu.temp)
else:
    print "Loading checkpoint ..."
    simulation.loadCheckpoint(args.res_file)

simulation.reporters.append(app.DCDReporter(args.traj, args.frequency))
simulation.reporters.append(app.StateDataReporter(args.output, args.frequency, step=True, potentialEnergy=True, temperature=True, remainingTime=True, totalSteps=simu.Nstep, separator='  '))
simulation.reporters.append(EnergyReporter(args.energy, args.frequency))
simulation.reporters.append(app.CheckpointReporter(args.res_file, long(args.frequency)*100))

print('Running ...')
t0 = time.time()
simulation.step(simu.Nstep)
#simulation.saveState('checkpoint.xml')
prodtime = time.time() - t0
print "Simulation speed: % .2e steps/day" % (86400*simu.Nstep/(prodtime))
