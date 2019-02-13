#!/usr/bin/env python

import os
import sys
from math import sqrt
import math
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP

from align import *


class proteinINFO(object):
    """
    Class for get protein structure information from pdb, along with fasta file(always sequence 
        from pdb was inconsistent with one from fasta)
    Input: <1> pdb file; <2> fasta file
    You can get:
    <1> sequence alignment between pdb sequence and fasta sequence, self.alignment
    <2> full length of 3-state SSE sequence, self.ss3seq
    <3> full length of distance matrix, self.dist_matrix
    <3> full lengrh of angle matrix, self.angle_matrix
    """
    def __init__(self, name, pdbfile, fastafile):
        names = {'HIS':'H','ASP':'D','ARG':'R','PHE':'F','ALA':'A','CYS':'C','GLY':'G',\
                 'GLN':'Q','GLU':'E','LYS':'K','LEU':'L','MET':'M','ASN':'N','SER':'S',\
                 'TYR':'Y','THR':'T','ILE':'I','TRP':'W','PRO':'P','VAL':'V','SER':'S'}
        # Load fasta residue sequence
        f = open(fastafile)
        ff = [line.rstrip("\n") for line in f]
        f.close()
        p_id = name
        self.seq = ff[1]

        # Load pdb information
        p = PDBParser(PERMISSIVE=1)                                             
        st = p.get_structure(p_id, pdbfile)
        model = st[0]                                                            
        tag = p_id[-1] 
        chain = model[tag] 
        residues = chain.get_residues()
        self.residues = [res for res in residues] 
        ## sequence info from pdb
        self.pdbseq  = "".join([names[res.get_resname()] for res in self.residues if \
                names.has_key(res.get_resname())])
        ## 3-state sse info from pdb
        dssp = DSSP(model, pdbfile)
        to3_dict = {'-':'C', 'G':'H', 'H':'H', 'I':'H', 'E':'E', 'B':'E', 'T':'C', \
                'S':'C', 'L':'C'}
        keys = list(dssp.keys())
        self.pdbss3seq = "".join([to3_dict[dssp[k][2]]for k in keys])
        tmpseq = "".join([dssp[k][1] for k in keys])

        if len(tmpseq) != len(self.pdbseq):
            if self.pdbseq[1:] == tmpseq:
                self.pdbss3seq = 'C' + self.pdbss3seq
            elif self.pdbseq[:-1] == tmpseq:
                self.pdbss3seq += 'C'
            else:
                print ("""Warning: %s sequence lenght doesn't equal to ss3 length\n"""
                      """pdb sequence: %s\n"""
                      """ss3 sequence: %s""" %(name, self.pdbseq, tmpseq))

        # Align the pdb sequence(always missing some residues) to fasta sequence
        alignment = AlignNW(self.seq, self.pdbseq)
        self.re_index = alignment['j']
        self.re_index = [i-1 for i in alignment['j']] # minus 1 for indexing

        # generate sequence alignment between pdb sequence and fasta sequence
        self.alignment = "".join([self.pdbseq[i] if i > -1 else "-" for i in self.re_index])
        self.alignment = "\n".join([self.seq, self.alignment])

        # generate full lenght of 3-state SSE sequence according to re-index
        self.ss3seq = "".join([self.pdbss3seq[i] if i > -1 else "C" for i in self.re_index])
        
        # generate full lenght of distance matrix(distance=-1 when disappear in pdbseq)
        self.dist_matrix = self.generate_dist_matrix()
        #np.savetxt("test.txt", self.dist_matrix)
        # generate full lenght of angle matrix(distance=None when disappear in pdbseq)
        self.angle_matrix = self.generate_angle_matrix()

    def generate_dist_matrix(self):
        seqLen = len(self.seq)
        dist_matrix = np.zeros((seqLen, seqLen))
        for i in range(seqLen):
            for j in range(seqLen):
                dist_matrix[i][j] = self.get_distance_beta(i, j)
        return dist_matrix

    def generate_angle_matrix(self):
        seqLen = len(self.seq)
        angle_matrix = np.zeros((seqLen, seqLen))
        for i in range(seqLen):
            for j in range(seqLen):
                angle_matrix[i][j] = self.angle(i, j)
        return angle_matrix

    def get_distance_beta(self, i, j):
        if self.re_index[i] == -1 or self.re_index[j] == -1:
            return -1
        res1 = self.residues[self.re_index[i]]
        res2 = self.residues[self.re_index[j]]
        s1 = 'CB'
        s2 = 'CB'
        if res1.get_resname() == 'GLY':
            s1 = 'CA'
        if res2.get_resname() == 'GLY':
            s2 = 'CA'
        try:
            return res1[s1] - res2[s2]
        except:
            return -1

    def angle(self, i, j):
        if self.re_index[i] == -1 or self.re_index[j] == -1:
            return None
        res1 = self.residues[self.re_index[i]]
        res2 = self.residues[self.re_index[j]]
        if res1.get_resname() == 'GLY' or res2.get_resname() == 'GLY':
            return 0
        CA1 = res1['CA'].get_vector()
        CA2 = res2['CA'].get_vector()
        try:
            CB1 = res1['CB'].get_vector()
            CB2 = res2['CB'].get_vector()
        except:
            return None
        v1 = CB1 - CA1
        v2 = CB2 - CA2
        COS = v1 * v2 / sqrt((v1*v1)*(v2*v2))
        return COS

