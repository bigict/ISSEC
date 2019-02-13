# -*- coding: utf-8 -*-
# Author: Jianwei Zhu




#############################################################################
#  Matrix made by matblas from blosum62.iij
#  BLOSUM Clustered Scoring Matrix in 1/2 Bit Units
#  Entries for the BLOSUM62 matrix at a scale of ln(2)/2.0
#  Cluster Percentage: >= 62
#  Entropy =   0.6979, Expected =  -0.5209
#             A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *
BLOSUM62 = [[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1,-1,-1,-4], # A
            [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1,-2, 0,-1,-4], # R
            [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 4,-3, 0,-1,-4], # N
            [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4,-3, 1,-1,-4], # D
            [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-1,-3,-1,-4], # C
            [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0,-2, 4,-1,-4], # Q
            [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1,-3, 4,-1,-4], # E
            [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-4,-2,-1,-4], # G
            [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0,-3, 0,-1,-4], # H
            [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3, 3,-3,-1,-4], # I
            [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4, 3,-3,-1,-4], # L
            [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0,-3, 1,-1,-4], # K
            [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3, 2,-1,-1,-4], # M
            [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3, 0,-3,-1,-4], # F
            [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-3,-1,-1,-4], # P
            [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0,-2, 0,-1,-4], # S
            [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1,-1,-1,-4], # T
            [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-2,-2,-1,-4], # W
            [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-1,-2,-1,-4], # Y
            [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3, 2,-2,-1,-4], # V
            [-2,-1, 4, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4,-3, 0,-1,-4], # B
            [-1,-2,-3,-3,-1,-2,-3,-4,-3, 3, 3,-3, 2, 0,-3,-2,-1,-2,-1, 2,-3, 3,-3,-1,-4], # J
            [-1, 0, 0, 1,-3, 4, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-2,-2,-2, 0,-3, 4,-1,-4], # Z
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-4], # X
            [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1]] # *
# Entries for the PAM250 matrix at a scale of ln(2)/3.0.
#           A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *
PAM250 = [[ 2,-2, 0, 0,-2, 0, 0, 1,-1,-1,-2,-1,-1,-3, 1, 1, 1,-6,-3, 0, 0,-1, 0,-1,-8], # A
          [-2, 6, 0,-1,-4, 1,-1,-3, 2,-2,-3, 3, 0,-4, 0, 0,-1, 2,-4,-2,-1,-3, 0,-1,-8], # R
          [ 0, 0, 2, 2,-4, 1, 1, 0, 2,-2,-3, 1,-2,-3, 0, 1, 0,-4,-2,-2, 2,-3, 1,-1,-8], # N
          [ 0,-1, 2, 4,-5, 2, 3, 1, 1,-2,-4, 0,-3,-6,-1, 0, 0,-7,-4,-2, 3,-3, 3,-1,-8], # D
          [-2,-4,-4,-5,12,-5,-5,-3,-3,-2,-6,-5,-5,-4,-3, 0,-2,-8, 0,-2,-4,-5,-5,-1,-8], # C
          [ 0, 1, 1, 2,-5, 4, 2,-1, 3,-2,-2, 1,-1,-5, 0,-1,-1,-5,-4,-2, 1,-2, 3,-1,-8], # Q
          [ 0,-1, 1, 3,-5, 2, 4, 0, 1,-2,-3, 0,-2,-5,-1, 0, 0,-7,-4,-2, 3,-3, 3,-1,-8], # E
          [ 1,-3, 0, 1,-3,-1, 0, 5,-2,-3,-4,-2,-3,-5, 0, 1, 0,-7,-5,-1, 0,-4, 0,-1,-8], # G
          [-1, 2, 2, 1,-3, 3, 1,-2, 6,-2,-2, 0,-2,-2, 0,-1,-1,-3, 0,-2, 1,-2, 2,-1,-8], # H
          [-1,-2,-2,-2,-2,-2,-2,-3,-2, 5, 2,-2, 2, 1,-2,-1, 0,-5,-1, 4,-2, 3,-2,-1,-8], # I
          [-2,-3,-3,-4,-6,-2,-3,-4,-2, 2, 6,-3, 4, 2,-3,-3,-2,-2,-1, 2,-3, 5,-3,-1,-8], # L
          [-1, 3, 1, 0,-5, 1, 0,-2, 0,-2,-3, 5, 0,-5,-1, 0, 0,-3,-4,-2, 1,-3, 0,-1,-8], # K
          [-1, 0,-2,-3,-5,-1,-2,-3,-2, 2, 4, 0, 6, 0,-2,-2,-1,-4,-2, 2,-2, 3,-2,-1,-8], # M
          [-3,-4,-3,-6,-4,-5,-5,-5,-2, 1, 2,-5, 0, 9,-5,-3,-3, 0, 7,-1,-4, 2,-5,-1,-8], # F
          [ 1, 0, 0,-1,-3, 0,-1, 0, 0,-2,-3,-1,-2,-5, 6, 1, 0,-6,-5,-1,-1,-2, 0,-1,-8], # P
          [ 1, 0, 1, 0, 0,-1, 0, 1,-1,-1,-3, 0,-2,-3, 1, 2, 1,-2,-3,-1, 0,-2, 0,-1,-8], # S
          [ 1,-1, 0, 0,-2,-1, 0, 0,-1, 0,-2, 0,-1,-3, 0, 1, 3,-5,-3, 0, 0,-1,-1,-1,-8], # T
          [-6, 2,-4,-7,-8,-5,-7,-7,-3,-5,-2,-3,-4, 0,-6,-2,-5,17, 0,-6,-5,-3,-6,-1,-8], # W
          [-3,-4,-2,-4, 0,-4,-4,-5, 0,-1,-1,-4,-2, 7,-5,-3,-3, 0,10,-2,-3,-1,-4,-1,-8], # Y
          [ 0,-2,-2,-2,-2,-2,-2,-1,-2, 4, 2,-2, 2,-1,-1,-1, 0,-6,-2, 4,-2, 2,-2,-1,-8], # V
          [ 0,-1, 2, 3,-4, 1, 3, 0, 1,-2,-3, 1,-2,-4,-1, 0, 0,-5,-3,-2, 3,-3, 2,-1,-8], # B
          [-1,-3,-3,-3,-5,-2,-3,-4,-2, 3, 5,-3, 3, 2,-2,-2,-1,-3,-1, 2,-3, 5,-2,-1,-8], # J
          [ 0, 0, 1, 3,-5, 3, 3, 0, 2,-2,-3, 0,-2,-5, 0, 0,-1,-6,-4,-2, 2,-2, 3,-1,-8], # Z
          [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-8], # X
          [-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8, 1]] # *
# mappping amino acid to index
# A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *
# 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
AAID = {'A': 0, 'B':20, 'C': 4, 'D': 3, 'E': 6, 'F':13, 'G':7,
        'H': 8, 'I': 9, 'J':21, 'K':11, 'L':10, 'M':12, 'N':2,
        'O':23, 'P':14, 'Q': 5, 'R': 1, 'S':15, 'T':16, 'U':4,
        'V':19, 'W':17, 'X':23, 'Y':18, 'Z':22}
#############################################################################




#############################################################################
# Subroutine AlignNW
# Needleman-Wunsch global alignment with affine gap penalty
# Usage: (score, align) = AlignNW(xseq, yseq)
# Input:  xseq, yseq        : str, the seqence x and y as strings
# Param:  d                 : int, gap opening penalty
#         e                 : int, gap extension penalty
#         r(gap)            : int, affine gap penalty, r(gap)=-d-(gap-1)e
#         Sab               : list, substitution matrix, default=BLOSUM62
# Output: return align      : dict, all alignment results
#         align['score']    : int, bit score
#         align['xseq']     : str, aligned residues of x with - as gap
#         align['yseq']     : str, aligned residues of y with - as gap
#         align['i']        : list, i[col], j[col] are aligned residues in column col
#         align['j']        :     first is 1 (NOT 0!), 0 means gap
#         align['imin']     : int, first aligned residue of sequence x
#         align['imax']     : int, last  aligned residue of sequence x
#         align['jmin']     : int, first aligned residue of sequence y
#         align['jmax']     : int, last  aligned residue of sequence y
# Algorithm: M, Ix, Iy      : three situations
#          Match (M)        Insert on reference (I_x)         Insert on query (I_y)
#           IGA x_i              AI   G   A  x_i                  GA  x_i  -   -
#           LGV y_j              GV  y_j  -   -                   SL   G   V  y_j
#   M(i, j) = max(M(i-1, j-1), I_x(i-1, j-1), I_y(i-1, j-1)) + s(x_i, y_j)
# I_x(i, j) = max(M(i-1, j)-d, I_x(i-1, j)-e)
# I_y(i, j) = max(M(i, j-1)-d, I_y(i, j-1)-e)
#############################################################################
def AlignNW(xseq, yseq, Sab=BLOSUM62, d=11, e=1, inf=999):
    # intialization
    n_row, n_col = len(xseq)+1, len(yseq)+1
    xmatchy = [[-inf for __ in range(n_col)] for _ in range(n_row)]
    xinsert = [[-inf for __ in range(n_col)] for _ in range(n_row)]
    yinsert = [[-inf for __ in range(n_col)] for _ in range(n_row)]
    xmatchy[0][0] = 0
    for i in range(1, n_row): xinsert[i][0] = -d - (i-1) * e
    for j in range(1, n_col): yinsert[0][j] = -d - (j-1) * e
    # iteration
    for i in range(1, n_row):
        for j in range(1, n_col):
            xmatchy[i][j] = max(xmatchy[i-1][j-1], xinsert[i-1][j-1], yinsert[i-1][j-1]) + Sab[AAID[xseq[i-1]]][AAID[yseq[j-1]]]
            xinsert[i][j] = max(xmatchy[i-1][j]-d, xinsert[i-1][j]-e)
            yinsert[i][j] = max(xmatchy[i][j-1]-d, yinsert[i][j-1]-e)
    # backtrack
    state, score = max(enumerate([xmatchy[-1][-1], xinsert[-1][-1], yinsert[-1][-1]]), key=lambda s: s[1])
    align = {'score' : score}
    r, c = n_row-1, n_col-1
    x, y, i, j = [], [], [], []
    while r > 0 or c > 0:
        if 0 == state and r > 0 and c > 0:
            s = xmatchy[r][c] - Sab[AAID[xseq[r-1]]][AAID[yseq[c-1]]]
            if s == xmatchy[r-1][c-1]: state = 0
            elif s == xinsert[r-1][c-1] : state = 1
            else : state = 2
            x.append(xseq[r-1]), y.append(yseq[c-1])
            i.append(r), j.append(c)
            r, c = r-1, c-1
        elif 1 == state and r > 0:
            if xinsert[r][c] == xmatchy[r-1][c] - d: state = 0
            else : state = 1
            x.append(xseq[r-1]), y.append('-')
            i.append(r), j.append(0)
            r, c = r-1, c
        elif 2 == state and c > 0:
            if yinsert[r][c] == xmatchy[r][c-1] - d: state = 0
            else : state = 2
            x.append('-'), y.append(yseq[c-1])
            i.append(0), j.append(c)
            r, c = r, c-1
    # get alignment results
    align['xseq'], align['yseq'] = ''.join([_ for _ in x[::-1]]), ''.join([_ for _ in y[::-1]])
    align['i'], align['j'] = i[::-1], j[::-1]
    align.update({'imin':0, 'imax':0, 'jmin':0, 'jmax':0})
    for i, j in zip(align['i'], align['j']):
        if i > 0 and j > 0:
            align['imin'], align['imax'] = align['imin'] if align['imin'] > 0 else i, i
            align['jmin'], align['jmax'] = align['jmin'] if align['jmin'] > 0 else j, j
    # return a dict which contain all alignment results
    return align




#############################################################################
# Subroutine AlignSW
# Smith-Waterman local alignment with affine gap penalty
# Usage: (score, align) = AlignSW(xseq, yseq)
# Input:  xseq, yseq        : str, the seqence x and y as strings
# Param:  d                 : int, gap opening penalty
#         e                 : int, gap extension penalty
#         r(gap)            : int, affine gap penalty, r(gap)=-d-(gap-1)e
#         Sab               : list, substitution matrix, default=BLOSUM62
# Output: return align      : dict, all alignment results
#         align['score']    : int, bit score
#         align['xseq']     : str, aligned residues of x with - as gap
#         align['yseq']     : str, aligned residues of y with - as gap
#         align['i']        : list, i[col], j[col] are aligned residues in column col
#         align['j']        :     first is 1 (NOT 0!), 0 means gap
#         align['imin']     : int, first aligned residue of sequence x
#         align['imax']     : int, last  aligned residue of sequence x
#         align['jmin']     : int, first aligned residue of sequence y
#         align['jmax']     : int, last  aligned residue of sequence y
# Algorithm: M, Ix, Iy      : three situations
#          Match (M)        Insert on reference (I_x)         Insert on query (I_y)
#           IGA x_i              AI   G   A  x_i                  GA  x_i  -   -
#           LGV y_j              GV  y_j  -   -                   SL   G   V  y_j
#   M(i, j) = max(    0, max(M(i-1, j-1), I_x(i-1, j-1), I_y(i-1, j-1)) + s(x_i, y_j))
# I_x(i, j) = max(    0, M(i-1, j) - d, I_x(i-1, j) - e)
# I_y(i, j) = max(    0, M(i, j-1) - d, I_y(i, j-1) - e)
#############################################################################
def AlignSW(xseq, yseq, Sab=BLOSUM62, d=11, e=1, inf=999):
    # intialization
    n_row, n_col = len(xseq)+1, len(yseq)+1
    xmatchy = [[0 for __ in range(n_col)] for _ in range(n_row)]
    xinsert = [[0 for __ in range(n_col)] for _ in range(n_row)]
    yinsert = [[0 for __ in range(n_col)] for _ in range(n_row)]
    # iteration
    for i in range(1, n_row):
        for j in range(1, n_col):
            xmatchy[i][j] = max(xmatchy[i-1][j-1], xinsert[i-1][j-1], yinsert[i-1][j-1]) + Sab[AAID[xseq[i-1]]][AAID[yseq[j-1]]]
            xmatchy[i][j] = max(0, xmatchy[i][j])
            xinsert[i][j] = max(0, xmatchy[i-1][j]-d, xinsert[i-1][j]-e)
            yinsert[i][j] = max(0, xmatchy[i][j-1]-d, yinsert[i][j-1]-e)
    # backtrack
    state, score, r, c = 0, 0, 0, 0
    for m in range(n_row):
        for n in range(n_col):
            sta, sco = max(enumerate([xmatchy[m][n], xinsert[m][n], yinsert[m][n]]), key=lambda s: s[1])
            if score < sco: state, score, r, c = sta, sco, m, n
    align = {'score' : score}
    x, y, i, j = [], [], [], []
    while score != 0:
        if 0 == state:
            score -= Sab[AAID[xseq[r-1]]][AAID[yseq[c-1]]]
            if score == xmatchy[r-1][c-1]: state = 0
            elif score == xinsert[r-1][c-1] : state = 1
            else : state = 2
            x.append(xseq[r-1]), y.append(yseq[c-1])
            i.append(r), j.append(c)
            r, c = r-1, c-1
        elif 1 == state:
            if score == xmatchy[r-1][c] - d: state, score = 0, score + d
            else : state, score = 1, score + e
            x.append(xseq[r-1]), y.append('-')
            i.append(r), j.append(0)
            r, c = r-1, c
        elif 2 == state:
            if score == xmatchy[r][c-1] - d: state, score = 0, score + d
            else : state, score = 2, score + e
            x.append('-'), y.append(yseq[c-1])
            i.append(0), j.append(c)
            r, c = r, c-1
    # get alignment results
    align['xseq'], align['yseq'] = ''.join([_ for _ in x[::-1]]), ''.join([_ for _ in y[::-1]])
    align['i'], align['j'] = i[::-1], j[::-1]
    align.update({'imin':0, 'imax':0, 'jmin':0, 'jmax':0})
    for i, j in zip(align['i'], align['j']):
        if i > 0 and j > 0:
            align['imin'], align['imax'] = align['imin'] if align['imin'] > 0 else i, i
            align['jmin'], align['jmax'] = align['jmin'] if align['jmin'] > 0 else j, j
    # return a dict which contain all alignment results
    return align



if __name__ == '__main__':
    print('Running Needleman–Wunsch algorithm ...')
    xseq, yseq = 'PRTEINS', 'PRTWPSEIN'
    align = AlignNW(xseq, yseq)
    print(align['score'])
    print(align['xseq'])
    print(align['yseq'])
    print(align['i'])
    print(align['j'])
    print(align['imin'])
    print(align['imax'])
    print(align['jmin'])
    print(align['jmax'])

    print('Running Smith–Waterman algorithm ...')
    xseq, yseq = 'PLEASANTLY', 'MEANLY'
    align = AlignSW(xseq, yseq)
    print(align['score'])
    print(align['xseq'])
    print(align['yseq'])
    print(align['i'])
    print(align['j'])
    print(align['imin'])
    print(align['imax'])
    print(align['jmin'])
    print(align['jmax'])
