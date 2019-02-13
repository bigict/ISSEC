#!/usr/bin/env python

from math import sqrt
import math
import numpy as np

class matrix4sse(object):
    """
    Iuput include: 
    <1> sse sequence of 3-states;
    <2> distance matrix, each element for distance of each residue pair;
    <3> angle matrix(cosine value), each element for distance of each residue pair.
    <4> distance threshold for a residue contact, 8 default
    <5> continual alpha length threshold, >3 defauld
    <6> continual beta length threshold, >3 defauld
    Output:
    <1> sse pattern in protein, self.sse_pattern
    <2> sse elements(H: helix sse element, E: beta sse element), self.elements
    <3> residue contact map, self.contact_map
    <4> sse contact map(1: aa, 2: bb parallel, 3: bb anti-parallel), self.sse_matrix_labeled
    <4> boxes and masks label for object detection
    """
    def __init__(self, ss3_seq, distance_matrix, angle_matrix,
            contact_distance=8, alpha_min_lenght=4, beta_min_lenght=1):
        self.sse3_seq = ss3_seq
        self.distance_matrix = distance_matrix
        self.angle_matrix = angle_matrix

        self.contact_distance = contact_distance
        self.alpha_len_thr = alpha_min_lenght
        self.beta_len_thr = beta_min_lenght
        self.angle_thr = -1.0/2

        self.betanum_thr = 4
        self.beta_distance_str = contact_distance

        self.helix_pos = []
        self.beta_pos = []
        # transfer -1 into 20
        temp = np.where(distance_matrix > 0, distance_matrix, 20) 
        self.contact_map = np.where(temp < self.contact_distance, 1, 0)
        self.findSSE()
        self.generate_sse_matrix()
        self.gen_boxAndMask()

    def findSSE(self):
        # Find SSE region(include alpha-helix and beta-sheet) in protein
        self.sse_elements = ''
        self.sse_pattern = np.array(['-'] * len(self.sse3_seq))

        ## Function to find H and E element
        start1 = -1
        start2 = -1
        t1 = False
        t2 = False
        self.contact_sse_indices = []
        for i in range(len(self.sse3_seq)):
            if self.sse3_seq[i] == 'H':
                t1 = True
            else:
                t1 = False
            if t1 == True and start1 == -1:
                start1 = i
            if t1 == False and start1 != -1:
                if i - start1 > self.alpha_len_thr:
                    self.sse_elements += 'H'
                    self.helix_pos.append([start1, i-1])
                    self.sse_pattern[start1: i] = 'H'
                    self.contact_sse_indices.append((start1, i))
                start1 = -1
            if self.sse3_seq[i] == 'E':
                t2 = True
            else:
                t2 = False
            if t2 == True and start2 == -1:
                start2 = i
            if t2 == False and start2 != -1:
                if i - start2 > self.beta_len_thr:
                    self.sse_elements += 'E'
                    self.beta_pos.append([start2, i-1])
                    self.sse_pattern[start2: i] = 'E'
                    self.contact_sse_indices.append((start2, i))
                start2 = -1

    def dealing_aa(self):
        # Used to put residue numbers of helix, one helix in one array
        self.helix_res = []                                          
        # Remember which helix one residue was on
        self.helix_num = {}                                          
        for j in range(len(self.helix_pos)):
            tmp = []
            for i in range(self.helix_pos[j][0],self.helix_pos[j][1]+1):
                tmp.append(i)
                self.helix_num[i] = j
            self.helix_res.append(tmp)

    def judge_aa_contact(self, a, b):
        H_num = self.helix_num[a]
        HELIX1 = self.helix_res[H_num]
        H_num = self.helix_num[b]
        HELIX2 = self.helix_res[H_num]
        a2 = a
        ###### Move forward 3 or 4 ##########
        if a+3 in HELIX1 and a+4 in HELIX1:
            if self.angle_matrix[a,a+3] < self.angle_matrix[a,a+4]:
                a2 = a + 4
            else:
                a2 = a + 3
        elif a+3 in HELIX1 and a+4 not in HELIX1:
            a2 = a + 3
        if a2 != a:
            for i in HELIX2:
                if i == b:
                    continue
                if self.distance_matrix[a2,i] <= 10 \
                        and self.angle_matrix[a2,i] < self.angle_thr:
                    return True
        ####### Move back 3 or 4 ##########
        a2 = a
        if a-3 in HELIX1 and a-4 in HELIX1:
            if self.angle_matrix[a,a-3] < self.angle_matrix[a,a-4]:
                a2 = a - 4
            else:
                a2 = a - 3
        elif a-3 in HELIX1 and a-4 not in HELIX1:
            a2 = a - 3
        if a2 != a:
            for i in HELIX2:
                if i == b:
                    continue
                if self.distance_matrix[a2,i] <= 10 \
                        and self.angle_matrix[a2,i] < self.angle_thr:
                    return True
        return False

    def find_aa_contacts(self):
        self.dealing_aa()
        self.aa_pairs = []
        for i in range(len(self.helix_pos)):
            for j in range(i+1, len(self.helix_pos)):
                for a in self.helix_res[i]:
                    if len(self.aa_pairs) > 0 and self.aa_pairs[-1] == [i, j]:
                        break
                    for b in self.helix_res[j]:
                        if self.distance_matrix[a,b] <= 10 \
                                and self.angle_matrix[a,b] < self.angle_thr \
                                and self.judge_aa_contact(a, b) == True:
                            self.aa_pairs.append([i, j])
                            break


    def judge_bb_contact(self, x, y):
        lx = self.beta_pos[x][1] - self.beta_pos[x][0] + 1
        ly = self.beta_pos[y][1] - self.beta_pos[y][0] + 1
        cmap = np.zeros((lx, ly))
        dp = np.zeros((lx+1, ly+1))
        buldge = np.zeros((lx+1, ly+1))
        ##### search parallel contacted beta sheet using DP #####
        for i in xrange(lx):
            for j in xrange(ly):
                cmap[i][j] = (1 if self.distance_matrix[self.beta_pos[x][0] + i, \
                        self.beta_pos[y][0] + j] <= self.beta_distance_str else 0)
                if cmap[i][j] == 1:
                    tt = False
                    if buldge[i+1][j] == 0 and dp[i+1][j] > dp[i][j]:
                        dp[i+1][j+1] = dp[i+1][j] + 1
                        buldge[i+1][j+1] = 1
                        tt = True
                    if buldge[i][j+1] == 0 and dp[i][j+1] > dp[i][j]:
                        dp[i+1][j+1] = dp[i][j+1] + 1
                        buldge[i+1][j+1] = 1
                        tt = True
                    if tt == False:
                        dp[i+1][j+1] = dp[i][j] + 1
                        buldge[i+1][j+1] = buldge[i][j]
                if dp[i+1][j+1] >= self.betanum_thr:
                    return 1
        dp = np.zeros((lx+1, ly+1))
        buldge = np.zeros((lx+1, ly+1))
        ##### search anti-parallel contacted beta sheet using DP #####
        for i in xrange(lx):
            for j in xrange(ly):
                cmap[i][j] = (1 if self.distance_matrix[self.beta_pos[x][0] + i, \
                        self.beta_pos[y][1] - j] <= self.beta_distance_str else 0)
                if cmap[i][j] == 1:
                    tt = False
                    if buldge[i+1][j] == 0 and dp[i+1][j] > dp[i][j]:
                        dp[i+1][j+1] = dp[i+1][j] + 1
                        buldge[i+1][j+1] = 1
                        tt = True
                    if buldge[i][j+1] == 0 and dp[i][j+1] > dp[i][j]:
                        dp[i+1][j+1] = dp[i][j+1] + 1
                        buldge[i+1][j+1] = 1
                        tt = True
                    if tt == False:
                        dp[i+1][j+1] = dp[i][j] + 1
                        buldge[i+1][j+1] = buldge[i][j]
                if dp[i+1][j+1] >= self.betanum_thr:
                    return 2
        return 0


    def find_bb_contacts(self):
        self.bb1_pairs = []
        self.bb2_pairs = []
        for i in xrange(len(self.beta_pos)):
            for j in xrange(i+1, len(self.beta_pos)):
                if self.judge_bb_contact(i, j) == 1:
                    self.bb1_pairs.append([i, j])
                if self.judge_bb_contact(i, j) == 2:
                    self.bb2_pairs.append([i, j])

    def generate_sse_matrix(self):
        self.find_aa_contacts()
        self.find_bb_contacts()

        lens = len(self.sse_elements)
        self.sse_matrix_labeled = np.zeros((lens, lens), dtype=np.int8)
        a = 0
        rec = np.zeros(lens).astype('int16')
        for i in xrange(lens):
            if self.sse_elements[i] == 'H':
                a += 1
            rec[i] = a
        for i in xrange(lens):
            for j in xrange(i+1, lens):
                if self.sse_elements[i] == 'H' and self.sse_elements[j] == 'H':
                    self.sse_matrix_labeled[i][j] = ([rec[i]-1, rec[j]-1] in self.aa_pairs)
                if self.sse_elements[i] == 'E' and self.sse_elements[j] == 'E':
                    if ([i - rec[i], j - rec[j]] in self.bb1_pairs):
                        self.sse_matrix_labeled[i][j] = 2    
                    elif ([i - rec[i], j - rec[j]] in self.bb2_pairs):
                        self.sse_matrix_labeled[i][j] = 3 
        self.sse_matrix_labeled += np.transpose(self.sse_matrix_labeled)
        self.sse_matrix = np.where(self.sse_matrix_labeled > 0, 1, 0)

    def gen_boxAndMask(self):
        self.boxes = []
        self.masks = []
        ssesLen = len(self.sse_matrix_labeled)
        ids = self.contact_sse_indices
        for i in xrange(ssesLen):
            for j in xrange(ssesLen):
                if self.sse_matrix_labeled[i][j] > 0:
                    cls = self.sse_matrix_labeled[i][j]
                    # box shape = (x1, y1, x2, y2, class)
                    self.boxes.append([ ids[j][0]-1, ids[i][0]-1, \
                            ids[j][1]+1, ids[i][1]+1, cls ])  
                    mask = np.zeros(shape=self.contact_map.shape, dtype=np.int8)
                    mask[ ids[i][0]:ids[i][1]+1, ids[j][0]:ids[j][1]+1 ] = \
                            self.contact_map[ ids[i][0]:ids[i][1]+1, ids[j][0]:ids[j][1]+1 ]
                    # mask shpe = (L, L), pixel on instance equal to 1
                    self.masks.append(mask.tolist())    
        self.instance_num = len(self.boxes)
        return self.instance_num, self.boxes, self.masks

    def output_contactmap(self):
        m = self.contact_map
        mp = []
        for x in m:
            mp.append([str(i) for i in x])
        for (x1, y1, x2, y2, _) in self.boxes:
            for i in xrange(x1, x2+1):
                mp[i][y1] = '|'
                mp[i][y2] = '|'
            for i in xrange(y1, y2+1):
                mp[x1][i] = '-'
                mp[x2][i] = '-'
        content = []
        content.append('\n>residue contacts map with boxes')
        for xx in mp:
            content.append(''.join(xx))
        return '\n'.join(content)


    def output_sse_matrix(self):
        content = []
        content.append('>residue sse')
        content.append(''.join(self.sse3_seq))
        content.append('>SSE patterns')
        content.append(''.join(list(self.sse_pattern)))
        content.append('>sse contacts map')
        content.append('    ' + '  '.join(list(self.sse_elements)) + '\n')
        for i in xrange(len(self.sse_matrix_labeled)):
            content.append(self.sse_elements[i] + '   ' + '  '.join([str(j) \
                    for j in self.sse_matrix_labeled[i]]))
        return '\n'.join(content) 

