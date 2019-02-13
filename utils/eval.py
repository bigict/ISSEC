import os

def prf(pred_boxes, gt_boxes, coverage = 0.7):
    # used to calculate precision, recall and F1 of one predicted instance
    countp = 0
    countr = 0
    for b in pred_boxes:
        for box in gt_boxes:
            if box_coverage(box, b) > coverage:
                countp += 1
                break
    for box in gt_boxes:
        for b in pred_boxes:
            if box_coverage(box, b) > coverage:
                countr += 1
                break
    precision = 1.0 * countp / (len(pred_boxes) + 0.000001)
    recall = 1.0 * countr / (len(gt_boxes) + 0.000001)
    F1 = 2 * precision * recall / (precision + recall + 0.0001)
    return precision, recall, F1

def box_coverage(real_box, pred_box):
    x1 = max(real_box[0], pred_box[0])
    y1 = max(real_box[1], pred_box[1])
    x2 = min(real_box[2], pred_box[2])
    y2 = min(real_box[3], pred_box[3])
    cx = x2-x1 if (x2-x1) > 0 else 0
    cy = y2-y1 if (y2-y1) > 0 else 0
    real_range = (real_box[2]-real_box[0]) * (real_box[3]-real_box[1])
    return 1.0 * cx * cy / real_range

def box_filter(boxes):
    # combine overlop boxes to one set
    overlap_sets = []
    for box in boxes:
        find = False
        for st in overlap_sets:
            if find == True:
                break
            for b in st:
                if box_coverage(box, b) > 0.2:
                    find = True
                    st.append(box)
                    break
        if find == False:
            overlap_sets.append([box])

    # pick box of top score in one set
    filter_boxes = []
    for s in overlap_sets:
        s = sorted(s, key=lambda x: x[-1], reverse=True)
        filter_boxes.append(s[0])
    return filter_boxes

def possible_boxes(sse3_seq):
    start1 = -1
    start2 = -1
    t1 = False
    t2 = False
    alpha_len_thr = 4
    beta_len_thr = 3
    contact_sse_indices = []
    l = len(sse3_seq)
    for i in range(l):
        if sse3_seq[i] == 'H':
            t1 = True
        else:
            t1 = False
        if t1 == True and start1 == -1:
            start1 = i
        if t1 == False and start1 != -1:
            if i - start1 > alpha_len_thr:
                contact_sse_indices.append((start1, i))
            start1 = -1
        if sse3_seq[i] == 'E':
            t2 = True
        else:
            t2 = False
        if t2 == True and start2 == -1:
            start2 = i
        if t2 == False and start2 != -1:
            if i - start2 > beta_len_thr:
                contact_sse_indices.append((start2, i))
            start2 = -1
    ll = len(contact_sse_indices)
    boxes = []
    for i in range(ll):
        for j in range(i+1, ll):
            p1 = contact_sse_indices[i]
            p2 = contact_sse_indices[j]
            boxes.append([max(0, p1[0]-1), max(0, p2[0]-1), min(p1[1]+1, l-1), min(p2[1]+1, l-1)])
            boxes.append([max(0, p2[0]-1), max(0, p1[0]-1), min(p2[1]+1, l-1), min(p1[1]+1, l-1)])

    return boxes

def box_filter_with_ss3(boxes, ss3):
    ss3_div_boxes = possible_boxes(ss3)
    # map predicted boxes to possible according to sse division
    candidate_set = {}
    for box in boxes:
        pos = -1
        max_cover = 0
        for i in range(len(ss3_div_boxes)):
            # get max box with coverage with possible one
            coverage = box_coverage(box, ss3_div_boxes[i])
            if coverage > max_cover:
                max_cover = coverage
                pos = i
        if pos > -1:
            # also remain the most score one
            if candidate_set.has_key(pos) and candidate_set[pos][-1] < box[-1]:
                candidate_set[pos] = box
            else:
                candidate_set[pos] = box
    filter_boxes = []
    for i, box_list in candidate_set.items():
        filter_boxes.append(ss3_div_boxes[i] + box_list[4:])
    return filter_boxes
