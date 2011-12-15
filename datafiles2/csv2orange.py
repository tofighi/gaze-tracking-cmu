from collections import defaultdict


    
def makeBigDatasetFromBatch():
    fw = open('dataset_batches_8_only.tab','w')
    fw.write('\t'.join([str(i) for i in range(400)]) + '\tclass_view\n')
    fw.write('c\t'*400 + 'd\n')
    fw.write('\t'*400 + 'class\n')
    
    lbl_data = open('window_vent_data2.csv','r').read()
    ml_data = []
    p_data = []
    for person in lbl_data.splitlines():
        lbls = person.split(',')[3:]
        print len(lbls), lbls
        p_data.append(lbls)
    
    print len(p_data)
    
    peopleNums = [5,4,4,3,3,5,4,3,3]
    batchNum = 0
    for fName in ['batch%d_features.dat' % (i+1) for i in [7]]:
        print fName 
        f_data = open(fName,'r').read()
        frameNum = 0
        basePeople = sum(peopleNums[:batchNum])
        for line in f_data.splitlines():
            personNum, normals = line.split()[0], line.split()[1:]
            personNum = str(int(personNum) + basePeople)
            print personNum, basePeople, frameNum//peopleNums[batchNum]
            lbl = p_data[int(personNum)-1][frameNum//peopleNums[batchNum]]
            ml_data.append(normals + [lbl])
            frameNum += 1
        batchNum += 1

    for line in ml_data:
        fw.write('\t'.join(line) + '\n')
    fw.close()
    
    
if __name__ == '__main__':
    #featureVec5D()
    makeBigDatasetFromBatch()
