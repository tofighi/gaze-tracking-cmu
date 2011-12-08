from collections import defaultdict

def featureVec35D():
    f = open('gazedata_ordered.dat','r').read()

    fw = open('dataset_engagement.tab','w')
    fw.write('\t'.join([str(i) for i in range(35)]) + '\tengagement\n')
    fw.write(('d\t'*35) + 'd\n')
    fw.write('\t'*35 + 'class\n')

    for line in f.splitlines():
        tokens = line.split(',')
        personNum = tokens[0]
        engagement = tokens[1]
        
        data = tokens[2:]
        
        fw.write('\t'.join(data) + '\t' + engagement + '\n')
        
    fw.close()
    
def featureVec5D():
    f = open('gazedata_ordered.dat','r').read()

    fw = open('dataset_engagement_5D.tab','w')
    fw.write('\t'.join(['floor','vent','robot','eachother','window','engagement']) + '\n')
    fw.write(('d\t'*5) + 'd\n')
    fw.write('\t'*5 + 'class\n')

    for line in f.splitlines():
        fv = defaultdict(int)
        tokens = line.split(',')
        personNum = tokens[0]
        engagement = tokens[1]
        
        data = tokens[2:]
        for datapoint in data:
            fv[datapoint] += 1
        
        fw.write('\t'.join([str(fv['floor']),
                            str(fv['vent']),
                            str(fv['robot']),
                            str(fv['eachother']),
                            str(fv['window'])]) + '\t' + engagement + '\n')

    fw.close()
    
def makeBigDatasetFromBatch():
    fw = open('dataset_batches_2_only.tab','w')
    fw.write('nx\tny\tnz\tn2x\tn2y\tn2z\tclass_view\n')
    fw.write('c\tc\tc\tc\tc\tc\td\n')
    fw.write('\t\t\t\t\t\tclass\n')
    
    lbl_data = open('gazedata_ordered.csv','r').read()
    ml_data = []
    p_data = []
    for person in lbl_data.splitlines():
        lbls = person.split(',')[2:]
        print len(lbls), lbls
        p_data.append(lbls)
    
    print len(p_data)
    
    peopleNums = [5,4,4,3,3,5,4,3,3]
    batchNum = 0
    for fName in ['batch%d.dat' % (i+1) for i in range(1,2)]:
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
