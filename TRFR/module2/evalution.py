
import argparse
import datetime
#average
def average(list):
	sum = 0
	for item in list:
		sum += item
	return sum/len(list)
#one term map
def cal_map(ytest,ypre):
	aps=[]
	for i in range(len(ypre)):
		correct = 0
		ranks = []
		for j in range(len(ypre[i])):
			if ypre[i][j]==ytest[i]:
				correct+=1
				ranks.append(correct / (1.0 + j))
		if len(ranks) == 0:
			ranks.append(0)
		aps.append(average(ranks))
	mean_ap = average(aps)
	return mean_ap*100
#two terms map
def cal_map2(ytest1,ytest2,ypre1,ypre2):
	aps = []
	for i in range(len(ypre1)):
		correct = 0
		ranks = []
		for j in range(len(ypre1[i])):
			if ypre1[i][j] == ytest1[i] and ypre2[i][j] == ytest2[i]:
				correct += 1
				ranks.append(correct / (1.0 + j))
		if len(ranks) == 0:
			ranks.append(0)
		aps.append(average(ranks))
	mean_ap = average(aps)
	return mean_ap * 100
#one term hits
def cal_hits(ytest,ypre):
	hit={}
	hit[1]=0
	hit[3]=0
	hit[5] = 0
	hit[10] = 0
	hit[20] = 0
	k=[1,3,5,10,20]

	for i in k:
		for j in range(len(ypre)):
			for m in range(len(ypre[j])):
				if (m+1)>i:break
				if ypre[j][m]==ytest[j] and (m+1)<=i:
					hit[i]+=1
					break
				if i==5 and m==4:
					print()
	num=len(ytest)
	hit[1]=hit[1]/num
	hit[3] = hit[3] / num
	hit[5] = hit[5] / num
	hit[10] = hit[10] / num
	hit[20] = hit[20] / num
	return hit

#two terms hits
def cal_hits2(ytest1,ytest2,ypre1,ypre2):
	hit={}
	hit[1]=0
	hit[3]=0
	hit[5] = 0
	hit[10] = 0
	hit[20] = 0
	k=[1,3,5,10,20]

	for i in k:
		for j in range(len(ypre1)):
			for m in range(len(ypre1[j])):
				if (m+1)>i:break
				if ypre1[j][m]==ytest1[j] and ypre2[j][m]==ytest2[j] and (m+1)<=i:
					hit[i]+=1
					break
	num=len(ytest1)
	hit[1]=hit[1]/num
	hit[3] = hit[3] / num
	hit[5] = hit[5] / num
	hit[10] = hit[10] / num
	hit[20] = hit[20] / num
	return hit

def map_hits(args,dataset):
	print('Evalution start.')
	testpath = './results/y_test'
	prepath1 = './results/y_pre1'
	#prepath2 = './results/y_pre2'
	ytest1 = []
	ypre1 = []
	ytest2 = []
	ypre2 = []
	with open(testpath, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			ytest1.append(concepts[1])
			#ytest2.append(concepts[2])
	with open(prepath1, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			temp = []
			for i in range(5):
				temp.append(concepts[i + 1])
			ypre1.append(temp)
	# with open(prepath2, 'r') as f:
	# 	for line in f.readlines():
	# 		concepts = line.strip().split("\t")
	# 		temp = []
	# 		for i in range(20):
	# 			temp.append(concepts[i + 1])
	# 		ypre2.append(temp)
	nowTime = datetime.datetime.now().strftime('%m_%d_%H_%M')  # now date
	with open('./eval_record/'+str(dataset)+"_"+str(nowTime)+"_"+str(args.epochs)+"_"+str(args.batch_size), 'w') as eval:
		eval.write(str(args)+'\n')
		eval.write('\n')
		#two terms
		score_map=cal_map(ytest1,ypre1)
		print("relation-MAP: "+str(score_map))
		eval.write("relation-MAP: "+str(score_map)+'\n')

		hits = cal_hits(ytest1, ypre1)
		for key in hits:
			print("relation-HITS@" + str(key) + ": " + str(hits[key]))
			eval.write("relation-HITS@" + str(key) + ": " + str(hits[key])+'\n')
		eval.write('\n')
		# score_map = cal_map(ytest2, ypre2)
		# print("entity-MAP: " + str(score_map))
		# eval.write("entity-MAP: " + str(score_map)+'\n')
		#
		# hits = cal_hits(ytest2, ypre2)
		# for key in hits:
		# 	print("entity-HITS@" + str(key) + ": " + str(hits[key]))
		# 	eval.write("entity-HITS@" + str(key) + ": " + str(hits[key])+'\n')
		# eval.write('\n')
		# 
		# score_map = cal_map2(ytest1,ytest2,ypre1,ypre2)
		# print("(e,r)-MAP: " + str(score_map))
		# eval.write("(e,r)-MAP: " + str(score_map)+'\n')
		#
		# hits = cal_hits2(ytest1,ytest2,ypre1,ypre2)
		# for key in hits:
		# 	print("(e,r)-HITS@" + str(key) + ": " + str(hits[key]))
		# 	eval.write("(e,r)-HITS@" + str(key) + ": " + str(hits[key])+'\n')


# if __name__ == '__main__':
# 	#map_hits(args)


