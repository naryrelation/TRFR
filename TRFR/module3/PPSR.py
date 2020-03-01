#coding=utf-8
from KB import KB
from BFS import BFS
import time
import re
import json
import  random
global number
#average
def average(list):
	sum = 0
	for item in list:
		sum += item
	return sum/len(list)

#2d dict
def dict(thedict, key_a, key_b,value):
	adic = thedict.keys()
	if key_a in adic:
		#dic0=dict(thedict[key_a])
		bdic=thedict[key_a].keys()
		if key_b in bdic:
			if thedict[key_a][key_b]!=value:
				print("wrong1"+str(key_a)+"\n")
		else:
			thedict[key_a].update({key_b: value})
			#print("wrong2" + str(key_a) + "\n")
	else:
		thedict.update({key_a:{key_b: value}})
#1d dict
def dic1_list(thedict, key_a,valuelist):
	adic = thedict.keys()
	a=[]
	if key_a in adic:
			thedict[key_a].append(valuelist)
	else:
		thedict.update({key_a:a})
		thedict[key_a].append(valuelist)
def takeSecond(elem):
	return elem[2]

#one term
def cal_map(ytest,ypre):
	aps=[]
	for i in range(len(ypre)):
		correct = 0
		ranks = []
		for j in range(len(ypre[i])):
			if ypre[i][j]==ytest[i]:
				correct+=1
				ranks.append(correct / (1.0 + j))
				break
		if len(ranks) == 0:
			ranks.append(0)
		aps.append(average(ranks))
	mean_ap = average(aps)
	return mean_ap*100
#two terms
def cal_map2(ytest1,ytest2,ypre1,ypre2):
	aps = []
	for i in range(len(ypre1)):
		correct = 0
		ranks = []
		for j in range(len(ypre1[i])):
			if ypre1[i][j] == ytest1[i] and ypre2[i][j] == ytest2[i]:
				correct += 1
				ranks.append(correct / (1.0 + j))
				break
		if len(ranks) == 0:
			ranks.append(0)
		aps.append(average(ranks))
	mean_ap = average(aps)
	return mean_ap * 100
#one term
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
	num=float(len(ytest))
	hit[1]=float(hit[1]/num)
	hit[3] = hit[3] / (num)
	hit[5] = hit[5] / (num)
	hit[10] = hit[10] / (num)
	hit[20] = hit[20] / (num)
	return hit

#two terms
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
	num=float(len(ytest1))
	hit[1]=hit[1]/num
	hit[3] = hit[3] / (num)
	hit[5] = hit[5] / (num)
	hit[10] = hit[10] / (num)
	hit[20] = hit[20] / (num)
	return hit


def gen_minerca_test(rnn_result,m_test,m_new_test1,m_new_test2,m_new_test3,dataset_test):
	n=1599
	entitypair=[]
	relation=[]
	entity_relaiton = {}
	for i in range(n):
		entitypair.append(list())
	i=0
	j=1
	with open(m_test, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			dict(entity_relaiton, concepts[0], concepts[1], concepts[2])
			if j%2==1:
				entitypair[i].append(concepts[0])
				relation.append(concepts[1])
				j+=1
			else:
				entitypair[i].append(concepts[0])
				j += 1
				i+=1

	i=0
	testline = []
	with open(rnn_result, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			concepts[0]=int(concepts[0])
			bdic1 = entity_relaiton[entitypair[concepts[0]][0]].keys()
			bdic2 = entity_relaiton[entitypair[concepts[0]][1]].keys()
			l=[1]

			i=1
			if concepts[i]==relation[concepts[0]]:
				testline.append(entitypair[concepts[0]][0]+"\t"+concepts[i]+"\t"+entity_relaiton[entitypair[concepts[0]][0]][concepts[i]]+"\n")
				testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +entity_relaiton[entitypair[concepts[0]][1]][concepts[i]] + "\n")
			else:
				if concepts[i] in bdic1 :
					testline.append(entitypair[concepts[0]][0] + "\t" + concepts[i] + "\t" +
								   entity_relaiton[entitypair[concepts[0]][0]][concepts[i]] + "\n")
				else:
					testline.append(entitypair[concepts[0]][0] + "\t" + concepts[i] + "\t" +
									"PAD" + "\n")
				if concepts[i] in bdic2:
					testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +
									entity_relaiton[entitypair[concepts[0]][1]][concepts[i]] + "\n")
				else:
					testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +
									"PAD" + "\n")
	with open(m_new_test1, 'w') as f:
		for i in range(len(testline)):
			f.write(testline[i])
	testline = []
	with open(rnn_result, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			concepts[0]=int(concepts[0])
			bdic1 = entity_relaiton[entitypair[concepts[0]][0]].keys()
			bdic2 = entity_relaiton[entitypair[concepts[0]][1]].keys()
			l=[1]
			i=2
			if concepts[i]==relation[concepts[0]]:
				testline.append(entitypair[concepts[0]][0]+"\t"+concepts[i]+"\t"+entity_relaiton[entitypair[concepts[0]][0]][concepts[i]]+"\n")
				testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +entity_relaiton[entitypair[concepts[0]][1]][concepts[i]] + "\n")
			else:
				if concepts[i] in bdic1 :
					testline.append(entitypair[concepts[0]][0] + "\t" + concepts[i] + "\t" +
								   entity_relaiton[entitypair[concepts[0]][0]][concepts[i]] + "\n")
				else:
					testline.append(entitypair[concepts[0]][0] + "\t" + concepts[i] + "\t" +
									"PAD" + "\n")
				if concepts[i] in bdic2:
					testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +
									entity_relaiton[entitypair[concepts[0]][1]][concepts[i]] + "\n")
				else:
					testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +
									"PAD" + "\n")
	with open(m_new_test2, 'w') as f:
		for i in range(len(testline)):
			f.write(testline[i])
	testline = []
	with open(rnn_result, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			concepts[0]=int(concepts[0])
			bdic1 = entity_relaiton[entitypair[concepts[0]][0]].keys()
			bdic2 = entity_relaiton[entitypair[concepts[0]][1]].keys()
			l=[1]
			i=3
			if concepts[i]==relation[concepts[0]]:
				testline.append(entitypair[concepts[0]][0]+"\t"+concepts[i]+"\t"+entity_relaiton[entitypair[concepts[0]][0]][concepts[i]]+"\n")
				testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +entity_relaiton[entitypair[concepts[0]][1]][concepts[i]] + "\n")
			else:
				if concepts[i] in bdic1 :
					testline.append(entitypair[concepts[0]][0] + "\t" + concepts[i] + "\t" +
								   entity_relaiton[entitypair[concepts[0]][0]][concepts[i]] + "\n")
				else:
					testline.append(entitypair[concepts[0]][0] + "\t" + concepts[i] + "\t" +
									"PAD" + "\n")
				if concepts[i] in bdic2:
					testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +
									entity_relaiton[entitypair[concepts[0]][1]][concepts[i]] + "\n")
				else:
					testline.append(entitypair[concepts[0]][1] + "\t" + concepts[i] + "\t" +
									"PAD" + "\n")
	with open(m_new_test3, 'w') as f:
		for i in range(len(testline)):
			f.write(testline[i])

# generate model result
def gen_model_result(m1,m2,m3,m_test1,m_test2,m_test3,result_file,plist):
	n1 = 1599#
	n2=319800#1599*200
	entitypair = []
	relation = []
	relation2 = []
	relation3 = []
	entity_relaiton = {}
	score=[]
	score2 = []
	score3 = []
	result = {}
	for i in range(n1):
		entitypair.append(list())
	for i in range(n2):
		score.append(list())
		score2.append(list())
		score3.append(list())
	i = 0
	j = 1
	with open(m_test1, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			#dict(entity_relaiton, concepts[0], concepts[1], concepts[2])
			if j % 2 == 1:
				#entitypair[i].append(concepts[0])
				relation.append(concepts[1])
				j += 1
			else:
				#entitypair[i].append(concepts[0])
				j += 1
				i += 1
	i = 0
	j = 1
	with open(m_test2, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			#dict(entity_relaiton, concepts[0], concepts[1], concepts[2])
			if j % 2 == 1:
				#entitypair[i].append(concepts[0])
				relation2.append(concepts[1])
				j += 1
			else:
				#entitypair[i].append(concepts[0])
				j += 1
				i += 1
	i = 0
	j = 1
	with open(m_test3, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			#dict(entity_relaiton, concepts[0], concepts[1], concepts[2])
			if j % 2 == 1:
				#entitypair[i].append(concepts[0])
				relation3.append(concepts[1])
				j += 1
			else:
				#entitypair[i].append(concepts[0])
				j += 1
				i += 1
	with open(m1, 'r') as f:
		score_number=0
		for line in f.readlines():
			concepts = line.strip().split("\t")
			#dict(result_score, concepts[0], concepts[1], concepts[2])
			score[score_number].append(concepts[0])
			score[score_number].append(concepts[1])
			score[score_number].append(float(concepts[2]))
			score_number+=1
	with open(m2, 'r') as f:
		score_number=0
		for line in f.readlines():
			concepts = line.strip().split("\t")
			#dict(result_score, concepts[0], concepts[1], concepts[2])
			score2[score_number].append(concepts[0])
			score2[score_number].append(concepts[1])
			score2[score_number].append(float(concepts[2]))
			score_number+=1
	with open(m3, 'r') as f:
		score_number=0
		for line in f.readlines():
			concepts = line.strip().split("\t")
			#dict(result_score, concepts[0], concepts[1], concepts[2])
			score3[score_number].append(concepts[0])
			score3[score_number].append(concepts[1])
			score3[score_number].append(float(concepts[2]))
			score_number+=1
	keylist=[]
	i=0
	for o in range(n1):
		p=i*100
		q=(i+1)*100
		str0=str(o)+"\t"+score[p][0]+"\t"+score[q][0]
		keylist.append(str0)
		hit=0
		list2 = ["none", "none", 0]
		for j in range(100):
			if hit == 100: break
			for k in range(100):
				if hit==100:break
				if score[p+j][1]==score[q+k][1]:
					str1=str(o)+"\t"+score[p+j][0]+"\t"+score[q+k][0]
					list1=[]
					list1.append(relation[o])
					list1.append(score[q+k][1])
					list1.append(score[p+j][2]*score[q+k][2]*plist[o][0]*(-1))
					dic1_list(result,str1,list1)
					hit+=1
		if hit<100:
			for r in range(100-hit):
				dic1_list(result,str0,list2)
		i=i+2
	i = 0
	for o in range(n1):
		p = i * 100
		q = (i + 1) * 100
		str0 = str(o)+"\t"+ score2[p][0] + "\t" + score2[q][0]
		keylist.append(str0)
		hit = 0
		list2 = ["none", "none", 0]
		for j in range(100):
			if hit == 100: break
			for k in range(100):
				if hit == 100: break
				if score2[p + j][1] == score2[q + k][1]:
					str1 = str(o)+"\t" + score2[p + j][0] + "\t" + score2[q + k][0]
					list1 = []
					list1.append(relation2[o])
					list1.append(score2[q + k][1])
					list1.append(score2[p + j][2] * score2[q + k][2] * plist[o][1]*(-1))
					dic1_list(result, str1, list1)
					hit += 1
		if hit < 100:
			for r in range(100 - hit):
				dic1_list(result, str0, list2)
		i = i + 2
	i = 0
	for o in range(n1):
		p = i * 100
		q = (i + 1) * 100
		str0 =str(o)+"\t"+ score3[p][0] + "\t" + score3[q][0]
		keylist.append(str0)
		hit = 0
		list2 = ["none", "none", 0]
		for j in range(100):
			if hit == 100: break
			for k in range(100):
				if hit == 100: break
				if score3[p + j][1] == score3[q + k][1]:
					str1 = str(o)+"\t" + score[p + j][0] + "\t" + score3[q + k][0]
					list1 = []
					list1.append(relation3[o])
					list1.append(score3[q + k][1])
					list1.append(score3[p + j][2] * score3[q + k][2] * plist[o][2]*(-1))
					dic1_list(result, str1, list1)
					hit += 1
		if hit < 100:
			for r in range(100 - hit):
				dic1_list(result, str0, list2)
		i = i + 2
	print()
	keylist=list(set(keylist))
	templist=[]
	for i in range(1599):
		templist.append("")
	for i in range(len(keylist)):
		concepts = keylist[i].strip().split("\t")
		templist[int(concepts[0])]=keylist[i]
	keylist=templist
	with open(result_file, 'w') as f:
			for i in range(len(keylist)):
				list2=result[keylist[i]]
				list3=[]
				list4=[]
				t1="a"
				t2="a"
				concepts = keylist[i].strip().split("\t")
				str2=concepts[1]+"\t"+concepts[2]
				list2.sort(key=takeSecond, reverse=False)
				for k in range(len(list2)):
					if list2[k][0]!=t1 or list2[k][1]!=t2:
						t1=list2[k][0]
						t2=list2[k][1]
						tt=t1+t2
						if tt not in list4:
							list3.append(list2[k])
							list4.append(tt)
				list2=list3
				for j in range(20):
					if j <len(list2):
						f.write(str2+"\t"+list2[j][0]+"\t"+list2[j][1]+"\n")
					else:
						f.write(str2 + "\t" + list2[0][0] + "\t" + list2[0][1] + "\n")
				#print(key)


def model_eval(model_result,true_result,output_dir):
	mresult1=[]
	mresult2=[]
	tresult1=[]
	tresult2=[]

	with open(true_result, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			tresult1.append(concepts[2])
			tresult2.append(concepts[3])
	n1 = len(tresult1)
	modellist=[]
	with open(model_result, 'r') as f:
		for line in f.readlines():
			modellist.append(line)
	for i in range(n1):
		temp1 = []
		temp2 = []
		for j in range(20):
				concepts = modellist[i*20+j].strip().split("\t")
				temp1.append(concepts[2])
				temp2.append(concepts[3])
		mresult1.append(temp1)
		mresult2.append(temp2)


	with open(output_dir, 'w') as eval:

		# 对两项分别计算map和hits@k
		score_map = cal_map(tresult1, mresult1)
		print("relation-MAP: " + str(score_map))
		eval.write("relation-MAP: " + str(score_map) + '\n')

		hits = cal_hits(tresult1, mresult1)
		for key in hits:
			print("relation-HITS@" + str(key) + ": " + str(hits[key]))
			eval.write("relation-HITS@" + str(key) + ": " + str(hits[key]) + '\n')
		eval.write('\n')
		score_map = cal_map(tresult2, mresult2)
		print("entity-MAP: " + str(score_map))
		eval.write("entity-MAP: " + str(score_map) + '\n')

		hits = cal_hits(tresult2, mresult2)
		for key in hits:
			print("entity-HITS@" + str(key) + ": " + str(hits[key]))
			eval.write("entity-HITS@" + str(key) + ": " + str(hits[key]) + '\n')
		eval.write('\n')
		# 综合计算map和hits@k
		score_map = cal_map2(tresult1,tresult2,mresult1,mresult2)
		print("(e,r)-MAP: " + str(score_map))
		eval.write("(e,r)-MAP: " + str(score_map) + '\n')

		hits = cal_hits2(tresult1,tresult2,mresult1,mresult2)
		for key in hits:
			print("(e,r)-HITS@" + str(key) + ": " + str(hits[key]))
			eval.write("(e,r)-HITS@" + str(key) + ": " + str(hits[key]) + '\n')


def get_p(rnn_pre2):
	pr=[]
	for i in range(1599):
		pr.append([])
	with open(rnn_pre2, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			p1=float(concepts[1][3:-2])
			p2 = float(concepts[2][3:-2])
			p3 = float(concepts[3][3:-2])
			pr[int(concepts[0])].append(p1)
			pr[int(concepts[0])].append(p2)
			pr[int(concepts[0])].append(p3)
	return pr


def error_ana(mfile,tfile):
	mrlist=[]
	trlist=[]
	melist=[]
	telist=[]
	i=0
	with open(mfile, 'r') as f:
		for line in f.readlines():
				concepts = line.strip().split("\t")
				mrlist.append(concepts[2])
				melist.append(concepts[3])
	with open(tfile, 'r') as f:
		for line in f.readlines():
			concepts = line.strip().split("\t")
			trlist.append(concepts[2])
			telist.append(concepts[3])
	n1=0.0
	n2=0.0
	for i in range(len(melist)):
		j=i/20
		if mrlist[i]!=trlist[j]:
			n1+=1
		if melist[i]!=telist[j]:
			n2+=1
	n3=float(len(melist))
	print(float(n1/n3))
	print(float(n2/n3))












if __name__ == "__main__":


	# generate model result
	m1=r'.\data\pathsanswers1'
	m2 = r'.\data\pathsanswers2'
	m3 = r'.\data\pathsanswers3'

	m_test1 = r'.\data\newtest1.txt'#concate with module2's result
	m_test2 = r'.\data\newtest2.txt'
	m_test3 = r'.\data\newtest3.txt'

	module2_result=r'.\data\y_pre1'
	module2_pre2 = r'.\data\y_pre2'

	result_file=r'.\data\model_result'

	#plist=get_p(module2_pre2)
	#gen_model_result(m1,m2,m3,m_test1,m_test2,m_test3,result_file,plist)

	#calculate MAP HITS
	model_result=r'.\data\model_result'
	true_result=r'.\data\true_result'
	output_dir=r'.\data\eval'

	model_eval(model_result,true_result,output_dir)
	#error analysis
	error_ana(model_result,true_result)



