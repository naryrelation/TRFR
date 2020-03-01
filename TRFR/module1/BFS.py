
from Queue import Queue
import random

def BFS(kb, entity1, entity2,n,list1,tag):
	res = foundPaths(kb)
	res.markFound(entity1, None, None)
	q = Queue()
	q.put(entity1)
	i = 0
	number=-1
	entity_list={}
	path_list={}
	while (not q.empty()):
		curNode = q.get()
		for path in kb.getPathsFrom(curNode):
			i=i+1
			if i>2000000: break
			nextEntity = path.connected_entity
			connectRelation = path.relation
			if tag=="neg" and list1.count(connectRelation)!=0:
				continue
			if (not res.isFound(nextEntity)) :
				q.put(nextEntity)
				res.markFound(nextEntity, curNode, connectRelation)
			if (nextEntity == entity2):
				number+=1
				entity_list[number], path_list[number] = res.reconstructPath(entity1, entity2)
				# k=0
				# for k in range(len(entity_list)):
				# 	if k==0:
				# 		continue
				# 	res.entities[entity_list[k]]=(False, None, None)
				res.entities[entity_list[number][len(entity_list[number])-1]] = (False, None, None)
				p = []
				while(not q.empty()):
					temp=q.get()
					p.append(temp)
				if len(p)!=0:
					p.pop(-1)
				j=0
				for j in range(len(p)):
					q.put(p[j])
				if number==(n-1):
					break
		if number == (n-1):
			break
				# if len(path_list) <n:
				# 	continue
				# if len(path_list) ==n:
				# 	return (True, entity_list, path_list)
				# if len(path_list) >n: break
	return (True, entity_list, path_list)

def test():
	pass

class foundPaths(object):
	def __init__(self, kb):
		self.entities = {}
		for entity, relations in kb.entities.iteritems():
			self.entities[entity] = (False, "", "")

	def isFound(self, entity):
		return self.entities[entity][0]
			

	def markFound(self, entity, prevNode, relation):
		self.entities[entity] = (True, prevNode, relation)

	def reconstructPath(self, entity1, entity2):
		entity_list = []
		path_list = []
		curNode = entity2
		while(curNode != entity1):
			entity_list.append(curNode)

			path_list.append(self.entities[curNode][2])
			curNode = self.entities[curNode][1]
		entity_list.append(curNode)
		entity_list.reverse()
		path_list.reverse()
		return (entity_list, path_list)

	def __str__(self):
		res = ""
		for entity, status in self.entities.iteritems():
			res += entity + "[{},{},{}]".format(status[0],status[1],status[2])
		return res			