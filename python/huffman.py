class Node(object):
	def __init__(self,name=None,value=None):
		self._name=name
		self._value=value
		self._left=None
		self._right=None
class HuffmanTree(object):
	def __init__(self,char_weights):
		self.a=[Node(part,char_weights[part]) for part in char_weights]
		
		while len(self.a)!=1:    
			self.a.sort(key=lambda node:node._value,reverse=True)
			c=Node(value=(self.a[-1]._value+self.a[-2]._value))
			c._left=self.a.pop(-1)
			c._right=self.a.pop(-1)
			self.a.append(c)
		self.root=self.a[0]
	re	self.b=range(100)          
	def pre(self,tree,length):
		node=tree
		if (not node):
			return
		elif node._name:
			print node._name + '\'s encode is :',
			for i in range(length):
				print self.b[i],
			print '\n'
			return
		self.b[length]=0
		self.pre(node._left,length+1)
		self.b[length]=1
		self.pre(node._right,length+1)

	def get_code(self):
		self.pre(self.root,0)

if __name__=='__main__':
	with open("ACGAN.py","r") as f:
		read_file= f.read() 
	print(str(read_file))
	s=str(read_file)
	resoult={}
	for i in set(s):
		resoult[i]=s.count(i)		
	print(resoult)
	tree=HuffmanTree(resoult)
	tree.get_code()