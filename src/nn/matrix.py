import random



class Matrix:
	def __init__(self,w,h):
		self.w=w
		self.h=h
		self.gen()
	def __str__(self):
		return self.__repr__()
	def __repr__(self):
		return str(self.data)



	def gen(self):
		self.data=[]
		for y in range(0,self.h):
			self.data+=[[0]*self.w]
		return self



	def randomize(self):
		for y in range(0,self.h):
			for x in range(0,self.w):
				self.data[y][x]=random.random()
		return self



	def fill(self,d):
		self.data=d
		return self



	def multS(self,s):
		return self.map(lambda v,*a:v*s)



	def multN(self,a):
		return self.map(lambda v,x,y:v*a.data[y][x])



	def map(self,f):
		for y in range(0,self.h):
			for x in range(0,self.w):
				self.data[y][x]=f(self.data[y][x],x,y)
		return self



	def to_array(self):
		a=[]
		self.map(lambda v,*d:a.append(v))
		return a



	@staticmethod
	def from_array(a):
		m=Matrix(len(a),1)
		m.fill([a])
		return m



	@staticmethod
	def mapN(a,f):
		m=Matrix(a.w,a.h)
		for y in range(0,a.h):
			for x in range(0,a.w):
				m.data[y][x]=f(a.data[y][x],x,y)
		return m



	@staticmethod
	def mult(a,b):
		if (a.h!=b.w):
			print("WRONG SIZE!")
			return None
		def f(e,x,y):
			s=0
			for k in range(0,a.h):
				s+=a.data[k][x]*b.data[y][k]
			return s
		return Matrix(a.w,b.h).map(f)



	@staticmethod
	def add(a,b):
		return Matrix(a.w,a.h).fill([[a.data[y][x]+b.data[y][x] for x in range(0,a.w)] for y in range(0,a.h)])



	@staticmethod
	def sub(a,b):
		return Matrix(a.w,a.h).fill([[a.data[y][x]-b.data[y][x] for x in range(0,a.w)] for y in range(0,a.h)])



	@staticmethod
	def diff(a,b):
		return Matrix(a.w,a.h).fill([[abs(a.data[y][x]-b.data[y][x]) for x in range(0,a.w)] for y in range(0,a.h)])



	@staticmethod
	def transpose(a):
		return Matrix(a.h,a.w).map(lambda v,x,y:a.data[x][y])