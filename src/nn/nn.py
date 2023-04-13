from .matrix import Matrix
import math



__all__=["ActivationFunction","NeuralNetwork","SIGMOID","TANH"]



class ActivationFunction:
	def __init__(self,f,df):
		self.f=f
		self.df=df



SIGMOID=ActivationFunction(
	lambda v,*a:1/(1+math.exp(-v)),
	lambda v,*a:v*(1-v)
)
TANH=ActivationFunction(
	lambda v,*a:math.tanh(v),
	lambda v,*a:1-(v**2)
)



class NeuralNetwork:
	def __init__(self,input_,hidden=None,output=None,lr=0.01):
		if (type(input_)==dict):
			self.fromJSON(input_)
		else:
			self.i=input_
			self.h=hidden+[output]
			self.wl=[]
			self.bl=[]
			for k in range(0,len(self.h)):
				s=(self.i if k==0 else self.h[k-1])
				e=self.h[k]
				self.wl.append(Matrix(e,s).randomize())
				self.bl.append(Matrix(e,1).randomize())
			self.lr=lr



	def predict(self,i):
		o=Matrix.from_array(i)
		for k in range(0,len(self.h)):
			o=Matrix.mult(self.wl[k],o)
			o=Matrix.add(o,self.bl[k])
			o=o.map(SIGMOID.f)
		return o.to_array()



	def train(self,i,t):
		def lrn(l1,l2,w,b,e,df,lr):
			g=Matrix.mapN(l2,df)
			g.multN(e)
			g.multS(lr)
			d=Matrix.transpose(l1)
			d=Matrix.mult(g,d)
			w=Matrix.add(w,d)
			b=Matrix.add(b,g)
			return w,b
		i=Matrix.from_array(i)
		ol=[]
		for k in range(0,len(self.h)):
			s=(i if k==0 else ol[-1])
			o=Matrix.mult(self.wl[k],s)
			o=Matrix.add(o,self.bl[k])
			ol.append(o.map(SIGMOID.f))
		t=Matrix.from_array(t)
		e=Matrix.sub(t,ol[-1])
		for k in range(len(self.h)-1,-1,-1):
			s=(ol[k-1] if k>0 else i)
			self.wl[k],self.bl[k]=lrn(s,ol[k],self.wl[k],self.bl[k],e,SIGMOID.df,self.lr)
			lw=Matrix.transpose(self.wl[k])
			e=Matrix.mult(lw,e)



	def train_multiple(self,d,t,log=True):
		l=-1
		for i in range(0,t):
			if (log==True and int(i/t*100)>l):
				l=int(i/t*100)
				print(f"{l}% complete...")
			for k in d:
				self.train(k[0],k[1])



	def test(self,d,log=True):
		if(log==True):
			print("TEST".center(40,"="))
		a=[]
		for k in d:
			o=self.predict(k[0])
			if(log==True):
				print(f"Input: {str(k[0])}\tTarget Output: {str(k[1])}\tOutput: {str(o)}")
			a+=Matrix.diff(Matrix.from_array(k[1]),Matrix.from_array(o)).to_array()
		return round((1-sum(a)/len(a))*10000)/100



	def toJSON(self):
		wl=[]
		for k in self.wl:
			wl.append(k.data)
		bl=[]
		for k in self.bl:
			bl.append(k.data)
		json={"i":self.i,"hl":self.h,"wl":wl,"bl":bl,"lr":self.lr}
		return json
	def fromJSON(self,json):
		self.i=json["i"]
		self.h=json["hl"]
		self.wl=[]
		for k in json["wl"]:
			self.wl.append(Matrix(len(k[0]),len(k)).fill(k))
		self.bl=[]
		for k in json["bl"]:
			self.bl.append(Matrix(len(k[0]),len(k)).fill(k))
		self.lr=json["lr"]
