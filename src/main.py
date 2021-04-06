from nn.nn import NeuralNetwork
from tkinter import Tk, Label
import json
import os
import random
import time



DB_PATH="./data/data.db"
DATA=[]
COLORS="red,green,blue,orange,yellow,pink,purple,brown,grey".split(",")



with open(DB_PATH,"r") as f:
	d=f.read()
	for k in d.split("\n"):
		DATA.append({"r":int(k.split(",")[0]),"g":int(k.split(",")[1]),"b":int(k.split(",")[2].split(":")[0]),"c":int(k.split(":")[1])})



def batch(BS):
	B=[]
	d=DATA[:]
	random.shuffle(d)
	for k in d[:BS]:
		B+=[[[k["r"]/255,k["g"]/255,k["b"]/255],[(0 if i!=k["c"] else 1) for i in range(0,len(COLORS))]]]
	return B



def train(NN,t,BS,S=0,log=True):
	l=-1
	st=time.time()
	for i in range(S*int(t/10000),t):
		if (log==True and int(i/t*10000)>l):
			l=int(i/t*10000)
			print(f"{l/100}% complete... ({int((time.time()-st)*100)/100}s) Acc={NN.test(batch(1000),log=False)}%")
			st=time.time()
		NN.train_multiple(batch(BS),1,log=False)
	open("./NN-data.json","w").write(json.dumps(NN.toJSON(),indent=4,sort_keys=True))



def predict(NN,C):
	o=NN.predict([i/255 for i in C])
	s=sum(o)
	i=0
	for k in o:
		o[i]=k/s
		i+=1
	i=0
	s=""
	for k in COLORS:
		s+=f"{('>> ' if o.index(max(o))==i else '')}{k.title()} \u2012> {int(o[i]*10000)/100}{(' <<' if o.index(max(o))==i else '')}\n"
		i+=1
	print(s)



def train_mode():
	if (os.path.isfile("./NN-data.json")):
		NN=NeuralNetwork(json.loads(open("./NN-data.json","r").read()))
	else:
		NN=NeuralNetwork(3,[12],len(COLORS),lr=0.005)
	print(NN.test(batch(1000),log=False))
	train(NN,50_000,100)
	print(NN.test(batch(1000),log=False))



def test_mode():
	NN=NeuralNetwork(json.loads(open("./NN-data.json","r").read()))
	tk=Tk()
	tk.resizable(0,0)
	tk.geometry("600x600")
	l=Label(tk,font="Consolas 80",width=600,height=600)
	def g(arg=None):
		C=[int(random.random()*255) for _ in range(3)]
		l["fg"]=("white" if (0.2126*C[0]/255+0.7152*C[1]/255+0.0722*C[2]/255)<0.5 else "black")
		l["bg"]=f"#{'0'*(2-len(hex(C[0])[2:]))+hex(C[0])[2:]}{'0'*(2-len(hex(C[1])[2:]))+hex(C[1])[2:]}{'0'*(2-len(hex(C[2])[2:]))+hex(C[2])[2:]}"
		o=NN.predict([i/255 for i in C])
		l["text"]=COLORS[o.index(max(o))]
		tk.after(500,g)
	g()
	l.pack()
	tk.mainloop()



# train_mode()
test_mode()
