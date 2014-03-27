####################
# Code to calculate and plotstationary 
# distributions in public goods games with punishment
# imitation and mutation as described in
# 'Corruption Drives the Emergence of Civil Society'
# Abdulah et al http://arxiv.org/abs/1307.6646
# A. Rutherford 2013
###################

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.linalg
import sys,pprint
from matplotlib import rc
import cPickle as pickle

rc('font',size=24)

v='POOL'
w='PEER'

x='COOPERATE'
y='DEFECT'
z='LONER'

c_='CORRUPT'
h='HYBRID'

M=100
N=5
s=0.001
r=3.0
c=1
sigma=1.0

G=0.7
B=1000.0
# Pool punishment

gamma=0.7
beta=0.7
# Peer punishment

K=0.5
# Bribe

payoffDict={}

strategies=['v','w','x','y','z','c','h']
#################
def p_h(V,W,X,Y,Z,C,H):
#################
	if not V+W+X+Y+Z+C+H==M:
		print 'Ph: DONT SUM TO M!!!!'
		sys.exit(1)

	returnVal=0.0

	returnVal+=r*((M-Z-Y-C)/float(M-Z))-1
	returnVal*=c
	returnVal-=G
	
	returnVal*=(1.-((scipy.special.binom(Z,N-1))/(scipy.special.binom(M-1,N-1))))
	# PGG payoff-pool payment

	returnVal+=sigma*(scipy.special.binom(Z,N-1)/scipy.special.binom(M-1,N-1))
	# Chance of non cooperation

	returnVal-=(((N-1)*(Y+C))/float(M-1))*gamma
	# Minus peer punishment
	
	if M-Y-2>-1:
		returnVal-=((((N-1)*(X))/float(M-1))*gamma)*(1.0-((scipy.special.binom(M-Y-2,N-2))/float(scipy.special.binom(M-2,N-2))))
	# Minus second-order peer punishment

	return returnVal

#################
def p_v(V,W,X,Y,Z,C,H):
#################
	if not V+W+X+Y+Z+C+H==M:
		print 'Pv: DONT SUM TO M!!!!'
		sys.exit(1)

	returnVal=0.0

	returnVal+=r*((M-Z-Y-C)/float(M-Z))-1
	returnVal*=c
	returnVal-=G
	returnVal*=(1.-((scipy.special.binom(Z,N-1))/(scipy.special.binom(M-1,N-1))))
	# PGG payoff-pool payment

	returnVal+=sigma*(scipy.special.binom(Z,N-1)/scipy.special.binom(M-1,N-1))
	# Chance of non cooperation

	return returnVal
#################
def p_w(V,W,X,Y,Z,C,H):
#################
	if not V+W+X+Y+Z+C+H==M:
		print 'Pw: DONT SUM TO M!!!!'
		sys.exit(1)

	returnVal=0.0

	returnVal+=r*((M-Z-Y-C)/float(M-Z))-1
	returnVal*=c*(1.-((scipy.special.binom(Z,N-1))/(scipy.special.binom(M-1,N-1))))
	# PGG payoff

	returnVal+=sigma*(scipy.special.binom(Z,N-1)/scipy.special.binom(M-1,N-1))
	# Chance of non cooperation

	returnVal-=(((N-1)*(Y+C))/float(M-1))*gamma
	# Minus peer punishment

	returnVal-=(B*(N-1)*(V+H))/float(M-1)
	# Second order pool punishment term
	# Peer punishers get pool punished

	if M-Y-2>-1:
		returnVal-=((((N-1)*(X))/float(M-1))*gamma)*(1.0-((scipy.special.binom(M-Y-2,N-2))/float(scipy.special.binom(M-2,N-2))))
	# Minus second-order peer punishment
	return returnVal

#################
def p_x(V,W,X,Y,Z,C,H):
#################
	if not V+W+X+Y+Z+C+H==M:
		print 'Px: DONT SUM TO M!!!!'
		sys.exit(1)

	returnVal=0.0

	returnVal+=r*((M-Z-Y-C)/float(M-Z))-1
	returnVal*=c*(1.-((scipy.special.binom(Z,N-1))/(scipy.special.binom(M-1,N-1))))
	# PGG payoff

	returnVal+=sigma*(scipy.special.binom(Z,N-1)/scipy.special.binom(M-1,N-1))
	# Chance of non cooperation

	returnVal-=(B*(N-1)*(V+H))/float(M-1)
	# Second order pool punishment term

	if M-Y-2>-1:
		returnVal-=(((N-1)*(W+H))/float(M-1))*beta*(1-((scipy.special.binom(M-Y-2,N-2)/(scipy.special.binom(M-2,N-2)))))
	# Second order peer punishment

	return returnVal

#################
def p_c(V,W,X,Y,Z,C,H):
#################

	if not V+W+X+Y+Z+C+H==M:
		print 'Px: DONT SUM TO M!!!!'
		sys.exit(1)

	returnVal=0.0

	returnVal+=c*r*((M-Z-Y-C)/float(M-Z))-(K*G)
	returnVal*=(1.-((scipy.special.binom(Z,N-1))/(scipy.special.binom(M-1,N-1))))
	# PGG payoff minus bribe

	returnVal+=sigma*(scipy.special.binom(Z,N-1)/scipy.special.binom(M-1,N-1))
	# Chance of non cooperation

	returnVal-=((N-1)*(W+H)*beta)/float(M-1)
	# Peer punishment term

	return returnVal
#################
def p_y(V,W,X,Y,Z,C,H):
#################
	if not V+W+X+Y+Z+C+H==M:
		print 'Px: DONT SUM TO M!!!!'
		sys.exit(1)

	returnVal=0.0

	returnVal+=c*r*((M-Z-Y-C)/float(M-Z))
	returnVal*=(1.-((scipy.special.binom(Z,N-1))/(scipy.special.binom(M-1,N-1))))
	# PGG payoff

	returnVal+=sigma*(scipy.special.binom(Z,N-1)/scipy.special.binom(M-1,N-1))
	# Chance of non cooperation

	returnVal-=(B*(N-1)*(V+H))/float(M-1)
	# Pool punishment term

	returnVal-=((N-1)*(W+H)*beta)/float(M-1)
	# Peer punishment term

	return returnVal
#################
def p_z(V,W,X,Y,Z,C,H):
#################
	return sigma

#################
def getNumbers(arg0,arg1,n):
#################

	vv=ww=xx=yy=zz=cc=hh=0

	if arg0=='x':xx=M-n
	elif arg0=='y':yy=M-n
	elif arg0=='z':zz=M-n
	elif arg0=='w':ww=M-n
	elif arg0=='v':vv=M-n
	elif arg0=='c':cc=M-n
	elif arg0=='h':hh=M-n

	if arg1=='x':xx=n
	elif arg1=='y':yy=n
	elif arg1=='z':zz=n
	elif arg1=='v':vv=n
	elif arg1=='w':ww=n
	elif arg1=='c':cc=n
	elif arg1=='h':hh=n

	return vv,ww,xx,yy,zz,cc,hh

#################
def rho(*args):
#################
# Pass strategy names as args
	global payoffDict

	if not len(args)==2:
		print 'NEED 2 ARGS!!!!'
		sys.exit(1)

	strat0=payoffDict[args[0]]
	strat1=payoffDict[args[1]]
	# Look up payoff functions from dictionary i.e. 'x' points to p_x

	returnVal=1.0

	for q in range(1,M):
		temp=0
		for n in range(1,q+1):
			vv,ww,xx,yy,zz,cc,hh=getNumbers(args[0],args[1],n)
			# Get number arguments for payoffs
			# Depends on which fixation probability i.e. xy,zx

			temp+=strat0(vv,ww,xx,yy,zz,cc,hh)-strat1(vv,ww,xx,yy,zz,cc,hh)

		returnVal+=np.exp(temp*s)

	return 1.0/returnVal

#################
def getTransitionMatrix(strategies):
#################
	d=len(strategies)

	matrix=np.zeros(shape=(d,d))

	for i in range(d):
		for j in range(d):
			if i==j:
				matrix[i,j]=1-np.sum([(1.0/(d-1))*rho(strategies[i],noti) for noti in strategies if not noti==strategies[i]])
			else:
				matrix[i,j]=(1.0/(d-1)*rho(strategies[i],strategies[j]))
	return np.array(matrix)

#################
def getStatDist(mat):
#################
# Get stationary dist by power method

	tempDist=np.array([0,0,0,1,0,0,0])
	for i in xrange(100000):
		tempDist=(np.dot(tempDist,mat))
		tempDist/=np.sum(tempDist)
	print tempDist
	return tempDist

#################
def main():
#################
	readIn=False
	if len(sys.argv)>1:
		readIn=True
	# if values already calculated, call script with an argument to simply
	# replot values already saved i.e. python transition_x_y_z_c_h.py 0
	if not readIn:

		dumpFile=open('DUMP.dat','w')
		
		payoffDict['v']=p_v
		payoffDict['w']=p_w
		payoffDict['x']=p_x
		payoffDict['y']=p_y
		payoffDict['z']=p_z
		payoffDict['c']=p_c
		payoffDict['h']=p_h
		# Look up payoff functions

		vVals=[]
		wVals=[]
		xVals=[]
		yVals=[]
		zVals=[]
		cVals=[]
		hVals=[]
		sVals=[0.1,0.15,0.16,0.17,0.18,0.2,0.5,0.75,1,2,3,4,5,10,12,15,17,18,25,30,35,40,45,50,60,70,80,100]
		sVals.sort()

		global B
		global G
		global s

		s=10000.0

		for val in sVals:
			B=val

			print 'G,B=>',G,B

			mat=getTransitionMatrix(strategies)

			pprint.pprint(mat)

			dist=getStatDist(mat)

			vVals.append(dist[0])
			wVals.append(dist[1])
			xVals.append(dist[2])
			yVals.append(dist[3])
			zVals.append(dist[4])
			cVals.append(dist[5])
			hVals.append(dist[6])

		pickle.dump(vVals,dumpFile)
		pickle.dump(wVals,dumpFile)
		pickle.dump(xVals,dumpFile)
		pickle.dump(yVals,dumpFile)
		pickle.dump(zVals,dumpFile)
		pickle.dump(cVals,dumpFile)
		pickle.dump(hVals,dumpFile)
		pickle.dump(sVals,dumpFile)

	else:
		dumpFile=open('DUMP.dat','r')

		vVals=pickle.load(dumpFile)
		wVals=pickle.load(dumpFile)
		xVals=pickle.load(dumpFile)
		yVals=pickle.load(dumpFile)
		zVals=pickle.load(dumpFile)
		cVals=pickle.load(dumpFile)
		hVals=pickle.load(dumpFile)
		sVals=pickle.load(dumpFile)

	xVals=[xxx+hhh for xxx,hhh in zip(xVals,hVals)]
	wVals=[www+xxx for www,xxx in zip(wVals,xVals)]
	vVals=[vvv+www for vvv,www in zip(vVals,wVals)]
	zVals=[zzz+vvv for zzz,vvv in zip(zVals,vVals)]
	cVals=[ccc+zzz for ccc,zzz in zip(cVals,zVals)]
	yVals=[yyy+ccc for yyy,ccc in zip(yVals,cVals)]
	# Stack values

	fig=plt.figure()
	ax=fig.add_subplot(111)

	ax.fill_between(sVals,hVals,0,color=(0.0,0.0,0.0,0.3),linewidth=0.0)
	ax.fill_between(sVals,xVals,hVals,color=(0,0,0,0.6),linewidth=0.0)
	ax.fill_between(sVals,wVals,xVals,color='blue',alpha=0.4,linewidth=0.0)
	ax.fill_between(sVals,vVals,wVals,color='blue',alpha=0.75,linewidth=0.0)
	ax.fill_between(sVals,zVals,vVals,color='black',alpha=0.8,linewidth=0.0)
	ax.fill_between(sVals,cVals,zVals,color='red',alpha=0.7,linewidth=0.0)
	ax.fill_between(sVals,yVals,cVals,color='red',alpha=0.5,linewidth=0.0)
	plt.xscale('log')
	
	plt.xlabel('Punishment Strength (B)')
	plt.ylabel('Proportion')

	plt.ylim(0,1.0)
	plt.xlim(min(sVals),max(sVals))
	# Fix axis scale

	plt.axvline((G*(M-1))/float(N-1),linestyle='--',color='k',linewidth=2)
	plt.axvline((G)/float(N-1),linestyle='--',color='k',linewidth=2)

	plt.annotate(r'$B=\frac{G}{N-1}$',xy=(0.015+0.175,0.2),xycoords='data',rotation='horizontal',size=30)
	plt.annotate(r'$B=\frac{M-1}{N-1}G$',xy=(0.4+G*((M-1)/float(N-1)),0.2),xycoords='data',rotation='horizontal',size=30)
	# Annotate discontinuities
	
	ax2=plt.twinx()

	colours=[(1,0,0,0.5),(1,0,0,0.7),(0,0,0,0.8),(0,0,1,0.75),(0,0,1,0.4),(0,0,0,0.6),(0.0,0.0,0.0,0.3)]
	colours.reverse()

	plt.yticks([0.1+(i*(1.0/7.0)) for i in range(7)],['-' for i in range(7)])

	[i.set_color(colours[ii]) for ii,i in enumerate(plt.gca().get_yticklabels())]

	plt.yticks([0.1+(i*(1.0/7.0)) for i in range(7)],[h,x,w,v,z,c_,y],rotation=45)
	# Fix ticks and right hand axis labels

	plt.savefig('PLOT.png')

	plt.show()

if __name__=='__main__':
	main()

