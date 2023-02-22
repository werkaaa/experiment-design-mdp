import torch
import cvxpy as cp
import numpy as np
from scipy.linalg import null_space, orth

def constraint_operator(a,b,c,emb,t):
	n = t.size()[0]
	Phi = emb.embed(t)
	m = Phi.size()[1]
	du_dt = emb.derivative_1(t)[0,:,:].T
	zeros = torch.zeros(n,m)
	D = torch.vstack([torch.hstack([-du_dt - a*Phi, zeros]),torch.hstack([b*Phi, -c*Phi - du_dt])])
	t0 = torch.Tensor([[0.]]).double()

	constraints_full = torch.vstack([D,torch.hstack([torch.zeros(size = (1,m)).double(),emb.embed(t0)])])

	return D,constraints_full


def fit_pharmaco(a,b,c,t_point,y_point,GP,t,emb,lam = 1. ,sigma = 1.):
	D,constraints_full = constraint_operator(a,b,c,emb,t)

	# this is the classical fit
	GP.fit_gp(t_point, y_point)
	mu, std = GP.mean_std(t)
	mu = mu.detach()
	std = std.detach()
	theta_mu, theta_var = GP.theta_mean(var=True)
	Phi = emb.embed(t)

	# constrained fit
	eps = 1e-1
	Q = emb.embed(t_point)
	m = Q.size()[1]

	theta1 = cp.Variable(m) # stomach
	theta2 = cp.Variable(m) # blood
	t0 = torch.Tensor([[0.]]).double()

	objective = cp.Minimize(
		cp.sum_squares(
			Q.detach().numpy() @ theta2 - y_point.detach().numpy())
		+ lam * sigma ** 2 * cp.sum_squares(theta2)+ lam * sigma ** 2 * cp.sum_squares(theta1))

	constraints = []
	constraints += [(emb.embed(t0)@theta1 - 1.)**2<=10e-5]
	constraints += [emb.embed(t0)@theta2 == 0.]
	constraints += [D.detach().numpy() @ cp.hstack([theta1,theta2]) <= np.ones(D.size()[0]) * eps ** 2]
	constraints += [D.detach().numpy() @ cp.hstack([theta1,theta2]) >= -np.ones(D.size()[0]) * eps ** 2]
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.MOSEK, verbose=False)

	theta_constrained = torch.from_numpy(theta1.value).view(-1, 1)
	theta_constrained2 = torch.from_numpy(theta2.value).view(-1, 1)

	mu_constrained = Phi @ theta_constrained
	mu_constrained = mu_constrained.detach()

	mu_constrained2 = Phi @ theta_constrained2
	mu_constrained2 = mu_constrained2.detach()

	#plt.plot(t.detach(),mu_constrained,lw= 2, alpha =0.5, color = 'tab:blue')
	#plt.plot(t.detach(),mu_constrained2, lw= 2, alpha =0.5, color = 'tab:purple')

	# Uncertainty
	constraints_full = torch.vstack([D,torch.hstack([emb.embed(t0),torch.zeros(size = (1,m)).double()])])
	C = torch.from_numpy(null_space(constraints_full.detach(), rcond=1e-10))
	P_C = C @ torch.pinverse(C, rcond=1e-15)

	# uncertainty over the trajectory
	zeros = torch.zeros(size = (m,m)).double()
	variance_full = torch.vstack([torch.hstack([theta_var,zeros]),torch.hstack([zeros,theta_var])])
	Phi_double = torch.hstack([Phi,Phi])
	covar_traj = Phi_double @ P_C @ variance_full @ P_C.T @ Phi_double.T
	std_traj = torch.sqrt(torch.diag(covar_traj)).detach()

	return (theta_constrained,theta_constrained2,std_traj)





def fit_pharmaco_perturbed(a,b,c,t_point,y_point,GP,t,emb,emb_patient,lam = 1. ,sigma = 1.):
	D,constraints_full = constraint_operator(a,b,c,emb,t)

	# this is the classical fit
	GP.fit_gp(t_point, y_point)
	mu, std = GP.mean_std(t)
	mu = mu.detach()
	std = std.detach()
	theta_mu, theta_var = GP.theta_mean(var=True)
	Phi = emb.embed(t)

	# constrained fit
	eps = 1e-1
	Q = emb.embed(t_point)
	m = Q.size()[1]
	Qp = emb_patient.embed(t_point)

	theta1 = cp.Variable(m) # stomach
	theta2 = cp.Variable(m) # blood
	theta3 = cp.Variable(m) # patient specifics

	t0 = torch.Tensor([[0.]]).double()

	objective = cp.Minimize(
		cp.sum_squares(
			Q.detach().numpy() @ theta2 + Qp.detach().numpy() @ theta3 - y_point.detach().numpy())
		+ lam * sigma ** 2 * cp.sum_squares(theta2)+ lam * sigma ** 2 * cp.sum_squares(theta1) + lam * sigma ** 2 * cp.sum_squares(theta3))

	constraints = []
	constraints += [(emb.embed(t0)@theta1 - 1.)**2<=10e-5]
	constraints += [emb.embed(t0)@theta2 == 0.]
	constraints += [D.detach().numpy() @ cp.hstack([theta1,theta2]) <= np.ones(D.size()[0]) * eps ** 2]
	constraints += [D.detach().numpy() @ cp.hstack([theta1,theta2]) >= -np.ones(D.size()[0]) * eps ** 2]
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.MOSEK, verbose=False)

	theta_constrained = torch.from_numpy(theta1.value).view(-1, 1)
	theta_constrained2 = torch.from_numpy(theta2.value).view(-1, 1)
	theta_constrained3 = torch.from_numpy(theta3.value).view(-1, 1)

	mu_constrained = Phi @ theta_constrained
	mu_constrained = mu_constrained.detach()

	mu_constrained2 = Phi @ theta_constrained2
	mu_constrained2 = mu_constrained2.detach()

	mu_constrained3 = Phi @ theta_constrained3
	mu_constrained3 = mu_constrained3.detach()

	#plt.plot(t.detach(),mu_constrained,lw= 2, alpha =0.5, color = 'tab:blue')
	#plt.plot(t.detach(),mu_constrained2, lw= 2, alpha =0.5, color = 'tab:purple')

	# Uncertainty
	constraints_full = torch.vstack([D,torch.hstack([emb.embed(t0),torch.zeros(size = (1,m)).double()])])
	C = torch.from_numpy(null_space(constraints_full.detach(), rcond=1e-10))
	P_C = C @ torch.pinverse(C, rcond=1e-15)

	# uncertainty over the trajectory
	zeros = torch.zeros(size = (m,m)).double()
	variance_full = torch.vstack([torch.hstack([theta_var,zeros]),torch.hstack([zeros,theta_var])])
	Phi_double = torch.hstack([Phi,Phi])
	covar_traj = Phi_double @ P_C @ variance_full @ P_C.T @ Phi_double.T
	std_traj = torch.sqrt(torch.diag(covar_traj)).detach()

	return (theta_constrained,theta_constrained2,theta_constrained3, std_traj)





























def fit_pharmaco_simplified(a,b,c,t_point,y_point,GP,t,emb,lam = 1. ,sigma = 1.):
	D,constraints_full = constraint_operator(a,b,c,emb,t)
	# this is the classical fit
	Phi = emb.embed(t)
	# constrained fit
	eps = 1e-1
	Q = emb.embed(t_point)
	m = Q.size()[1]

	theta1 = cp.Variable(m) # stomach
	theta2 = cp.Variable(m) # blood
	t0 = torch.Tensor([[0.]]).double()

	objective = cp.Minimize(
		cp.sum_squares(
			Q.detach().numpy() @ theta2 - y_point.detach().numpy())
		+ lam * sigma ** 2 * cp.sum_squares(theta2)+ lam * sigma ** 2 * cp.sum_squares(theta1))

	newD = torch.from_numpy(orth(D.detach().numpy().T,rcond = 1e-5)).T
	constraints = []
	constraints += [(emb.embed(t0)@theta1 - 1.)**2<=10e-5]
	constraints += [emb.embed(t0)@theta2 == 0.]
	constraints += [newD.detach().numpy() @ cp.hstack([theta1,theta2]) <= np.ones(newD.size()[0]) * eps ** 2]
	constraints += [newD.detach().numpy() @ cp.hstack([theta1,theta2]) >= -np.ones(newD.size()[0]) * eps ** 2]
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.MOSEK, verbose=False)
	theta_constrained2 = torch.from_numpy(theta2.value).view(-1, 1)
	return None,theta_constrained2,None

def fit_pharmaco_closed_form(a,b,c,t_point,y_point,t,emb,lam = 1. ,sigma = 1.):
	D, constraints_full = constraint_operator(a, b, c, emb, t)
	Phi = emb.embed(t)
	t0 = torch.Tensor([[0.]]).double()
	p = 100
	# constrained fit
	Q = emb.embed(t_point)
	m = Q.size()[1]
	N = Q.size()[0]

	# constraints space
	eval_1 = torch.hstack([torch.zeros(size=(1, m)).double(), emb.embed(t0)]) # blood initial conditions
	eval_2 = torch.hstack([emb.embed(t0), torch.zeros(size=(1, m)).double()]) # stomach initial conditions
	constraints_full = torch.vstack([D, eval_1])
	X = torch.hstack([torch.zeros(size = (N,m)).double(),Q])

	I = torch.eye(2*m,2*m).double()

	M1 = X.T@X + I *lam *sigma**2
	v1 = X.T@y_point
	M2 = constraints_full
	v2 = torch.zeros(size = (constraints_full.size()[0],1)).double().view(-1)
	M3 = eval_2
	v3 = torch.ones(size =(1,1)).double().view(-1)

	M = torch.vstack([M1,M2,M3])
	v = torch.hstack([v1,v2,v3])
	thetas = torch.linalg.lstsq(M,v)[0]

	theta_constrained2 = thetas[m:]
	mu_constrained2 = Phi @ theta_constrained2
	mu_constrained2 = mu_constrained2.detach()
	return mu_constrained2
