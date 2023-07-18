
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.patches import Ellipse
import pickle
import math
import time
from lion_pytorch import Lion
torch.set_default_dtype(torch.float64)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CramerUnit(nn.Module):
    def __init__(self):
        super().__init__()
        # antiderivative of Phi (CDF of N(0,1)) is F.gelu(z) + sqrt(1/(2*pi)) * exp(-z**2/2)
        self.unit = lambda z: 2 * F.gelu(z) - z + 0.7978845608028654 * torch.exp(-z**2/2)
    def forward(self, m1, s1, m2, s2):
        v = torch.sqrt(s1**2 + s2**2 + 1e-20) 
        return v*self.unit((m1-m2)/v)
        # This function is 1-Lipschitz
cramer = CramerUnit()

def calc_loss(pi, mu, sigma, pi2, mu2, sigma2):
    # pi, mu, sigma: (batch_size, n)
    # pi2, mu2, sigma2: (batch_size, n2)
    n=pi.size(-1)
    n2=pi2.size(-1)

    def r(T, d, N):
        return T.unsqueeze(d).repeat(
            [1,N,1] if d==-2 else [1,1,N] if d==-1 else None
        )

    I1 = r(pi, -1,n ) * r(pi, -2,n ) * cramer(r(mu, -1,n ), r(sigma, -1,n ), r(mu, -2,n ), r(sigma, -2,n ))
    I2 = r(pi2,-1,n2) * r(pi2,-2,n2) * cramer(r(mu2,-1,n2), r(sigma2,-1,n2), r(mu2,-2,n2), r(sigma2,-2,n2))
    I3 = r(pi, -1,n2) * r(pi2,-2,n ) * cramer(r(mu, -1,n2), r(sigma, -1,n2), r(mu2,-2,n ), r(sigma2,-2,n ))
    #I4 = r(pi2,-1,n ) * r(pi, -2,n2) * cramer(r(mu2,-1,n ), r(sigma2,-1,n ), r(mu, -2,n2), r(sigma, -2,n2))

    # We don't want sigmas to be negative. That has no mathematical meaning.
    penalty_item = nn.ReLU()(-10*sigma).sum()
    #print(I3.sum(), I4.sum())
    loss = (I3.sum() - I1.sum() + I3.sum() - I2.sum() + penalty_item) # / batch_size
    return loss


class GaussianMixture(nn.Module):
    def __init__(self, n, m, p=None, mu=None, s=None, target=False):
        # m: dim
        # n: number of gaussians
        # p: proportions
        # mu: expectation
        # s: variance matrix params
        super().__init__() 
        self.n = n
        self.m = m
        self.p = nn.Parameter(0.05*torch.randn((n,)) if p is None else(
            torch.zeros((n,)) if isinstance(p, int) or isinstance(p, float) else torch.DoubleTensor(p)
        ))
        self.mu = nn.Parameter(1*torch.randn((n, m)) if mu is None else(
            torch.zeros((n,m)) if isinstance(mu, int) or isinstance(mu, float) else torch.DoubleTensor(mu)
        ))
        self.s = nn.Parameter(0.2*torch.randn((n, m, m))/m if s is None else (
            torch.zeros((n, m, m)) if (isinstance(s, int) or isinstance(s, float)) and s==0 else (
                torch.eye(m).unsqueeze(0).repeat([n, 1, 1]) if (
                    isinstance(s, int) or isinstance(s, float)) and s==1 
                    else torch.DoubleTensor(s)
            )
        ))
        if target:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def forward(self, x: torch.Tensor):
        # (batch_size, self.m) -> (batch_size, self.n, self.m, 1)
        batch_size = x.shape[0]
        x = x.unsqueeze(1).repeat([1, self.n, 1]).unsqueeze(-1)

        # pi to (batch_size, self.n)
        pis = (nn.Softmax(dim=-1)(self.p)).unsqueeze(0).repeat([batch_size, 1])
        
        # mu_temp to (batch_size, self.n, 1, self.m)
        mu_temp = self.mu.unsqueeze(0).repeat([batch_size, 1, 1]).unsqueeze(-2)

        # s_temp to (batch_size, self.n, self.m, self.m)
        s_temp = self.s.unsqueeze(0).repeat([batch_size, 1, 1, 1])

        # mus (batch_size, self.n, 1, 1) -> (batch_size, self.n)
        mus = torch.matmul(mu_temp, x).squeeze(-1).squeeze(-1)

        s_temp = torch.matmul(s_temp, x)
        sigmas = torch.matmul(s_temp.mT, s_temp).squeeze(-1).squeeze(-1)
        sigmas = torch.sqrt(sigmas)

        return pis, mus, sigmas

    def forward2(self, x):
        # x is of shape (m,), an m-dim vector
        # please implement the (negative) log-likelihood here
        # using logsumexp, softmax or other functions

        # Compute the log-probabilities of each Gaussian component
        log_p = torch.log_softmax(self.p, dim=0) # shape (n,)
        log_gauss = torch.empty(self.n) # shape (n,)
        for i in range(self.n):
            # Compute the Mahalanobis distance between x and mu[i]
            d_temp = torch.matmul(x - self.mu[i], self.s_inv[i]) # sigma = s^T s, sigma^-1 = s_inv s_inv^T
            nrm = torch.linalg.vector_norm(d_temp) # shape ()
            _, absdet = torch.slogdet(self.s_inv[i])
            # Compute the log-density of the multivariate normal distribution
            log_norm = -0.91893853320467274 * self.m + absdet # shape ()
            log_gauss[i] = log_norm - nrm # shape ()

        # Compute the log-likelihood using logsumexp
        log_likelihood = torch.logsumexp(log_p + log_gauss, dim=0) # shape ()

        # Return the negative log-likelihood

        return -log_likelihood

    def transfer_to_s_inv(self):
        # Compute the inverse of each s matrix and detach from the computation graph
        s_inv_list = [torch.inverse(s).detach() for s in self.s] # list of tensors of shape (m, m)
        # Stack the inverse matrices into a tensor of shape (n, m, m)
        s_inv_tensor = torch.stack(s_inv_list, dim=0) # tensor of shape (n, m, m)
        # Create a new parameter for s_inv and assign it to self.s_inv
        self.s_inv = nn.Parameter(s_inv_tensor) # parameter of shape (n, m, m)
        self.s = None
    
    
def generate_unit_vectors(batch_size, m):
    if m==2:
        phi = 2*math.pi/batch_size
        u=random.uniform(0,2*math.pi)
        s=u+phi*torch.arange(batch_size)
        return torch.stack([torch.cos(s), torch.sin(s)], dim=-1)

data=np.loadtxt('./data/ring-line-square.txt',delimiter=' ')
data-=data.mean(axis=0)
print(data.shape)

target_model = GaussianMixture(data.shape[0], data.shape[1], p=0, mu=data, s=0, target=True)
target_model.train(False)

def plot_cov_ellipse(pos,cov, nstd=2, ax=None, c='r'):    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, facecolor=c, edgecolor=c, linewidth=2,zorder=25, alpha = 0.2)

    ax.add_artist(ellip)
    return ellip



n=10
m=2

def experiment(seed, SC2):
    set_seed(seed)
    model=GaussianMixture(n,m)

    plt.figure(figsize=(6,5))
    plt.plot(data[:,0],data[:,1],'x',zorder=0)
    for k in range(n):
        mu = model.mu[k].detach().numpy()
        sigma = torch.matmul(model.s[k].mT, model.s[k]).detach().numpy()
        plot_cov_ellipse(mu, sigma, nstd=2, ax=None)
        plt.scatter(mu[0],mu[1],c='r',marker='.',linewidth=0.7,zorder=5)
    plt.savefig(f"init_{seed}.jpg")

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='SlicedCramerGMM',
                    comment='Sliced Cramer GMM learning process')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    if SC2:
        opt = Lion([
            {'params': model.p, 'lr':  5e-6},
            {'params': model.mu, 'lr': 0.02},
            {'params': model.s, 'lr':  3e-3},
        ])
        batch_size=7
        losses=[]
        with writer.saving(plt.figure(figsize=(20.2, 6)), f"SlicedCramerLearning_{seed}.mp4", 100):
            for i in range(1200):
                if i % 10 == 0: print(i)
                opt.zero_grad()
                rand_vecs = generate_unit_vectors(batch_size, model.m)
                pi, mu, sigma = model.forward(rand_vecs)
                pi2, mu2, sigma2 = target_model.forward(rand_vecs)
                loss = calc_loss(pi, mu, sigma, pi2, mu2, sigma2)
                loss.backward()
                losses.append(math.log(float(loss)))
                opt.step()
                
                if i%4 == 0:
                    plt.subplot(1,3,1)
                    plt.cla()
                    plt.plot(data[:,0],data[:,1],'x',zorder=0)
                    for k in range(n):
                        mu = model.mu[k].detach().numpy()
                        sigma = torch.matmul(model.s[k].mT, model.s[k]).detach().numpy()
                        plot_cov_ellipse(mu, sigma)
                        plt.scatter(mu[0],mu[1], c='r',marker='.',linewidth=0.7,zorder=5)
                    plt.title('Data & GMM',fontsize=15)

                    plt.subplot(1,3,2)
                    plt.cla()
                    circle = plt.Circle((0, 0), 1.0, color='k',linewidth=0.8, fill=False)
                    plt.gca().add_artist(circle)
                    vecs = rand_vecs.detach().numpy()
                    plt.plot(vecs[:,0],vecs[:,1],'o',markersize=10,linewidth=3,markerfacecolor='None',markeredgecolor='r',markeredgewidth=3)
                    plt.xlim(-1.2,1.2)
                    plt.ylim(-1.2,1.2)
                    plt.title('Random projections',fontsize=15)

                    plt.subplot(1,3,3)
                    plt.cla()
                    plt.plot(np.asarray(losses),c='k',linewidth=1)
                    plt.title('Log (Sliced Cramer 2-loss)',fontsize=15)
                    writer.grab_frame()
        
        plt.figure(figsize=(6,5))
        plt.plot(data[:,0],data[:,1],'x',zorder=0)
        for k in range(n):
            mu = model.mu[k].detach().numpy()
            sigma = torch.matmul(model.s[k].mT, model.s[k]).detach().numpy()
            plot_cov_ellipse(mu, sigma, nstd=2, ax=None)
            plt.scatter(mu[0],mu[1],c='r',marker='.',linewidth=0.7,zorder=5)

        plt.savefig(f"sc2_{seed}.jpg")

        plt.figure(figsize=(6,5))
        plt.plot(np.asarray(losses),c='k',linewidth=0.6)
        plt.title('Log (Sliced Cramer 2-loss)',fontsize=15)
        plt.savefig(f"sc2_{seed}_loss.jpg")


    model.transfer_to_s_inv()
    opt = Lion([
        {'params': model.p,     'lr': 5e-6},
        {'params': model.mu,    'lr': 0.02},
        {'params': model.s_inv, 'lr': 3e-3},
    ])
    batch_size=256
    losses=[]
    with writer.saving(plt.figure(figsize=(13.4, 6)), f"LikelihoodLearning_{seed}_{SC2}.mp4", 100):
        for i in range(200 if SC2 else 1200):
            if i % 10 == 0: print(i)
            opt.zero_grad()
            idxs = np.random.choice(len(data), batch_size)
            loss = torch.empty(1)
            for j in idxs:
                loss += model.forward2(torch.tensor(data[j]))
            loss /= 16
            if math.isnan(float(loss)) or loss >= 1000 or loss <= 0:
                losses.append(float('nan'))
            else:
                loss.backward()
                losses.append(float(loss))
                opt.step()
            
            if i%4 == 0:
                plt.subplot(1,2,1)
                plt.cla()
                plt.plot(data[:,0],data[:,1],'x',zorder=0)
                for k in range(n):
                    mu = model.mu[k].detach().numpy()
                    sigma = np.linalg.inv(torch.matmul(model.s_inv[k], model.s_inv[k].mT).detach().numpy())
                    plot_cov_ellipse(mu, sigma)
                    plt.scatter(mu[0],mu[1], c='r',marker='.',linewidth=0.7,zorder=5)
                plt.title('Data & GMM',fontsize=15)

                plt.subplot(1,2,2)
                plt.cla()
                plt.plot(np.asarray(losses),c='k',linewidth=1)
                plt.title('-Log(Likelihood)',fontsize=15)

                writer.grab_frame()
    
    plt.figure(figsize=(6,5))
    plt.plot(data[:,0],data[:,1],'x',zorder=0)
    for k in range(n):
        mu = model.mu[k].detach().numpy()
        sigma = np.linalg.inv(torch.matmul(model.s_inv[k], model.s_inv[k].mT).detach().numpy())
        plot_cov_ellipse(mu, sigma)
        plt.scatter(mu[0],mu[1], c='r',marker='.',linewidth=0.7,zorder=5)
    plt.savefig(f"nll_{seed}_{SC2}.jpg")

    plt.figure(figsize=(6,5))
    plt.plot(np.asarray(losses),c='k',linewidth=0.6)
    plt.title('-Log(Likelihood)',fontsize=15)
    plt.savefig(f"nll_{seed}_loss.jpg")

experiment(123, True)
