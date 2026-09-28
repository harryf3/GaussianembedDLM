"""Read-only CPU audit of saved DLM checkpoints. No training or checkpoint writes.
Run: python3 experiment_review/checkpoint_diagnostics.py
Architecture dimensions are inferred from tensors; heads/dropout follow notebook source.
These latter settings are not preserved in historical checkpoints.
"""
import ast, collections, json, math, pathlib, torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(4)
torch.manual_seed(1234)
ROOT=pathlib.Path(__file__).resolve().parent.parent
OUT=pathlib.Path(__file__).resolve().parent
text='\n\n'.join(''.join(p.read_text().replace('\r\n','\n').replace('\r','\n').splitlines(keepends=True)[7:]).lstrip('\n') for p in sorted((ROOT/'shakespeare-dataset-main/text').glob('*.txt')))
counts=collections.Counter(text)

class GaussianEmbedding(nn.Module):
 def __init__(self,V,D):
  super().__init__(); self.mu=nn.Parameter(torch.zeros(V,D)); self.raw_std=nn.Parameter(torch.zeros(V,D))
 @property
 def std(self): return F.softplus(self.raw_std)
 def sample(self,ids): return self.mu[ids]+self.std[ids]*torch.randn(*ids.shape,self.mu.shape[-1])
 def log_probs(self,x):
  prec=self.std.square().reciprocal()
  maha=x.square()@prec.T-2*x@(self.mu*prec).T+(self.mu.square()*prec).sum(-1)
  return F.log_softmax(-.5*maha-self.std.log().sum(-1),-1)

class Model(nn.Module):
 def __init__(self,state,is_dlm):
  super().__init__(); self.is_dlm=is_dlm
  key='embedding.weight' if is_dlm else 'embedding.mu'
  V,D=state[key].shape; self.D=D
  L=len({k.split('.')[2] for k in state if k.startswith('transformer.layers.')})
  H=8 if D==128 else 4
  self.embedding=nn.Embedding(V,D) if is_dlm else GaussianEmbedding(V,D)
  if 'embedding.token_probs' in state: self.embedding.register_buffer('token_probs',state['embedding.token_probs'])
  self.position=nn.Embedding(*state['position.weight'].shape)
  self.register_buffer('time_frequencies',state['time_frequencies'])
  self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(D,H,state['transformer.layers.0.linear1.weight'].shape[0],.1,batch_first=True,norm_first=False,activation='gelu'),L)
  self.final_norm=nn.LayerNorm(D)
  self.head_name='eps_head' if 'eps_head.weight' in state else 'x0_head'
  setattr(self,self.head_name,nn.Linear(D,D))
  if is_dlm: self.rounding_bias=nn.Parameter(torch.zeros(V))
  self.load_state_dict(state,strict=True); self.eval()
 def forward(self,x,t):
  angles=t[:,None]*self.time_frequencies
  h=x+self.position(torch.arange(x.size(1)))[None]+torch.cat([angles.sin(),angles.cos()],-1)[:,None]
  return getattr(self,self.head_name)(self.final_norm(self.transformer(h)))
 def logits(self,x): return F.linear(x,self.embedding.weight,self.rounding_bias) if self.is_dlm else self.embedding.log_probs(x)

def main():
 results=[]
 for file in sorted((ROOT/'checkpoints').glob('*.pt')):
  cp=torch.load(file,map_location='cpu',weights_only=True); state=cp['model']; is_dlm='embedding.weight' in state
  m=Model(state,is_dlm); chars=cp['chars']; V=len(chars); D=m.D
  p=torch.tensor([counts[c] for c in chars],dtype=torch.float); p/=p.sum()
  tok=torch.tensor([chars.index(c) for c in text],dtype=torch.long)
  blocks=tok[:len(tok)//64*64].view(-1,64)
  ids=blocks[torch.randperm(len(blocks),generator=torch.Generator().manual_seed(1234))[:32]]
  mean=m.embedding.weight.detach() if is_dlm else m.embedding.mu.detach()
  var=torch.full_like(mean,cp['config']['embedding_std']**2) if is_dlm else m.embedding.std.detach().square()
  mu=(p[:,None]*mean).sum(0); centered=mean-mu
  C=(p[:,None]*centered).T@centered+torch.diag((p[:,None]*var).sum(0))
  steps=sorted(set(int(v['step']) for v in cp.get('optimizer',{}).get('state',{}).values() if 'step' in v))
  T=cp['config']['diffusion_steps'] if is_dlm else 1000
  if is_dlm:
   prog=torch.arange(T+1,dtype=torch.float64)/T; curve=1-(prog+cp['config']['schedule_offset']).sqrt()
   beta=(1-curve[1:]/curve[:-1]).clamp(max=.999); abar=torch.cat([torch.ones(1),torch.cumprod(1-beta,0)]).float()
  else: abar=torch.cat([torch.ones(1),torch.cumprod(1-torch.linspace(1e-4,.02,T),0)])
  rec={'checkpoint':file.name,'parameters':sum(x.numel() for x in m.parameters()),'dim':D,'optimizer_steps':steps,'head_assumption':8 if D==128 else 4,
       'weighted_mean_rms':mu.square().mean().sqrt().item(),'mixture_variance':C.diag().mean().item(), 'between_token_variance':(p[:,None]*centered.square()).sum(0).mean().item(),'within_token_variance':(p[:,None]*var).sum(0).mean().item(),
       'covariance_error':(torch.linalg.matrix_norm(C-torch.eye(D))/math.sqrt(D)).item(), 'eigenvalue_min':torch.linalg.eigvalsh(C).min().item(),'eigenvalue_max':torch.linalg.eigvalsh(C).max().item()}
  with torch.no_grad():
   x0=mean[ids]+var[ids].sqrt()*torch.randn(*ids.shape,D)
   rec['clean_rounding_accuracy']=(m.logits(x0).argmax(-1)==ids).float().mean().item()
   rec['mean_decodes_as']=chars[m.logits(mu).argmax().item()]
   rec['zero_decodes_as']=chars[m.logits(torch.zeros(D)).argmax().item()]
   baseline=(x0-mu).square().mean().item(); rec['batch_constant_mean_mse']=baseline
   rec['timesteps']=[]
   for tval in ([1,10,100,250,500,750,1000] if not is_dlm else [1,10,100,250,500,1000,1500,2000]):
    a=abar[tval]; noise=torch.randn_like(x0); xt=a.sqrt()*x0+(1-a).sqrt()*noise
    pred=m(xt,torch.full((len(ids),),tval))
    if m.head_name=='eps_head': pred=(xt-(1-a).sqrt()*pred)/a.sqrt()
    logit=m.logits(pred); decoded=logit.argmax(-1)
    # Analytic, independent-token posterior using the learned Gaussian codebook and corpus frequencies.
    vt=a*var+(1-a); mt=a.sqrt()*mean
    logp=-.5*(xt.square()@(1/vt).T-2*xt@(mt/vt).T+(mt.square()/vt).sum(-1)+vt.log().sum(-1))+p.log()
    prob=logp.softmax(-1)
    conditional=prob@(mean*(1-a)/vt)+xt*(prob@(a.sqrt()*var/vt))
    row={'t':tval,'alpha_bar':a.item(),'mse':(pred-x0).square().mean().item(),'accuracy':(decoded==ids).float().mean().item(),'output_variance':pred.flatten(0,1).var(dim=0,correction=0).mean().item(),'output_norm':pred.square().mean().item(), 'constant_mean_mse':baseline,'analytic_tokenwise_mse':(conditional-x0).square().mean().item(),'analytic_tokenwise_accuracy':(prob.argmax(-1)==ids).float().mean().item(),'decoder_ce':F.cross_entropy(logit.flatten(0,1),ids.flatten()).item(),'top_output':chars[decoded.flatten().bincount(minlength=V).argmax().item()]}
    rec['timesteps'].append(row)
  print(json.dumps(rec),flush=True)
  results.append(rec)
 (OUT/'checkpoint_metrics.json').write_text(json.dumps(results,indent=2))

if __name__ == "__main__":
 main()
