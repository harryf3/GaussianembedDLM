"""Paired CPU diagnostic. Permute batches independently at each sequence position.
Each token retains its own noisy vector and position; neighboring context is randomized.
"""
from checkpoint_diagnostics import ROOT, OUT, Model, text, torch, json
all_results=[]
for name in ['gauss_embed_dlm_wass_5m.pt','dlm_5m.pt']:
 cp=torch.load(ROOT/'checkpoints'/name,map_location='cpu',weights_only=True); state=cp['model']; is_dlm='embedding.weight' in state
 m=Model(state,is_dlm); chars=cp['chars']; D=m.D
 tok=torch.tensor([chars.index(c) for c in text],dtype=torch.long)
 blocks=tok[:len(tok)//64*64].view(-1,64)
 ids=blocks[torch.randperm(len(blocks),generator=torch.Generator().manual_seed(777))[:64]]
 mean=m.embedding.weight.detach() if is_dlm else m.embedding.mu.detach()
 var=torch.full_like(mean,cp['config']['embedding_std']**2) if is_dlm else m.embedding.std.detach().square()
 if is_dlm:
  T=2000; curve=1-(torch.arange(T+1,dtype=torch.float64)/T+cp['config']['schedule_offset']).sqrt(); beta=(1-curve[1:]/curve[:-1]).clamp(max=.999)
  abar=torch.cat([torch.ones(1),torch.cumprod(1-beta,0)]).float()
 else:
  T=1000; abar=torch.cat([torch.ones(1),torch.cumprod(1-torch.linspace(1e-4,.02,T),0)])
 rows=[]
 with torch.no_grad():
  for seed in [1701,1702,1703]:
   torch.manual_seed(seed); x0=mean[ids]+var[ids].sqrt()*torch.randn(*ids.shape,D)
   perm=torch.stack([torch.randperm(len(ids)) for _ in range(64)],1); positions=torch.arange(64)[None,:]
   for tv in ([100,250,400,500,600] if not is_dlm else [500,1000,1250,1500,1750]):
    a=abar[tv]; xt=a.sqrt()*x0+(1-a).sqrt()*torch.randn_like(x0); ts=torch.full((len(ids),),tv)
    pred=m(xt,ts); mixed=m(xt[perm,positions],ts)
    original_err=(pred-x0).square().mean(); mixed_err=(mixed-x0[perm,positions]).square().mean()
    original_acc=(m.logits(pred).argmax(-1)==ids).float().mean(); mixed_acc=(m.logits(mixed).argmax(-1)==ids[perm,positions]).float().mean()
    rows.append({'seed':seed,'t':tv,'mse':original_err.item(),'shuffled_mse':mixed_err.item(),'accuracy':original_acc.item(),'shuffled_accuracy':mixed_acc.item()})
 record={'checkpoint':name,'rows':rows};all_results.append(record); print(json.dumps(record),flush=True)
(OUT/'context_metrics.json').write_text(json.dumps(all_results,indent=2))
