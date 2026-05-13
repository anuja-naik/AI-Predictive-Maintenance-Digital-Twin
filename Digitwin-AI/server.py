"""
DigiTwin.AI — Flask Backend Server
Team: Tech Titans | PICT Pune | April 2026
Run: pip install flask flask-cors numpy && python server.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import math, threading

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# AUTOENCODER
# ─────────────────────────────────────────
class AutoencoderNumPy:
    def __init__(self, input_dim=3, lr=0.005):
        self.input_dim = input_dim
        self.lr = lr
        def xavier(fi, fo):
            limit = math.sqrt(6.0/(fi+fo))
            return np.random.uniform(-limit, limit, (fo, fi))
        self.W1=xavier(3,8); self.b1=np.zeros(8)
        self.W2=xavier(8,4); self.b2=np.zeros(4)
        self.W3=xavier(4,2); self.b3=np.zeros(2)
        self.W4=xavier(2,4); self.b4=np.zeros(4)
        self.W5=xavier(4,8); self.b5=np.zeros(8)
        self.W6=xavier(8,3); self.b6=np.zeros(3)

    @staticmethod
    def relu(x): return np.maximum(0,x)
    @staticmethod
    def relu_deriv(x): return (x>0).astype(float)
    @staticmethod
    def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
    @staticmethod
    def sigmoid_deriv(x):
        s=1/(1+np.exp(-np.clip(x,-500,500))); return s*(1-s)

    def forward(self, x):
        z1=x@self.W1.T+self.b1; a1=self.relu(z1)
        z2=a1@self.W2.T+self.b2; a2=self.relu(z2)
        z3=a2@self.W3.T+self.b3; lat=self.relu(z3)
        z4=lat@self.W4.T+self.b4; a4=self.relu(z4)
        z5=a4@self.W5.T+self.b5; a5=self.relu(z5)
        z6=a5@self.W6.T+self.b6; recon=self.sigmoid(z6)
        return {'z1':z1,'a1':a1,'z2':z2,'a2':a2,'z3':z3,'lat':lat,
                'z4':z4,'a4':a4,'z5':z5,'a5':a5,'z6':z6,'recon':recon}

    def compute_loss(self, x, recon): return float(np.mean((x-recon)**2))

    def backward(self, x, cache):
        recon=cache['recon']; N=x.shape[0] if x.ndim>1 else 1
        dL=2*(recon-x)/self.input_dim
        dz6=dL*self.sigmoid_deriv(cache['z6'])
        dW6=dz6.T@cache['a5']/N; db6=dz6.mean(0) if x.ndim>1 else dz6
        da5=dz6@self.W6; dz5=da5*self.relu_deriv(cache['z5'])
        dW5=dz5.T@cache['a4']/N; db5=dz5.mean(0) if x.ndim>1 else dz5
        da4=dz5@self.W5; dz4=da4*self.relu_deriv(cache['z4'])
        dW4=dz4.T@cache['lat']/N; db4=dz4.mean(0) if x.ndim>1 else dz4
        dlat=dz4@self.W4; dz3=dlat*self.relu_deriv(cache['z3'])
        dW3=dz3.T@cache['a2']/N; db3=dz3.mean(0) if x.ndim>1 else dz3
        da2=dz3@self.W3; dz2=da2*self.relu_deriv(cache['z2'])
        dW2=dz2.T@cache['a1']/N; db2=dz2.mean(0) if x.ndim>1 else dz2
        da1=dz2@self.W2; dz1=da1*self.relu_deriv(cache['z1'])
        dW1=dz1.T@x/N; db1=dz1.mean(0) if x.ndim>1 else dz1
        for W,dW,b,db in [(self.W1,dW1,self.b1,db1),(self.W2,dW2,self.b2,db2),
                           (self.W3,dW3,self.b3,db3),(self.W4,dW4,self.b4,db4),
                           (self.W5,dW5,self.b5,db5),(self.W6,dW6,self.b6,db6)]:
            W-=self.lr*dW; b-=self.lr*db

    def train(self, data, epochs=80, batch_size=32):
        losses=[]; n=len(data)
        print(f"\n{'='*55}\n  Training Autoencoder | 3→8→4→2→4→8→3 | {n} samples\n{'='*55}")
        for ep in range(epochs):
            idx=np.random.permutation(n); epoch_loss=0; batches=0
            for i in range(0,n,batch_size):
                batch=data[idx[i:i+batch_size]]
                cache=self.forward(batch)
                loss=self.compute_loss(batch,cache['recon'])
                self.backward(batch,cache)
                epoch_loss+=loss; batches+=1
            avg=epoch_loss/batches; losses.append(avg)
            if ep%10==0 or ep==epochs-1:
                print(f"  Epoch {ep+1:3d}/{epochs} | MSE: {avg:.5f}")
        print(f"\n  ✓ Done. Final MSE: {losses[-1]:.5f}")
        return losses

    def predict_mse(self, x):
        if x.ndim==1: x=x.reshape(1,-1)
        cache=self.forward(x)
        return self.compute_loss(x,cache['recon'])

# ─────────────────────────────────────────
# HEALTH / RUL
# ─────────────────────────────────────────
ANOMALY_TH=0.15; WARN_TH=0.08; CRIT_HEALTH=20

def mse_to_health(m): return max(0,round((1-min(m/0.20,1))*100))
def compute_rul(health,temp,load,vib):
    stress=(temp-20)/100*0.40 + load/100*0.35 + vib/10*0.25
    dr=0.1+stress*0.9
    return max(0,round(max(0,health-CRIT_HEALTH)/max(dr,0.01)*1.5))
def get_status(mse,health):
    if mse>ANOMALY_TH or health<30: return {'s':'CRITICAL','c':'red','action':'Immediate maintenance required.'}
    if mse>WARN_TH or health<60:    return {'s':'WARNING','c':'amber','action':'Schedule maintenance within 48 hours.'}
    return {'s':'NORMAL','c':'green','action':'Continue monitoring. No intervention needed.'}
def get_recs(temp,load,vib,mse,health):
    r=[]
    if temp>90: r.append(f"Temp {temp:.1f}C exceeds safe limit. Improve cooling.")
    if load>80: r.append(f"Load {load:.0f}% overload. Reduce by {load-70:.0f}%.")
    if vib>7:   r.append(f"Vibration {vib:.1f}mm/s critical. Inspect bearings.")
    if mse>ANOMALY_TH: r.append(f"Anomaly detected MSE={mse:.4f}.")
    if health<40: r.append(f"Health {health}% critically low. Immediate maintenance.")
    if not r: r.append("All parameters nominal. No action required.")
    return r

# ─────────────────────────────────────────
# LIVE SIMULATOR
# ─────────────────────────────────────────
class LiveSimulator:
    def __init__(self):
        self.mode='degrading'; self.tick=0
        self.temp=65.0; self.load=50.0; self.vib=2.5
        self.history=[]

    def next(self, model):
        self.tick+=1; t=self.tick
        if self.mode=='degrading':
            self.temp=min(120, 65+t*0.8+np.random.normal(0,1))
            self.load=min(100, 50+t*0.6+np.random.normal(0,1))
            self.vib =min(10,  2.5+t*0.1+np.random.normal(0,0.2))
        elif self.mode=='spike':
            phase=t%30
            if phase<10:
                self.temp=65+np.random.normal(0,2); self.load=50+np.random.normal(0,3); self.vib=2.5+np.random.normal(0,0.3)
            elif phase<18:
                self.temp=105+np.random.normal(0,2); self.load=88+np.random.normal(0,2); self.vib=8.0+np.random.normal(0,0.5)
            else:
                fr=(phase-18)/12
                self.temp=105-fr*40+np.random.normal(0,2); self.load=88-fr*38+np.random.normal(0,2); self.vib=8.0-fr*5.5+np.random.normal(0,0.3)
        elif self.mode=='fluctuating':
            self.temp=np.clip(65+20*np.sin(t/8)+np.random.normal(0,5),20,115)
            self.load=np.clip(50+25*np.sin(t/5+1)+np.random.normal(0,5),0,100)
            self.vib =np.clip(2.5+4*abs(np.sin(t/6))+np.random.normal(0,0.5),0,10)

        temp=round(float(np.clip(self.temp,20,120)),1)
        load=round(float(np.clip(self.load,0,100)),1)
        vib =round(float(np.clip(self.vib,0,10)),2)
        inp=np.array([(temp-20)/100,load/100,vib/10],dtype=np.float32)
        mse=model.predict_mse(inp); health=mse_to_health(mse)
        rul=compute_rul(health,temp,load,vib)
        st=get_status(mse,health); reclist=get_recs(temp,load,vib,mse,health)
        entry={'tick':t,'temp':temp,'load':load,'vib':vib,
               'mse':round(mse,5),'health':health,'rul':rul,'status':st,'recs':reclist}
        self.history.append(entry)
        if len(self.history)>60: self.history.pop(0)
        return entry

    def reset(self, mode):
        self.mode=mode; self.tick=0
        self.temp=65.0; self.load=50.0; self.vib=2.5; self.history=[]
        print(f"\n  [LIVE] Mode → {mode.upper()}")

# ─────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────
np.random.seed(42)
model=None; live_sim=LiveSimulator()
loss_log=[]
training_status={'done':False,'epoch':0,'loss':0,'accuracy':0,
                 'normal_mse_mean':0,'normal_mse_max':0,
                 'anomaly_mse_mean':0,'anomaly_mse_max':0,'threshold':0}

def generate_normal(n=800):
    t=np.clip(np.random.normal(0.45,0.08,n),0.05,0.95)
    l=np.clip(np.random.normal(0.50,0.10,n),0.05,0.95)
    v=np.clip(np.random.normal(0.25,0.05,n),0.05,0.95)
    return np.column_stack([t,l,v]).astype(np.float32)

def generate_anomalous(n=150):
    k=n//3
    o=np.column_stack([np.random.uniform(0.8,1.0,k),np.random.normal(0.5,0.08,k),np.random.normal(0.25,0.05,k)])
    l=np.column_stack([np.random.normal(0.45,0.08,k),np.random.uniform(0.85,1.0,k),np.random.uniform(0.6,0.9,k)])
    b=np.column_stack([np.random.normal(0.45,0.08,k),np.random.normal(0.5,0.1,k),np.random.uniform(0.75,1.0,k)])
    return np.clip(np.vstack([o,l,b]),0,1).astype(np.float32)

def train_model():
    global model, loss_log
    print("\n[SERVER] Generating data...")
    normal=generate_normal(800); anomalous=generate_anomalous(150)
    model=AutoencoderNumPy(input_dim=3,lr=0.005)
    losses=model.train(normal,epochs=80,batch_size=32); loss_log=losses
    normal_mses=[model.predict_mse(x) for x in normal[:50]]
    anomaly_mses=[model.predict_mse(x) for x in anomalous[:50]]
    threshold=np.percentile(normal_mses,95)
    correct=sum(m>threshold for m in anomaly_mses)
    accuracy=round(correct/len(anomaly_mses)*100,1)
    training_status.update({'done':True,'epoch':80,'loss':round(losses[-1],5),
        'accuracy':accuracy,
        'normal_mse_mean':round(float(np.mean(normal_mses)),4),
        'normal_mse_max':round(float(np.max(normal_mses)),4),
        'anomaly_mse_mean':round(float(np.mean(anomaly_mses)),4),
        'anomaly_mse_max':round(float(np.max(anomaly_mses)),4),
        'threshold':round(float(threshold),4)})
    print(f"\n[SERVER] ✓ Ready! Accuracy={accuracy}% | Threshold={threshold:.4f}\n")

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route('/')
def index(): return jsonify({'status':'DigiTwin.AI backend','trained':training_status['done']})

@app.route('/status')
def status(): return jsonify(training_status)

@app.route('/loss_log')
def get_loss_log(): return jsonify({'losses':loss_log})

@app.route('/simulate', methods=['POST'])
def simulate():
    if not training_status['done']: return jsonify({'error':'Still training'}),503
    d=request.json
    temp=float(d.get('temp',65)); load=float(d.get('load',50)); vib=float(d.get('vib',2.5))
    inp=np.array([(temp-20)/100,load/100,vib/10],dtype=np.float32)
    mse=model.predict_mse(inp); health=mse_to_health(mse)
    rul=compute_rul(health,temp,load,vib); st=get_status(mse,health)
    return jsonify({'temp':temp,'load':load,'vib':vib,'mse':round(mse,5),
                    'health':health,'rul':rul,'status':st,'recs':get_recs(temp,load,vib,mse,health)})

@app.route('/live/tick')
def live_tick():
    if not training_status['done']: return jsonify({'error':'Still training'}),503
    return jsonify(live_sim.next(model))

@app.route('/live/mode/<mode>', methods=['POST'])
def set_mode(mode):
    if mode not in ('degrading','spike','fluctuating'): return jsonify({'error':'Invalid mode'}),400
    live_sim.reset(mode); return jsonify({'mode':mode,'ok':True})

@app.route('/live/history')
def live_history(): return jsonify({'history':live_sim.history})

if __name__=='__main__':
    print("\n"+"="*55)
    print("  DigiTwin.AI — Backend Server")
    print("  Tech Titans | PICT Pune | April 2026")
    print("="*55)
    threading.Thread(target=train_model,daemon=True).start()
    print("\n[SERVER] Training in background... API at http://localhost:5000")
    print("[SERVER] Dashboard auto-connects after training (~15s)\n")
    app.run(debug=False,port=5000,threaded=True)
