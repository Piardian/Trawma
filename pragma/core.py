from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .orderflow_engine import OrderFlowConfig, calculate_orderflow_features, detect_orderflow_pools

def run_cli(description: str, argv=None) -> int:
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--engine', choices=('amt','orderflow','hybrid'), default='amt')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--symbol', default='UNKNOWN')
    parser.add_argument('--output', type=Path)
    args=parser.parse_args(argv)
    if args.engine == 'amt': raise NotImplementedError('Trawma AMT engine not implemented yet')
    if args.engine == 'hybrid': raise NotImplementedError('Hybrid requires the Trawma AMT POI provider')
    data=pd.read_csv(args.data)
    features=calculate_orderflow_features(data, config=OrderFlowConfig())
    pools=detect_orderflow_pools(data, config=OrderFlowConfig())
    last=features.iloc[-1] if len(features) else None
    result={'symbol':args.symbol,'engine':args.engine,'bars':len(data),'delta_source':str(last['delta_source']) if last is not None else 'empty','last_bar_delta':float(last['bar_delta']) if last is not None else None,'last_cvd':float(last['cvd']) if last is not None else None,'detected_pool_count':len(pools),'pools':[{'kind':p.kind,'side':p.side,'level':p.level,'lower':p.lower,'upper':p.upper,'timestamp':str(p.timestamp),'source':p.source} for p in pools],'note':'Feature/pool report only; not a trade-level backtest.'}
    text=json.dumps(result,indent=2,default=str)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text+'\n',encoding='utf-8')
    print(text); return 0
