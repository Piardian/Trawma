import json

with open("artifacts/real_walk_forward_100k_comparison.json", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    pf = f"{d['profit_factor']:.2f}" if d['profit_factor'] is not None else "N/A"
    print(f"{d['symbol']} {d.get('timeframe', '')} {d['engine']}: trades={d['total_trades']}, WR={d['win_rate']*100:.1f}%, PF={pf}, NetR={d['net_r']:+.2f}, MaxDD={d['max_drawdown_r']:.2f}")
