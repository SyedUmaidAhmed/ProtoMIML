#!/usr/bin/env python3
"""How many deployed models were ever trained through the fused loss?

The gates sigma(g_l) start at exactly 0.5 and receive gradient only from the
fused logits (Phase-2 ramp, epochs 30-70).  A model whose restored checkpoint
precedes the ramp therefore has every gate still at exactly 0.5.  This script
counts such models from the archived fold files, under the standard and the
fully fused schedule, and writes results/revision/checkpoint_audit.json.

    python revision_checkpoint_audit.py
"""
import json
import os

import numpy as np

REV = os.path.join('results', 'revision')
DS = ['scene', 'reuters', 'letter_carroll', 'letter_frost', 'MSCV2']
TAGS = {'standard': 'protomiml_standard_{}', 'fullfuse': 'protomiml_fullfuse_standard_{}'}


def audit(tag):
    n_models = n_init = 0
    for f in range(10):
        z = np.load(os.path.join(REV, tag, f'fold_{f}.npz'))
        g = np.asarray(z['gate'])  # [seeds, labels]
        n_models += g.shape[0]
        n_init += int(np.sum(np.all(np.abs(g - 0.5) < 1e-6, axis=1)))
    return {'models': n_models, 'gates_at_init': n_init}


def main():
    out = {sched: {ds: audit(t.format(ds)) for ds in DS} for sched, t in TAGS.items()}
    for sched, r in out.items():
        r['total'] = {k: sum(r[ds][k] for ds in DS) for k in ['models', 'gates_at_init']}
        print(sched, r)
    json.dump(out, open(os.path.join(REV, 'checkpoint_audit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
