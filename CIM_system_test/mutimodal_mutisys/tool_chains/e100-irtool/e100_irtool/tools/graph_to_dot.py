from copy import deepcopy
_themes = {'default': {'node': {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'cyan', 'color': 'cyan'}, 'edge': {'color': 'blue'}, 'cluster': {'color': 'gray90', 'labeljust': 'r'}, 'input': {'fillcolor': 'green'}, 'output': {'fillcolor': 'green'}, 'block-io': {'style': 'dotted'}}}

def get_attrs(theme, obj, bracket=True, **kwargs):
    if isinstance(obj, str):
        key = obj
    else:
        key = getattr(obj, 'tag', None)
    attrs = {}
    for (k, v) in theme.get(key, {}).items():
        if v not in (None, ''):
            attrs[k] = v
    for (k, v) in kwargs.items():
        if v not in (None, ''):
            attrs[k] = v
    if attrs:
        if bracket:
            yield '['
        for (k, v) in attrs.items():
            yield f'{k}="{v}"'
        if bracket:
            yield ']'

def graph_to_dot(grf, theme='default', label=None, **kwargs):
    theme = deepcopy(_themes[theme])
    for (k, v) in kwargs.items():
        if k not in theme:
            theme[k] = v
        else:
            theme[k].update(v)
    if label is None:

        def label(n, v):
            return
    print()
    print()
    print()
    nds = grf.sorted_nodes()
    for v in nds:
        if v.group is None:
            print()
    pre = 0
    for (lev, g) in grf.iter_groups():
        for i in range(pre, lev - 1, -1):
            print()
        ind = ' ' * (lev * 2 - 1)
        print()
        for a in get_attrs(theme, 'cluster', False):
            print()
        for a in get_attrs(theme, g, False, label=label(g.name, g)):
            print()
        for v in nds:
            if v.group == g.name:
                print()
        pre = lev
    for i in range(pre, 0, -1):
        print()
    for e in grf.iter_edges():
        print()
    print('}')