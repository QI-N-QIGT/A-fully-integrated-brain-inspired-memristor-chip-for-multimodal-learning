#!/bin/bash
set -e
cd $(dirname $0)

[[ "$1" = "clean" ]] && rm -f _* && exit

e100="python -m e100_irtool.cmd"
bat=5

export E100_MODULES="e100_irtool.tools.loop"
export E100_DEBUG=1

run() {
    i=$1
    dim=$2
    [[ ! -f "ir-$i.yaml" ]] && echo "Error: ir-$i.yaml not found" && return 1

    $e100 ir ir-$i.yaml

    $e100 fl ir-$i.yaml _ir-$i.fl.yaml
    $e100 ir _ir-$i.fl.yaml

    [[ -z "$dim" ]] && return

    $e100 si -d $dim ir-$i.yaml _ir-$i.si.yaml || :
    $e100 si -d $dim _ir-$i.fl.yaml _ir-$i.fl.si.yaml

    $e100 gi -n $bat -d $dim _ir-$i.fl.yaml _inp-$i.np
    $e100 gw -d $dim _ir-$i.fl.yaml _wts-$i.np
    $e100 run _ir-$i.fl.yaml _inp-$i.np _wts-$i.np _oup-$i.np

    [[ ! -f "ir-$i-th.py" ]] && return

    $e100 np2th _inp-$i.np _inp-$i.th
    $e100 np2th _wts-$i.np _wts-$i.th
    python ir-$i-th.py _inp-$i.th _wts-$i.th _oup-$i.th

    $e100 cmp _oup-$i.np _oup-$i.th
}

run 1 28,28
run 2 32,32
