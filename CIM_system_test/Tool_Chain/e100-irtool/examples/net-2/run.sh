#!/bin/sh

E100="python -m e100_irtool.cmd"
IR=ir.yaml
IR_f=ir_f.yaml
IR_s=ir_s.yaml
BATCH=10
# OA=-oa

fail() {
	echo
	echo "!!! failed"
	exit 1
}

run() {
	echo '>>' $@
	$@ || fail
	echo
}

inf() {
    echo "---" $@ "---"
    echo
}

inf "clean all"
rm -f ir_*.yaml *.np *.th *.tf
[ "$1" = "clean" ] && exit

inf "flatten layers"
run $E100 flatten-layers $IR $IR_f

inf "shape inference"
run $E100 shape-inference $IR_f $IR_s

inf "generate inputs and weights"
run $E100 gen-input -rt numpy $IR_f -t float32 -x 256 -n $BATCH inp.np
run $E100 gen-weight -rt numpy $IR_f -t float32 -x -1 wts.np

inf "run with numpy runtime"
run $E100 run -rt numpy $OA $IR_f inp.np wts.np out.np

inf "run with torch runtime"
run $E100 np2th inp.np inp.th
run $E100 np2th wts.np wts.th
run $E100 run -rt torch $OA $IR_f inp.th wts.th out.th

inf "run with tensorflow runtime"
run $E100 np2tf inp.np inp.tf -t NXC
run $E100 np2tf wts.np wts.tf -t XCC
run $E100 run -rt tensorflow $OA $IR_f inp.tf wts.tf out.tf
run $E100 tf2np out.tf out.tf.np -t NCX

inf "compare results"
run $E100 cmp out.np out.th
run $E100 cmp out.np out.tf.np

inf "succeeded"
