#!/bin/sh

E100="python -m e100_irtool.cmd"
IR=ir.yaml
IR_s=ir_s.yaml
DIMS="32,32"
BATCH=10

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
run rm -f ir_*.yaml *.np *.th *.tf
[ "$1" = "clean" ] && exit

inf "shape inference"
run $E100 shape-inference -d $DIMS $IR $IR_s

inf "generate inputs and weights"
run $E100 gen-input -rt numpy $IR -t float32 -x 256 -d $DIMS -n $BATCH inp.np
run $E100 gen-weight -rt numpy $IR -t float32 -x -1 -d $DIMS wts.np

inf "run with numpy runtime"
run $E100 run -rt numpy -oa $IR inp.np wts.np out.np

inf "run with torch runtime"
run $E100 np2th inp.np inp.th
run $E100 np2th wts.np wts.th
run $E100 run -rt torch -oa $IR inp.th wts.th out.th

inf "run with tensorflow runtime"
run $E100 np2tf inp.np inp.tf -t NXC
run $E100 np2tf wts.np wts.tf -t XCC
run $E100 run -rt tensorflow -oa $IR inp.tf wts.tf out.tf
run $E100 tf2np out.tf out.tf.np -t NCX

inf "run with torch model"
run python run_torch.py inp.th wts.th out.m.th -a

inf "run with tensorflow model"
run python run_tensorflow.py inp.tf wts.tf out.m.tf -a
run $E100 tf2np out.m.tf out.m.tf.np -t NCX

inf "compare results"
run $E100 cmp out.np out.th
run $E100 cmp out.np out.tf.np
run $E100 cmp out.np out.m.th
run $E100 cmp out.np out.m.tf.np

inf "succeeded"
