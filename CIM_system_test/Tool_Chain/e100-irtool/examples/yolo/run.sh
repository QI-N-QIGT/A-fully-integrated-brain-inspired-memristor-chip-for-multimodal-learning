#!/bin/sh

BASE=$(realpath $(dirname $0))
E100="python -m e100_irtool.cmd"
YOLO="python $BASE/e100_yolov3.py"
: ${YOLOv3:=$BASE/yolov3}
: ${DTYPE:=float32}
: ${DIMS:="640,480"}
: ${BATCH:=3}

export PYTHONPATH=$PYTHONPATH:$BASE:$YOLOv3
export E100_MODULES=e100_yolov3

fail() {
	echo
	echo "!!" command failed
	exit 1
}

run() {
	echo ">>" $@
    $@ || fail && echo "<<" done && sleep 1
	echo
}

inf() {
    echo "--" $@ "--"
    echo
}

warn() {
	echo
	echo "%%" $@
}

inf "clean *.yaml *.dot *.pdf *.np *.th *.tf"
rm -f *.yaml *.dot *.pdf *.np *.th *.tf
[ "$1" = "clean" ] && exit

inf "create yolov3 IR"
run $YOLO gen-ir yolov3.yaml
run $YOLO gen-ir --tiny yolov3-tiny.yaml

inf "draw IR graph"
run $E100 ir-dot yolov3.yaml yolov3.dot
run $E100 ir-dot yolov3-tiny.yaml yolov3-tiny.dot
if which dot >/dev/null; then
	run dot -Tpdf yolov3.dot -o yolov3.pdf
	run dot -Tpdf yolov3-tiny.dot -o yolov3-tiny.pdf
else
	warn dot command not found
fi

inf "shape inference"
run $E100 si -d $DIMS yolov3.yaml yolov3.si.yaml
run $E100 si -d $DIMS yolov3-tiny.yaml yolov3-tiny.si.yaml

inf "flatten IR"
run $E100 fl yolov3.yaml yolov3.fl.yaml
run $E100 fl yolov3-tiny.yaml yolov3-tiny.fl.yaml

inf "generate random inputs (batch=$BATCH) and weights"
run $E100 gi -rt numpy -d $DIMS -n $BATCH -t $DTYPE yolov3.fl.yaml inp.np
if [ -f $YOLOv3/yolov3.pt ]; then
	run $YOLO pt2wt $YOLOv3/yolov3.pt wts.np
else
	run $E100 gw -rt numpy -d $DIMS -t $DTYPE -x -0.1 yolov3.fl.yaml wts.np
fi
if [ -f $YOLOv3/yolov3-tiny.pt ]; then
	run $YOLO pt2wt --tiny $YOLOv3/yolov3-tiny.pt wts-tiny.np
else
	run $E100 gw -rt numpy -d $DIMS -t $DTYPE -x -0.1 yolov3-tiny.fl.yaml wts-tiny.np
fi

inf "run with numpy runtime"
run $E100 run -rt numpy yolov3.fl.yaml inp.np wts.np oup.np
run $E100 run -rt numpy yolov3-tiny.fl.yaml inp.np wts-tiny.np oup-tiny.np

inf "run with torch runtime"
run $E100 np2th inp.np inp.th
run $E100 np2th wts.np wts.th
run $E100 run -rt torch yolov3.fl.yaml inp.th wts.th oup.th
run $E100 np2th inp.np inp.th
run $E100 np2th wts-tiny.np wts-tiny.th
run $E100 run -rt torch yolov3-tiny.fl.yaml inp.th wts-tiny.th oup-tiny.th

inf "run with tensorflow runtime"
run $E100 np2tf -t NXC inp.np inp.tf
run $E100 np2tf -t XCC wts.np wts.tf
run $E100 run -rt tensorflow yolov3.fl.yaml inp.tf wts.tf oup.tf
run $E100 tf2np oup.tf oup.tf.np
run $E100 np2tf -t XCC wts-tiny.np wts-tiny.tf
run $E100 run -rt tensorflow yolov3-tiny.fl.yaml inp.tf wts-tiny.tf oup-tiny.tf
run $E100 tf2np oup-tiny.tf oup-tiny.tf.np

inf "run with yolov3 model and weights"
if [ -f $YOLOv3/yolov3.pt ]; then
	run $YOLO run-pt --keep $YOLOv3/yolov3.pt inp.th oup.pt.th
else
	warn $YOLOv3/yolov3.pt file not found
fi
if [ -f $YOLOv3/yolov3-tiny.pt ]; then
	run $YOLO run-pt --keep $YOLOv3/yolov3-tiny.pt inp.th oup-tiny.pt.th
else
	warn $YOLOv3/yolov3-tiny.pt file not found
fi

inf "compare outputs"
run $E100 cmp oup.np oup.th
run $E100 cmp oup.np oup.tf.np
[ -f oup.pt.th ] && run $E100 cmp oup.np oup.pt.th
run $E100 cmp oup-tiny.np oup-tiny.th
run $E100 cmp oup-tiny.np oup-tiny.tf.np
[ -f oup-tiny.pt.th ] && run $E100 cmp oup-tiny.np oup-tiny.pt.th

inf all succeeded