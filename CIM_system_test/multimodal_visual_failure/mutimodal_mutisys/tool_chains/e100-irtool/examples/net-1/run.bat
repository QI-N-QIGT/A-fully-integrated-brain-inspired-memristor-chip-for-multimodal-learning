@ECHO OFF
SETLOCAL

:: SET E100_DEBUG=1
SET E100_ATOL=1e-4
SET E100_RTOL=1e-3

SET E100=python -m e100_irtool.cmd
SET IR=ir.yaml
SET IR_s=ir_s.yaml
SET DIMS=32,32
SET BATCH=10

ECHO # clear output files
DEL /Q *.np *.th *.tf %IR_s% 2>NUL
IF "%~1"=="clean" EXIT /B
ECHO.

ECHO # shape inference
CALL :E100 shape-inference -d %DIMS% %IR% %IR_s% || GOTO :Fail

ECHO # generate inputs and weights
CALL :E100 gen-input -rt numpy %IR% -t float32 -x 256 -d %DIMS% -n %BATCH% inp.np || GOTO :Fail
CALL :E100 gen-weight -rt numpy %IR% -t float32 -x -1 -d %DIMS% wts.np || GOTO :Fail

ECHO # run with numpy runtime
CALL :E100 run -rt numpy -oa %IR% inp.np wts.np out.np || GOTO :Fail

ECHO # run with torch runtime
CALL :E100 np2th inp.np inp.th || GOTO :Fail
CALL :E100 np2th wts.np wts.th || GOTO :Fail
CALL :E100 run -rt torch -oa %IR% inp.th wts.th out.th || GOTO :Fail

ECHO # run with tensorflow runtime
CALL :E100 np2tf inp.np inp.tf -t NXC || GOTO :Fail
CALL :E100 np2tf wts.np wts.tf -t XCC || GOTO :Fail
CALL :E100 run -rt tensorflow -oa %IR% inp.tf wts.tf out.tf || GOTO :Fail
CALL :E100 tf2np out.tf out.tf.np -t NCX || GOTO :Fail

ECHO # run with torch model
ECHO run_torch.py inp.th wts.th out.m.th
python run_torch.py inp.th wts.th out.m.th -a || GOTO :Fail
ECHO.

ECHO # run with tensorflow model
ECHO python run_tensorflow.py inp.tf wts.tf out.m.tf -a
python run_tensorflow.py inp.tf wts.tf out.m.tf -a || GOTO :Fail
CALL :E100 tf2np out.m.tf out.m.tf.np -t NCX || GOTO :Fail

ECHO # compare results
CALL :E100 cmp out.np out.th || GOTO :Fail
CALL :E100 cmp out.np out.tf.np || GOTO :Fail
CALL :E100 cmp out.np out.m.th || GOTO :Fail
CALL :E100 cmp out.np out.m.tf.np || GOTO :Fail

GOTO :End

:E100
ECHO e100 %*
%E100% %*
ECHO.
EXIT /B

:Fail
ECHO.

:End
IF NOT ERRORLEVEL 1 (
	ECHO # succeeded
) ELSE (
	ECHO # failed
)