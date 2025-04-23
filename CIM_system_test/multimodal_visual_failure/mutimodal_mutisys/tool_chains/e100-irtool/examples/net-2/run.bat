@ECHO OFF
SETLOCAL

:: SET E100_DEBUG=1
:: SET E100_ATOL=1e-3
:: SET E100_RTOL=1e-2

SET E100=python -m e100_irtool.cmd
SET IR=ir.yaml
SET IR_f=ir_f.yaml
SET IR_s=ir_s.yaml
SET BATCH=10
:: SET OA=-oa

ECHO # clear output files
DEL /Q ir_*.yaml *.np *.th *.tf 2>NUL
IF "%~1"=="clean" EXIT /B
ECHO.

ECHO # flatten layers
CALL :E100 flatten-layers %IR% %IR_f% || GOTO :Fail

ECHO # shape inference
CALL :E100 shape-inference %IR_f% %IR_s% || GOTO :Fail

ECHO # generate inputs and weights
CALL :E100 gen-input -rt numpy %IR_f% -t float32 -x 256 -n %BATCH% inp.np || GOTO :Fail
CALL :E100 gen-weight -rt numpy %IR_f% -t float32 -x -1 wts.np || GOTO :Fail

ECHO # run with numpy runtime
CALL :E100 run -rt numpy %OA% %IR_f% inp.np wts.np out.np || GOTO :Fail

ECHO # run with torch runtime
CALL :E100 np2th inp.np inp.th || GOTO :Fail
CALL :E100 np2th wts.np wts.th || GOTO :Fail
CALL :E100 run -rt torch %OA% %IR_f% inp.th wts.th out.th || GOTO :Fail

ECHO # run with tensorflow runtime
CALL :E100 np2tf inp.np inp.tf -t NXC || GOTO :Fail
CALL :E100 np2tf wts.np wts.tf -t XCC || GOTO :Fail
CALL :E100 run -rt tensorflow %OA% %IR_f% inp.tf wts.tf out.tf || GOTO :Fail
CALL :E100 tf2np out.tf out.tf.np -t NCX || GOTO :Fail

ECHO # compare results
CALL :E100 cmp out.np out.th || GOTO :Fail
CALL :E100 cmp out.np out.tf.np || GOTO :Fail

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
