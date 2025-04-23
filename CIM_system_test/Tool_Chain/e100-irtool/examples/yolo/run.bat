@ECHO OFF
SETLOCAL

SET PYTHONPATH=%PYTHONPATH%;yolov3
SET E100_MODULES=e100_yolov3
SET E100=python -m e100_irtool.cmd
SET YOLO=python e100_yolov3.py
if "%DTYPE%"=="" SET DTYPE=float32
if "%DIMS%"=="" SET DIMS=640,480
if "%BATCH%"=="" SET BATCH=1

:: clean

ECHO -- clean *.yaml *.np *.th *.tf *.dot *.pdf -- && ECHO.
DEL /Q *.yaml *.np *.th *.tf *.dot *.pdf 2>NUL
if "%~1"=="clean" EXIT /B

:: IR

ECHO -- create yolov3 IR -- && ECHO.
CALL :Run %YOLO% gen-ir yolov3.yaml	|| GOTO :Fail
CALL :Run %YOLO% gen-ir --tiny yolov3-tiny.yaml	|| GOTO :Fail

ECHO -- draw IR graph -- && ECHO.
CALL :Run %E100% ir-dot yolov3.yaml yolov3.dot || GOTO :Fail
CALL :Run %E100% ir-dot yolov3-tiny.yaml yolov3-tiny.dot || GOTO :Fail
where dot.exe >NUL 2>NUL || GOTO :Nodot
CALL :Run dot -Tpdf yolov3.dot -o yolov3.pdf || GOTO :Fail
CALL :Run dot -Tpdf yolov3-tiny.dot -o yolov3-tiny.pdf || GOTO :Fail
GOTO :Withdot
:Nodot
ECHO [!] dot.exe may not exit && ECHO.
:Withdot

ECHO -- shape inference -- && ECHO.
CALL :Run %E100% si -d %DIMS% yolov3.yaml yolov3.si.yaml || GOTO :Fail
CALL :Run %E100% si -d %DIMS% yolov3-tiny.yaml yolov3-tiny.si.yaml || GOTO :Fail

ECHO -- flatten IR -- && ECHO.
CALL :Run %E100% fl yolov3.yaml yolov3.fl.yaml || GOTO :Fail
CALL :Run %E100% fl yolov3-tiny.yaml yolov3-tiny.fl.yaml || GOTO :Fail

:: data

ECHO -- generate random inputs (batch=%BATCH%) and weights -- && ECHO.
CALL :Run %E100% gi -rt numpy -d %DIMS% -n %BATCH% -t %DTYPE% yolov3.fl.yaml inp.np || GOTO :Fail
IF EXIST yolov3/yolov3.pt (
	CALL :Run %YOLO% pt2wt yolov3/yolov3.pt wts.np || GOTO :Fail
) ELSE (
	CALL :Run %E100% gw -rt numpy -d %DIMS% -t %DTYPE% -x -0.1 yolov3.fl.yaml wts.np || GOTO :Fail
)
IF EXIST yolov3/yolov3-tiny.pt (
	CALL :Run %YOLO% pt2wt --tiny yolov3/yolov3-tiny.pt wts-tiny.np || GOTO :Fail
) ELSE (
	CALL :Run %E100% gw -rt numpy -d %DIMS% -t %DTYPE% -x -0.1 yolov3-tiny.fl.yaml wts-tiny.np || GOTO :Fail	
)

:: run by runtime

ECHO -- run with numpy runtime -- && ECHO.
CALL :Run %E100% run -rt numpy yolov3.fl.yaml inp.np wts.np oup.np || GOTO :Fail
CALL :Run %E100% run -rt numpy yolov3-tiny.fl.yaml inp.np wts-tiny.np oup-tiny.np || GOTO :Fail

ECHO -- run with torch runtime -- && ECHO.
CALL :Run %E100% np2th inp.np inp.th || GOTO :Fail
CALL :Run %E100% np2th wts.np wts.th || GOTO :Fail
CALL :Run %E100% run -rt torch yolov3.fl.yaml inp.th wts.th oup.th || GOTO :Fail
CALL :Run %E100% np2th inp.np inp.th || GOTO :Fail
CALL :Run %E100% np2th wts-tiny.np wts-tiny.th || GOTO :Fail
CALL :Run %E100% run -rt torch yolov3-tiny.fl.yaml inp.th wts-tiny.th oup-tiny.th || GOTO :Fail

ECHO -- run with tensorflow runtime -- && ECHO.
CALL :Run %E100% np2tf -t NXC inp.np inp.tf || GOTO :Fail
CALL :Run %E100% np2tf -t XCC wts.np wts.tf || GOTO :Fail
CALL :Run %E100% run -rt tensorflow yolov3.fl.yaml inp.tf wts.tf oup.tf || GOTO :Fail
CALL :Run %E100% tf2np oup.tf oup.tf.np || GOTO :Fail
CALL :Run %E100% np2tf -t XCC wts-tiny.np wts-tiny.tf || GOTO :Fail
CALL :Run %E100% run -rt tensorflow yolov3-tiny.fl.yaml inp.tf wts-tiny.tf oup-tiny.tf || GOTO :Fail
CALL :Run %E100% tf2np oup-tiny.tf oup-tiny.tf.np || GOTO :Fail

:: run by yolov3

ECHO -- run with yolov3 model and weights -- && ECHO.
IF EXIST yolov3/yolov3.pt (
	CALL :Run %YOLO% run-pt --keep yolov3/yolov3.pt inp.th oup.pt.th || GOTO :Fail
) ELSE (
	ECHO [!] yolov3/yolov3.pt file not found && ECHO.
)
IF EXIST yolov3/yolov3-tiny.pt (
	CALL :Run %YOLO% run-pt --keep yolov3/yolov3-tiny.pt inp.th oup-tiny.pt.th || GOTO :Fail
) ELSE (
	ECHO [!] yolov3/yolov3-tiny.pt file not found && ECHO.
)

:: compare

ECHO -- compare outputs -- && ECHO.
CALL :Run %E100% cmp oup.np oup.th || GOTO :Fail
CALL :Run %E100% cmp oup.np oup.tf.np || GOTO :Fail
IF EXIST oup.pt.th (
	CALL :Run %E100% cmp oup.np oup.pt.th || GOTO :Fail
)
CALL :Run %E100% cmp oup-tiny.np oup-tiny.th || GOTO :Fail
CALL :Run %E100% cmp oup-tiny.np oup-tiny.tf.np || GOTO :Fail
IF EXIST oup-tiny.pt.th (
	CALL :Run %E100% cmp oup-tiny.np oup-tiny.pt.th || GOTO :Fail
)

GOTO :End

:Run
	ECHO ^>^> %*
	%* || EXIT /B
	ECHO ^<^< done
	TIMEOUT /T 1 /NOBREAK >NUL
	ECHO.
	EXIT /B

:Fail
:End
	IF NOT ERRORLEVEL 1 (
		ECHO -- all succeeded --
	) ELSE (
		ECHO !! failed
	)
	EXIT /B
