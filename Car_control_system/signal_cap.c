#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

void signal_handler(int signum) {
    printf("Signal %d received\n", signum);
    fflush(stdout);
    if(signum == SIGHUP)
       exit(-1);
}

int main() {
    // 注册信号处理器
    signal(SIGINT, signal_handler); // Ctrl+C
    signal(SIGQUIT, signal_handler); // Ctrl+\//
    signal(SIGILL, signal_handler); // 非法指令
    signal(SIGHUP, signal_handler); // 非法指令
    signal(SIGURG, signal_handler); // 非法指令
    signal(SIGABRT, signal_handler); // 异常终止
    signal(SIGFPE, signal_handler); // 浮点异常
    signal(SIGKILL, signal_handler); // 强制终止
    signal(SIGSEGV, signal_handler); // 无效内存引用
    signal(SIGPIPE, signal_handler); // 管道破裂
    signal(SIGALRM, signal_handler); // 闹钟
    signal(SIGTERM, signal_handler); // 终止
    signal(SIGUSR1, signal_handler); // 用户定义信号1
    signal(SIGUSR2, signal_handler); // 用户定义信号2
    signal(SIGCHLD, signal_handler); // 子进程终止
    signal(SIGCONT, signal_handler); // 继续执行
    signal(SIGSTOP, signal_handler); // 停止执行
    signal(SIGTSTP, signal_handler); // 交互式停止
    signal(SIGTTIN, signal_handler); // 后台读
    signal(SIGTTOU, signal_handler); // 后台写

    while (1) {
        sleep(1); // 让程序持续运行
    }

    return 0;
}
