/*
 * =====================================================================================
 *
 *       Filename:  a111_helper.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  08/22/2023 01:21:41 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>

#include "a111_helper.h"




int a111_helper_test_func(int i)
{
  int ret = 0;
  
  printf("hello, a111_helper! \r\n");
  printf("param of this func is %d\r\n", i);

  return ret;
}
