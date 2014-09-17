#include "common.h"
#include <stdio.h>
#include <string.h>

int load(char *filename, float *data, U32 n)
{
    const char s[2] = " ";
    char *token;
    int numline = 0, numword = 0;
    size_t len = 0;
    char *line = NULL;
    FILE *fp = fopen(filename, "r");

    while (getline(&line, &len, fp) != -1) {
        numword = 0;
        token = strtok(line, s);
        while( token != NULL )
        {
            *(data + numline*n + numword) = (float)atof(token);
            token = strtok(NULL, s);
            if(++numword >= n)    break;
        }
        numline++;
    }
    fclose(fp);
    return numline;
}
