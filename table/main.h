#ifndef MAIN_H
#define MAIN_H

uint stringToByteArray(char *str, byte **array);
uint stringToByteArray(char *str, uint **array);
void printHexArray(uint *array, uint size);
void copyTables();
void freeTables();

extern uint *cTe0, *cTe1, *cTe2, *cTe3, *cTe4, *cTd0, *cTd1, *cTd2, *cTd3, *cTd4;
extern uint Te0[256], Te1[256], Te2[256], Te3[256], Te4[256], Td0[256], Td1[256], Td2[256], Td3[256], Td4[256];

#endif
