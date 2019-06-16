#import backprop as bp
import srn as bp
import numpy as np
import random

print("---------------------- Part 1: Testing the backprop with OR/AND/XOR  ----------------------")

Test_OR = False
Test_AND = False
Test_XOR = True

Size = 100
# Training
Input_Array = np.array(np.zeros((Size, 2)))
Output_Array_OR = np.array(np.zeros((Size, 1)))
Output_Array_AND = np.array(np.zeros((Size, 1)))
Output_Array_XOR = np.array(np.zeros((Size, 1)))
for ii in range(np.shape(Input_Array)[0]):
    Input_Array[ii, 0] = random.randint(0, 1)
    Input_Array[ii, 1] = random.randint(0, 1)
    if Input_Array[ii, 0] or Input_Array[ii, 1]:
        Output_Array_OR[ii] = 1
    if Input_Array[ii, 0] and Input_Array[ii, 1]:
        Output_Array_AND[ii] = 1
    if Input_Array[ii, 0] != Input_Array[ii, 1]:
        Output_Array_XOR[ii] = 1

if Test_OR:
    backp_OR = bp.SRN(2, 1, 3)
    backp_OR.train(Input_Array, Output_Array_OR, bp.niter, 0.5, 0, 0)
if Test_AND:
    backp_AND = bp.SRN(2, 1, 3)
    backp_AND.train(Input_Array, Output_Array_AND, bp.niter, 0.5, 0, 0)
if Test_XOR:
    backp_XOR = bp.SRN(2, 1, 3)
    #backp_XOR.load("Part1")
    backp_XOR.train(Input_Array, Output_Array_XOR, 1000, 0.5, 0, 0) # Uncomment to train instead of load weights

# Testing

Input_Array = np.array(np.zeros((Size, 2)))
Output_Array_OR = np.array(np.zeros((Size, 1)))
Output_Array_AND = np.array(np.zeros((Size, 1)))
Output_Array_XOR = np.array(np.zeros((Size, 1)))
for ii in range(np.shape(Input_Array)[0]):
    Input_Array[ii, 0] = random.randint(0, 1)
    Input_Array[ii, 1] = random.randint(0, 1)
    if Input_Array[ii, 0] or Input_Array[ii, 1]:
        Output_Array_OR[ii] = 1
    if Input_Array[ii, 0] and Input_Array[ii, 1]:
        Output_Array_AND[ii] = 1
    if Input_Array[ii, 0] != Input_Array[ii, 1]:
        Output_Array_XOR[ii] = 1

Total_Truly_OR = 0
SUM_OR = 0
MIN_OR = 1
Total_Truly_NOR = 0
SUM_NOR = 0
MAX_NOR = 0

Total_Truly_AND = 0
SUM_AND = 0
MIN_AND = 1
Total_Truly_NAND = 0
SUM_NAND = 0
MAX_NAND = 0

Total_Truly_XOR = 0
SUM_XOR = 0
MIN_XOR = 1
Total_Truly_NXOR = 0
SUM_NXOR = 0
MAX_NXOR = 0


for ii in range(Size):
    if Test_OR:
        Result_OR = backp_OR.test(Input_Array[ii, :])
        if Output_Array_OR[ii]:
            SUM_OR += Result_OR
            if Result_OR < MIN_OR:
                MIN_OR = Result_OR
            Total_Truly_OR += 1
        else:
            SUM_NOR += Result_OR
            if Result_OR > MAX_NOR:
                MAX_NOR = Result_OR
            Total_Truly_NOR += 1
    if Test_AND:
        Result_AND = backp_AND.test(Input_Array[ii, :])
        if Output_Array_AND[ii]:
            SUM_AND += Result_AND
            if Result_AND < MIN_AND:
                MIN_AND = Result_AND
            Total_Truly_AND += 1
        else:
            SUM_NAND += Result_AND
            if Result_AND > MAX_NAND:
                MAX_NAND = Result_AND
            Total_Truly_NAND += 1
    if Test_XOR:
        Result_XOR = backp_XOR.test(Input_Array[ii, :])
        if Output_Array_XOR[ii]:
            SUM_XOR += Result_XOR
            if Result_XOR < MIN_XOR:
                MIN_XOR = Result_XOR
            Total_Truly_XOR += 1
        else:
            SUM_NXOR += Result_XOR
            if Result_XOR > MAX_NXOR:
                MAX_NXOR = Result_XOR
            Total_Truly_NXOR += 1

print("\n")

if Test_OR:
    MEAN_OR = SUM_OR/Total_Truly_OR
    MEAN_NOR = SUM_NOR/Total_Truly_NOR
    print("The mean correct prediction of OR is:", str(MEAN_OR)[1:-1])
    print("The mean correct prediction of NOR is:", str(MEAN_NOR)[1:-1])
    print("The lowest prediction of OR was:", str(MIN_OR)[1:-1])
    print("The highest prediction of NOR was:", str(MAX_NOR)[1:-1])
    print()
if Test_AND:
    MEAN_AND = SUM_AND/Total_Truly_AND
    MEAN_NAND = SUM_NAND/Total_Truly_NAND
    print("The mean correct prediction of AND is:", str(MEAN_AND)[1:-1])
    print("The mean correct prediction of NAND is:", str(MEAN_NAND)[1:-1])
    print("The lowest prediction of AND was:", str(MIN_AND)[1:-1])
    print("The highest prediction of NAND was:", str(MAX_NAND)[1:-1])
    print()
if Test_XOR:
    MEAN_XOR = SUM_XOR/Total_Truly_XOR
    MEAN_NXOR = SUM_NXOR/Total_Truly_NXOR
    print("The mean correct prediction of XOR is:", str(MEAN_XOR)[1:-1])
    print("The mean correct prediction of NXOR is:", str(MEAN_NXOR)[1:-1])
    print("The lowest prediction of XOR was:", str(MIN_XOR)[1:-1])
    print("The highest prediction of NXOR was:", str(MAX_NXOR)[1:-1])
    print()
    backp_XOR.save("Part1")

print(Input_Array[0, :])
print(backp_XOR.test(Input_Array[0, :]))