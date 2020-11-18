def Check_LP_sol (X):
    """
    Check whether a solution for the LP problem satisfies all the constraints
    """

    if (X[0] > 1):
        return False

    if (X[1] > 1):
        return False

    if (X[2] > 1):
        return False

    if (X[3] > 1):
        return False

    if (X[4] > 1):
        return False

    if (X[5] > 1):
        return False

    if (X[6] > 1):
        return False

    if (X[7] > 1):
        return False

    if (X[8] > 1):
        return False

    if (X[9] > 1):
        return False

    if (X[10] > 1):
        return False

    if (X[11] > 1):
        return False

    if (X[12] > 1):
        return False

    if (X[13] > 1):
        return False

    if (X[14] > 1):
        return False

    if (X[15] > 1):
        return False

    if (X[16] > 1):
        return False

    if (X[17] > 1):
        return False

    if (not (X[0] + X[1]+ X[2]+ X[3]+ X[4]+ X[5]+ X[6]+ X[7]+ X[8] == 1)):
        return False
    if (not (X[9] + X[10]+ X[11]+ X[12]+ X[13]+ X[14]+ X[15]+ X[16]+ X[17] == 1)):
        return False

    if (-1*X[0] + 1*X[2] + 1*X[10] + 2*X[11]  > 0):
        return False

    if (1*X[3] + 2*X[4] + 3*X[5] + 1*X[12] + 2*X[13] + 3*X[14]  > 3):
        return False

    if (1*X[6] + 2*X[7] + 3*X[8] + 1*X[15] + 2*X[16] + 3*X[17]  > 3):
        return False

    if (5.0*X[3] + 5.0*X[4] + 5.0*X[5] + 5.0*X[6] + 5.0*X[7] + 5.0*X[8] + 5.0*X[12] + 5.0*X[13] + 5.0*X[14] + 5.0*X[15] + 5.0*X[16] + 5.0*X[17] > 8.0):
        return False

    if (1.0*X[3] + 1.0*X[4] + 1.0*X[5] + 5.0*X[6] + 5.0*X[7] + 5.0*X[8] + 1.0*X[12] + 1.0*X[13] + 1.0*X[14] + 5.0*X[15] + 5.0*X[16] + 5.0*X[17] > 8.0):
        return False

    if (1.0*X[3] + 1.0*X[4] + 1.0*X[5] + 1.0*X[12] + 1.0*X[13] + 1.0*X[14] > 8.0):
        return False

    if (10.0000*X[0] + 8.6667*X[1] + 8.4000*X[2] + 6.0000*X[3] + 4.6667*X[4] + 4.4000*X[5] + 2.0000*X[6] + 0.6667*X[7] + 0.4000*X[8] > 10.0):
        return False

    if (10.0000*X[9] + 8.6667*X[10] + 8.4000*X[11] + 6.0000*X[12] + 4.6667*X[13] + 4.4000*X[14] + 2.0000*X[15] + 0.6667*X[16] + 0.4000*X[17] > 10.0):
        return False



    return True
