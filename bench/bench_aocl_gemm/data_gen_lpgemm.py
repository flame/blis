# Initializing global mnk_array.This array will be used to store all mnk values
mnk_array = []

max_elem = 2500;
out_file_name = "accuracy_test_data_lpgemm.txt"
# Important mnk generator function.This will generate all possible combinations
# of m,n,k values using formula m(t+1)=ROUND(m(t)*Base,0)+offset
def mnk_generator():
    k_1 = 1
    incr_k = 20
    while (k_1 <= max_elem):
        n_1 = 1
        incr_n = 20
        while (n_1 <= max_elem):
            m_1 = 1
            incr_m = 20
            while (m_1 <= max_elem):
                mnk_array.append([m_1, n_1, k_1])
                if (m_1 == 1):
                    m_1 = m_1 + 9
                else:
                    m_1 = m_1 + incr_m
            if (n_1 == 1):
                n_1 = n_1 + 9
            else:
                n_1 = n_1 + incr_n
        if (k_1 == 1):
            k_1 = k_1 + 9
        else:
            k_1 = k_1 + incr_k

def data_gen():
    mnk_generator()

    fout = open(out_file_name, "w")

    for ele in mnk_array:
        fout.write("i r " + str(ele[0]) + " " + str(ele[1]) + " " + str(ele[2]) + " " +\
                str(ele[2]) + " " + str(ele[1]) + " " + str(ele[1]) + "\n")

    fout.truncate(fout.tell() - 1)
    fout.close()

##__main__
data_gen()
