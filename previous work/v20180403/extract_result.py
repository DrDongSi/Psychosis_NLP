# file = open("result-200_acc_noPunc.txt","r") 
# file = ['500_accurate_nopunc_result.txt', '500_accurate_punc_result.txt', 
#         '500_search_nopunc_result.txt', '500_search_punc_result.txt' ]


file = ['500_accurate_nopunc_UNbalanced_result.txt', '500_accurate_punc_UNbalanced_result.txt',
        '500_search_nopunc_UNbalanced_result.txt', '500_search_punc_UNbalanced_result.txt']
file = ["500_accurate_nopunc_UNbalanced_result.txt"]

# file = ['1000_accurate_nopunc_result.txt', '1000_accurate_punc_result.txt', 
#         '1000_search_nopunc_result.txt', '1000_search_punc_result.txt' ]


# file = ['result-50ep-150tk-5w-5m-acc-np.txt', 'result-50ep-150tk-5w-5m-acc-p.txt', 
#        'result-50ep-150tk-5w-5m-sch-np.txt', 'result-50ep-150tk-5w-5m-sch-p.txt', 
#        'result-50ep-200tk-5w-5m-acc-np.txt', 'result-50ep-200tk-5w-5m-acc-p.txt', 
#        'result-50ep-200tk-5w-5m-sch-np.txt', 'result-50ep-200tk-5w-5m-sch-p.txt']




# file = ['ub500-50ep-150tk-5w-5m-acc-np.txt', 'ub500-50ep-150tk-5w-5m-acc-p.txt', 
#         'ub500-50ep-150tk-5w-5m-sch-np.txt', 'ub500-50ep-150tk-5w-5m-sch-p.txt', 
#         'ub500-50ep-200tk-5w-5m-acc-np.txt', 'ub500-50ep-200tk-5w-5m-acc-p.txt',
#         'ub500-50ep-200tk-5w-5m-sch-np.txt', 'ub500-50ep-200tk-5w-5m-sch-p.txt'
        
#        ]


# file = ['ub500-50ep-500tk-5w-5m-acc-np.txt', 'ub500-50ep-500tk-5w-5m-acc-p.txt',
#         'ub500-50ep-500tk-5w-5m-sch-np.txt', 'ub500-50ep-500tk-5w-5m-sch-p.txt'
#        ]

# print("abc")
for fname in file:
#     print(fname)
    string = "step -"
    loss = "loss"
    acc = "acc"
    val_loss = "val_loss"
    val_acc = "val_acc"
    
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    with open(fname) as f:
        line = f.readline()
        while line:

            if (line.find(string)>0):
                result = line[line.find(string):]
                loss_index = result.find(loss)
                acc_index = result.find(acc)
                
                val_loss_index = result.find(val_loss)
                val_acc_index = result.find(val_acc)
#                 print(result)

                if (result.find(loss)>0):
                    temp = result.split(':',4)
                    for i in range(1,5):
                        
                        
#                         print(temp[i])
                        result = temp[i].split(' ')
#                         print(result)
                        if i == 1:
                            loss_list.append(float(result[1]))
#                             print (result)
                        if i == 2:
                            acc_list.append(float(result[1]))
#                             print(result)
                        if i == 3:
                            val_loss_list.append(float(result[1]))
#                             print(result)
                        if i == 4:
                            val_acc_list.append(float(result[1]))
#                             print(result)
#                     print("abc")
#                 loss_temp = result[loss_index+6:loss_index+12]
#                 acc_temp = result[acc_index+5:acc_index+11]
#                 val_loss_temp = result[val_loss_index+10:val_loss_index+16]
#                 val_acc_temp = result[val_acc_index+9:val_acc_index+15]
                
#                 if (result.find(loss)>0):
#                     loss_list.append(float(loss_temp))

#                 if (result.find(acc)>0):
#                     acc_list.append(float(acc_temp))
                    
#                 if (result.find(val_loss)>0):
#                     val_loss_list.append(float(val_loss_temp))
                    
                    
#                 if (result.find(val_acc)>0):
#                     val_acc_list.append(float(val_acc_temp))

            line = f.readline()
            
            
    print("file name:  " + fname)
    print("accuracy")
    print(acc_list)
    
    print("loss")
    print(loss_list)

    
    print("val_acc")
    print(val_acc_list)
    
    print("val_loss")
    print(val_loss_list)
#     print()
#     print()