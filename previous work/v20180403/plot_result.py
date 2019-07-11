###############################################
## All Modes: 500                           ##
##############################################
import matplotlib.pyplot as plt
y_500_acc_noPunc = [0.4962, 0.56, 0.7525, 0.89, 0.895, 0.9237, 0.955, 0.945, 0.965, 0.9438, 0.9825, 0.9838, 0.9875, 0.9813, 0.9913, 0.9938, 0.9963, 0.995, 0.96, 0.9825, 0.9988, 0.9988, 0.9988, 0.9963, 0.9988, 1.0, 0.9988, 0.9988, 1.0, 1.0, 0.9988, 1.0, 0.9988, 1.0, 0.9975, 0.9988, 0.9988, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9988, 0.9988, 1.0, 1.0, 1.0]
y_500_search_noPunc = [0.4913, 0.4675, 0.6312, 0.8238, 0.87, 0.9112, 0.8962, 0.9463, 0.97, 0.96, 0.9675, 0.985, 0.97, 0.9775, 0.9888, 0.945, 0.9725, 0.995, 0.9625, 0.9675, 0.9975, 0.9925, 0.99, 0.9925, 0.9963, 0.9963, 0.9913, 0.9975, 0.9975, 0.9988, 0.9963, 0.9925, 0.9988, 0.9975, 1.0, 0.9988, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
y_1000_acc_noPunc = [0.5087, 0.51, 0.51, 0.6438, 0.9187, 0.96, 0.9725, 0.9738, 0.985, 0.9813, 0.9863, 0.9688, 0.9713, 0.975, 0.9925, 0.9963, 0.9938, 0.9875, 0.9938, 0.9863, 0.9875, 0.9925, 0.9975, 0.9988, 0.9975, 1.0, 0.9988, 0.9988, 0.9988, 0.9988, 0.9888, 0.9963, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
y_1000_search_noPunc = [0.5063, 0.63, 0.6625, 0.7575, 0.8738, 0.8975, 0.9262, 0.9575, 0.965, 0.9338, 0.985, 0.98, 0.9988, 1.0, 0.9988, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

x = range(1,51)
# plt.plot(x,y_200,'r',  x,y_500,'b',  x,y_1000,'g')
# plt.plot(x,y_200_acc_noPunc,'r', marker = 'o', label = "noPunc_Accurate Mode")
# plt.plot(x,y_200_search_noPunc,'b', marker = '>', label = "noPunc_Search Mode")
# plt.plot(x,y_200_acc_Punc,'g', marker = '*', label = "Punc_Accurate Mode")
# plt.plot(x,y_200_search_Punc,'c', marker = '.', label = "Punc_Search Mode")


fig, ax = plt.subplots(figsize = (70,50))
ax.plot(x,y_500_acc_noPunc,'r', marker = 'o', label = "500_noPunc_Accurate Mode")
ax.plot(x,y_500_search_noPunc,'b', marker = '>', label = "500_noPunc_Search Mode")
ax.plot(x,y_1000_acc_noPunc,'g', marker = '*', label = "1000_noPunc_Accurate Mode")
ax.plot(x,y_1000_search_noPunc,'c', marker = '.', label = "1000_noPunc_Search Mode")
# Plot circle, etc, then:
ax.set(xlim=[0, 10], ylim=[0.4, 1], aspect=50)


plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs number of Epoch (500_1000_NoPunc)')
plt.legend() # label in the bottom-right coner
plt.savefig('500_1000_noPunc.png')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #label outside the graph, next to the top-right coner
plt.show()
