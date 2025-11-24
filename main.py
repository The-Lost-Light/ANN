import lib
from hopfield import HopfieldNetwork

train_data, test_data, p = lib.load_patterns("Basic", 12)
model_base = HopfieldNetwork(12 * p)
model_base.train(train_data, threadhold=True)
string = lib.list2string(model_base.predict(test_data[0], asynchronous=True), p)
print(string)

print("")

train_data, test_data, p = lib.load_patterns("Bonus", 10)
model_bonus = HopfieldNetwork(10 * p)
model_bonus.train(train_data, threadhold=True)
string = lib.list2string(model_bonus.predict(test_data[0], asynchronous=True), p)
print(string)
