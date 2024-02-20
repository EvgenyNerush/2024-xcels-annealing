import numpy as np
import matplotlib.pyplot as plt
import functools
from annealing import (minimum_position)

# Тестируем написанный алгоритм на многомерных функционалах

n = 2 # x dimension
#start_x = np.random.random(n)
#initial_halfwidth = np.repeat(3.5,n)
#t_fade = 30
#initial_halfwidth = np.repeat(8,n)
#t_fade = 20

def target1(xs):
    ys = [x**2 for x in xs]
    return sum(ys)

def target2(xs):
    ampl = 0.1
    return target1(xs) + ampl - ampl * np.cos(2 * np.pi * xs[0] / 0.2)

def target3(xs):
    ys = [x**2 for x in xs]
    r = np.sqrt(sum(ys))
    return r**2 / (1 + 0.5 * r)

# ==== let's plot targets for one-dimensional x ====
p = 200
a = 4
xs = a * (np.arange(p) / p - 0.5)
plt.plot(xs, [target1([x]) for x in xs], label="1")
plt.plot(xs, [target2([x]) for x in xs], label="2")
plt.plot(xs, [target3([x]) for x in xs], label="3")
plt.legend()
plt.savefig("test_annealing_targets.png")
#plt.show()

# ==== let's try an example ====

start_x = np.random.random(n)
initial_halfwidth = [2.0 for _ in start_x]
x, f, err = minimum_position(target1, start_x, initial_halfwidth, t_fade = 20, attempts = 3)
print("x = {}".format(x))
print("f = {}".format(f))
print("error = {}".format(err))

## ==== let's try to find minimum of a target ====
## for different parameters
#
#print("*******************************")
#
#n_av = 100
#
#f_acc = 0
#err_acc = 0
#for i in range(n_av):
#    start_x = np.random.random(n)
#    initial_halfwidth = [1.5 for _ in start_x]
#    x, f, err = minimum_position(target1, start_x, initial_halfwidth, t_fade = 20, attempts = 1)
#    f_acc += f
#    err_acc += err
#print("*******************************")
#print("f = {}".format(f_acc / n_av))
#print("error = {}".format(err_acc / n_av))
#
#f_acc = 0
#err_acc = 0
#for i in range(n_av):
#    start_x = np.random.random(n)
#    initial_halfwidth = [2.0 for _ in start_x]
#    x, f, err = minimum_position(target1, start_x, initial_halfwidth, t_fade = 20, attempts = 1)
#    f_acc += f
#    err_acc += err
#print("*******************************")
#print("f = {}".format(f_acc / n_av))
#print("error = {}".format(err_acc / n_av))
#
#f_acc = 0
#err_acc = 0
#for i in range(n_av):
#    start_x = np.random.random(n)
#    initial_halfwidth = [3.0 for _ in start_x]
#    x, f, err = minimum_position(target1, start_x, initial_halfwidth, t_fade = 20, attempts = 1)
#    f_acc += f
#    err_acc += err
#print("*******************************")
#print("f = {}".format(f_acc / n_av))
#print("error = {}".format(err_acc / n_av))
#
