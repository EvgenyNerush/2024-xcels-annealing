#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import datetime

# Расстояние между x и y. x и y --- списки одинаковой длины
def dist(x, y):
    diff = [(x[i] - y[i])**2 for i in range(len(x))]
    return math.sqrt(sum(diff))

# Генерируем нового кандидата. `from_x` and `halfwidth` --- списки одинаковой длины
def proposal(from_x, halfwidth):
    candidate = [random.uniform(from_x[i] - halfwidth[i], from_x[i] + halfwidth[i]) for i in range(len(from_x))]
    return candidate


# `t_fade` - характерное время уменьшения температуры
def temperature(t_fade, t):
    return math.exp( -t / t_fade )

# полуширина для proposal density
def halfwidth(initial_halfwidth, t_fade, t):
    return [x * temperature(t_fade, t) for x in initial_halfwidth]

# Generator of (x, t) pairs, which returns elements infinitely. Thus, the stop condition can be implemented independently. Also, the
# "restart" of the algorithm is possible at any point (e.g., with another t_fade).
def gen_to_minimum_position(f_to_minimize, start_x, initial_halfwidth, t_fade, start_t = 0):
    t, x = start_t, start_x
    # иначе мы дважды за шаг вызываем ресурсоёмкую f_to_minimize
    current_f = f_to_minimize(x)
    while True:
        candidate_x = proposal(x, halfwidth(initial_halfwidth, t_fade, t))
        new_f = f_to_minimize(candidate_x)
        acceptance_ratio = math.exp( (current_f - new_f) / temperature(t_fade, t) )
        if random.uniform(0,1) < acceptance_ratio:
            x = candidate_x
            current_f = new_f
        t += 1
        yield (t, x, current_f)

# return xs without duplicates and indices of the returned values
def remove_duplicates(xs):
    n = len(xs)
    js = [0]
    x = xs[0]
    for i in range(1, n):
        b = True
        for j in range(len(x)):
            b = b and (xs[i][j] == x[j])
        if not b:
            js.append(i)
            x = xs[i]
    return ([xs[j] for j in js], js)

assert (remove_duplicates([[1],[1],[1]])[0] == [[1]]), "remove_duplicates_1"
assert (remove_duplicates([[1],[1],[2],[3],[3],[4],[5],[5]])[0] == [[1],[2],[3],[4],[5]]), "remove_duplicates_2"
assert (remove_duplicates([[1],[1],[2],[3],[3],[4],[5],[5]])[1] == [0,2,3,5,6]), "remove_duplicates_2"

# Generator of x that corresponds to the position of f minimum.
# It tries to reach the desired error_level. The error level is computed as
# ratio of average gradient on the interval [0, t_fade) to that on the interval [t_fade, 2 * t_fade).
# If the desired error level is not reached during the first try, the second attempt is made.
#
# Returns (x, f(x), current_error) for x that gives minimal f, also logs (x, f(x)) pairs to file "minimum_position.lg".
def minimum_position(f_to_minimize, start_x, initial_halfwidth, t_fade = 20, attempts = 3, error_level = 0.4, first_grad = None, start_t = 0):
    gen = gen_to_minimum_position(f_to_minimize, start_x, initial_halfwidth, t_fade, start_t)
    seq = []
    with open("minimum_position.log", "a") as file:
        file.write("\n====================\n")
        file.write("{}\n".format(datetime.datetime.now()))
        file.write("attempts = {}\n".format(attempts))
        file.write("--------------------\n")
        file.write("x, f(x)\n")
        for _ in range (2 * t_fade):
            a = next(gen)
            file.write("{}, {}\n".format(a[1], a[2]))
            seq.append(a)
        _, seq_x, seq_f = tuple(zip(*seq))
        xs, js = remove_duplicates(seq_x)
        fs = [seq_f[j] for j in js]
        #
        js1 = [j for j in js if j < t_fade]
        n1 = len(js1)
        n2 = len(js)
        xs1 = xs[:n1]
        fs1 = fs[:n1]
        xs2 = xs[n1:n2]
        fs2 = fs[n1:n2]
        if len(xs1) > 1 and len(xs2) > 1:
            grads1 = [abs((fs1[i] - fs1[i+1]) / dist(xs1[i], xs1[i+1])) for i in range(n1 - 1)]
            grads2 = [abs((fs2[i] - fs2[i+1]) / dist(xs2[i], xs2[i+1])) for i in range(n2 - n1 - 1)]
            if first_grad == None:
                grad = sum(grads1) / (n1 - 1)
            else:
                grad = first_grad
            error = sum(grads2) / (n2 - n1 - 1) / grad
        else:
            error = 1
            grad = first_grad
        x = xs[0]
        f = fs[0]
        for i in range(1,len(xs)):
            if fs[i] < f:
                x = xs[i]
                f = fs[i]
        file.write("x = {}, f = {}, error = {}\n".format(x, f, error))
        file.close()
        if error <= error_level or attempts <= 1:
            return (x, f, error)
        else:
            # меняем параметры для следующего запуска
            hw = [1.5 * x for x in halfwidth(initial_halfwidth, t_fade, 2 * t_fade)] # 1.0 соответствует продолжению отжига без изменений
            tf = t_fade
            st = start_t + int(0.5 * 2 * t_fade) # прошло 2 t_fade, но меняя этот параметр, мы меняем температуру
            return minimum_position(f_to_minimize, x, hw, tf, attempts - 1, error_level, grad, start_t + 2 * t_fade)

