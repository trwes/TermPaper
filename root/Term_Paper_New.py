from math import sin, cos, e, pi, log
import matplotlib.pyplot as plt
import scipy.optimize
import time



def custom_func(func_str):
    def result_func(x):
        nonlocal func_str
        func_str = func_str.replace('^', '**')
        return eval(func_str.replace('x', '(' + str(float(x)) + ')'))
    return result_func


def add_superscript(string):
    string = string.replace('**', '^').replace(' ', '') + '+'
    new_str = string[:]
    if '^' not in string:
        return string[:-1]
    else:
        start_copy = False
        current_power = ''
        semaphore = 0
        for char in string:
            if char == '^' and start_copy is False:
                start_copy = True

            if char == '(' and start_copy is True:
                semaphore += 1
                continue
            elif char == ')' and start_copy is True:
                semaphore -= 1
                continue

            if semaphore == 0 and char in ['-', '+', '*', '/']:
                start_copy = False

            if start_copy is True:
                current_power += char
            elif start_copy is False and len(current_power) > 0:
                current_power = current_power[1:]
                new_str = new_str.replace(f'^{current_power}', f'<sup>{add_superscript(current_power)}</sup>',  1)\
                    .replace(f'^({current_power})', f'<sup>{add_superscript(current_power)}</sup>',  1)

                current_power = ''
        return new_str[:-1]


def preProc(func_str):  # Протестовано + працює з y
    # Pre Processing
    # Результат - список доданків многочлена
    func_str = func_str.replace('e+-', 'e-')
    func_str = func_str.replace('**', '^')
    func_str = func_str.replace('e+', '?')
    func_str = func_str.replace('e-', '&')
    func_str = func_str.replace('-', '+-')
    semaphore = 0
    new_str = ''
    # Замінюємо "+" не в дужках на "$", щоб в подальшому розбити многочлен на доданки
    for char in func_str:
        if char == '(':
            semaphore += 1
        elif char == ')':
            semaphore -= 1
        if semaphore == 0 and char == '+':
            new_str += '$'
        else:
            new_str += char
    new_str = new_str.replace('?', 'e+').replace('&', 'e-')
    terms = new_str.split('$')

    while '' in terms:
        terms.remove('')
    return terms


def addition_of_array(arr):  # Протестовано
    res = ''
    for el in arr:
        res += el + "+"
    if len(res) > 0:
        res = res[:-1]
    return res.replace('+-', '-')


def symbol_multiplication(term):  # Протестовано + працює з y
    # символьне множення в доданку. Par exemple: x*x*x = x^3
    term = term.replace("**", "^").replace("e+-", "e-")
    term = term.replace("e-", "?").replace("e+", "#")
    tester_str = term.replace('(-', '#').replace('^-', '#').replace('*-', '#').replace('/-', '#')
    if tester_str[0] == '-':
        tester_str = tester_str[1:]
    if '-' in tester_str:
        raise TypeError('Неправильне використання функції: додавання і віднімання не допускається')
    while '(' in term:
        in_paranthesis = ''
        start_copy = False
        for char in term:
            if char == ')':
                break
            if start_copy is True:
                in_paranthesis += char
            if char == '(':
                start_copy = True
        inner_multiplication = symbol_multiplication(in_paranthesis)
        term = term.replace('(' + in_paranthesis + ')', inner_multiplication, 1)

    if '+' in term:
        raise TypeError('Неправильне використання функції: "+" не допускається')
    term = term.replace('-x', '-1*x')
    mul_arr = term.split("*")
    for mul in mul_arr:
        if '(' in mul:
            mul = mul[1:-1]
            if len(preProc(mul)) >= 2:
                raise TypeError('Неправильне використання функції: додавання і віднімання не допускається')
    _power = 0
    _coef = 1

    for expr in mul_arr:
        expr = expr.replace("?", "e-").replace("#", "e+")
        if 'x' in expr:
            if '^' in expr:
                if expr[0] == '(' and expr[-1] == ')':
                    expr = expr[1:-1]
                circonflex_inx = expr.index('^')
                _power += float(eval(expr[circonflex_inx+1:].replace('^', '**')))
            else:
                _power += 1
        else:
            try:
                _coef *= float(eval(expr.replace('^', '**')))
            except:
                _coef = str(_coef)+'*'+str(expr)
    if float(_power) != 0:
        return f'{_coef}*x^{_power}'
    else:
        return f'{_coef}'


def term_multiplication(term):  # Протестовано + працює з y але TODO дужка в дужці як ((1/2)*x+1)*(x+2)
    # Множення доданків. Par example: x*(x+2) = x^2 + 2*x^1 ou (x+1)*(x+2) = x^2+3*x^1+2
    # base case - немає дужок, немає, що множити або 2 множники
    try:
        res = symbol_multiplication(term)
        return res
    except:
        pass
    if '(' not in term and ')' not in term:  # TODO процессинг строки
        return term
    elif '*' not in term:
        return term

    semaphore = 0
    buffer = ''
    for char in term:
        if char == '(':
            semaphore += 1
        elif char == ')':
            semaphore -= 1

        if char == '*' and semaphore == 0:
            break
        buffer += char
    if buffer+'*' not in term:
        second_multiplier = '1'
    else:
        second_multiplier = term.replace(buffer+'*', '', 1)
    result_arr = []
    # Може бути 3 випадки: перший множник в дужках або ні, або є множення між дужками
    if buffer[0] == '(' and buffer[-1] == ')' and ')*(' not in buffer:
        buffer = buffer
        buffer = buffer[1:-1]
        buf_terms = preProc(buffer)
    else:
        buf_terms = [buffer]
    # Те ж саме і з другим множником
    if second_multiplier[0] == '(' and second_multiplier[-1] == ')' and ')*(' not in second_multiplier:
        second_multiplier = second_multiplier
        second_multiplier = second_multiplier[1:-1]
        second_multiplier_terms = preProc(second_multiplier)
    else:
        second_multiplier_terms = [second_multiplier]
    for mul1 in buf_terms:
        mul1 = term_multiplication(mul1)
        if '+' in mul1 or '-' in mul1:
            mul1 = '(' + mul1 + ')'
        for mul2 in second_multiplier_terms:
            mul2 = term_multiplication(mul2)
            if '+' in mul2 or '-' in mul2:
                mul2 = '(' + mul2 + ')'
            step = term_multiplication(mul1+'*'+mul2)
            result_arr.append(step)

    res = ''
    for el in result_arr:
        res += el+'+'
    return res[:-1].replace('*x^0.0', '').replace('*x^0', '').replace('+-', '-')


def sum_up(func):
    # Увага: вирази не можуть містити степенів типу (...)^n
    terms = preProc(func)
    _coefs = []
    _powers = []
    for term in terms:
        term = term.replace('+-', '-')
        term = term_multiplication(term)
        inner_terms = preProc(term)
        for expr in inner_terms:
            expr = expr.replace('+-', '-')
            if 'x^' not in expr:
                _coefs.append(float(expr))
                _powers.append(0)
            else:
                circon_inx = expr.index('^')
                _powers.append(float(expr[circon_inx+1:]))
                x_inx = expr.index('x')
                _coefs.append(float(expr[:x_inx-1]))
    result = []
    is_checked = []
    for i in range(len(_powers)):
        current_power = _powers[i]
        if current_power in is_checked:
            continue
        current_coef = _coefs[i]
        for j in range(i+1, len(_powers)):
            if current_power == _powers[j] and _powers[j] not in is_checked:
                current_coef += float(_coefs[j])
        is_checked.append(current_power)
        result.append((current_coef, current_power))
    res = ''
    for tup in result:
        res += f'{tup[0]}*x^{tup[1]}+'
    if len(res) > 0:
        res = res[:-1]

    return res.replace('*x^0.0', '').replace('*x^0', '')


def pascal(n):
    # повертає коефіцієнти бінома ньютона порядку n
    if float(n)-int(n) != 0.0:
        raise TypeError
    if n == 0:
        return [1]
    elif n > 0:
        pascal_n_1 = [0] + pascal(n-1) + [0]
        pascal_n = [pascal_n_1[i]+pascal_n_1[i+1] for i in range(len(pascal_n_1)-1)]
        return pascal_n
    else:
        raise TypeError


def binomial(term):
    # Біном ньютона. Par example: (x+y)^2 = x^2+2*x*y+y^2
    # Увага: подавати лише вирази (...)^n, n є N. Ділення не допускається
    term = term.replace('e-', '$').replace('e+', '%')
    n = 0
    semaphore = 0
    for i in range(len(term)):
        if term[i] == '(':
            semaphore += 1
        elif term[i] == ')':
            semaphore -= 1
        if term[i] == '^' and semaphore == 0:
            n = float(term[i+1:])
            break
    if n - int(n) != 0:
        raise TypeError

    if ')^' not in term:   # Немає сенсу підносити у степінь те, де немає піднесення
        return term.replace('$', 'e-').replace('%', 'e+')
    # Base case: тільки 1 доданок в дужках
    pluses = 0
    first_char_is_minus = False
    simple_case = False
    for k in range(len(term)):
        char = term[k]
        if char == '+' or char == '-':
            pluses += 1
        if k == 1 and char == '-':
            first_char_is_minus = True
    if pluses == 1 and (first_char_is_minus is True):
        simple_case = True
    if pluses == 0 or (simple_case is True):  # Якщо в дужках немає плюсів, то це просто піднесення в степінь
        internal = ''
        start_copy = False
        for char in term:
            if char == ')':
                break
            if start_copy is True:
                internal += char
            if char == '(':
                start_copy = True
        internal = sum_up(internal.replace('$', 'e-').replace('%', 'e+'))
        if 'x' in internal:
            spl = internal.split('*')
            int_coef = float(spl[0])**n
            circon_inx = spl[1].index('^')
            power = spl[1][circon_inx+1:]
            int_power = float(power)*n
            return f'{int_coef}*x^{int_power}'
        else:
            return f'{float(internal)**n}'
    else:
        semaphore = -1
        a = ''
        is_plus = True
        for i in range(len(term)):
            char = term[i]
            if (char == '+' or char == '-') and semaphore == 0 and a != '':
                if char == '-':
                    is_plus = False
                break
            if semaphore >= 0:
                a += char
            if char == '(':
                semaphore += 1
            elif char == ')':
                semaphore -= 1
        base = ''
        semaphore = 0
        for char in term:
            if char == '^' and semaphore == 0:
                break
            if semaphore >= 1:
                base += char
            if char == '(':
                semaphore += 1
            elif char == ')':
                semaphore -= 1
        if is_plus is True:
            b = base.replace(a + '+', '', 1)[:-1]  # Бо в кінці ще закрита дужка
        else:
            b = base.replace(a, '', 1)[:-1]  # Бо в кінці ще закрита дужка
        coefs = pascal(n)
        plein_terms = []
        for k in range(int(n)+1):
            pow1 = n-k
            pow2 = k
            res = ''
            if pow1 != 0:
                res += '(' + binomial(f'({a})^{pow1}') + ')'
            if pow1 != 0 and pow2 != 0:
                res += '*'
            if pow2 != 0:
                res += '(' + binomial(f'({b})^{pow2}') + ')'
            plein_terms.append(res)
        for i in range(len(plein_terms)):
            plein_terms[i] = plein_terms[i].replace('$', 'e-').replace('%', 'e+')
            plein_terms[i] = sum_up(f'{coefs[i]}*'+(plein_terms[i]))
        answer = ''
        for el in plein_terms:
            answer += el + '+'
        return sum_up(answer[:-1]).replace('+-', '-')  # sum up не було


def prepare_polynomial(func):
    terms = preProc(func)
    res_arr = []
    for term in terms:
        semaphore = 0
        term = term+'#'
        new_term = ''
        last_char = ''
        for char in term:
            if char == '(':
                semaphore += 1
            elif char == ')':
                semaphore -= 1

            if last_char == ')' and char != '^' and semaphore == 0:  # Тобто ми за дужками, за якими немає степені
                new_term += '^1'
            if char == '*' and semaphore == 0:
                new_term += '&'
            elif char != '#':
                new_term += char
            last_char = char

        multiplicateurs = new_term.split('&')
        binoms = []
        for mult in multiplicateurs:
            mult = mult.replace('+-', '-')
            binoms.append(binomial(mult))

        retrouver = ''
        for el in binoms:
            retrouver += '(' + el + ')*'
        if len(retrouver) > 0:
            retrouver = retrouver[:-1]
        result = term_multiplication(retrouver)
        res_arr.append(result)

    added = addition_of_array(res_arr)

    return sum_up(added)


def derivative(function):
    function = function.replace('**', '^')
    terms = preProc(function)
    parenthesis_counter = 0
    for char in function:
        if char == '(':
            parenthesis_counter += 1
    if parenthesis_counter == 0:
        # Case principal: немає дужок - тут може бути тільки поліном, бо par example: sin(...) містить дужки
        # Знаходження степенів
        # Todo від'ємні степені, записані, як 1/x
        powers = []
        for i in range(len(terms)):
            if terms[i] == '':
                terms[i] = '0'
            terms[i] = symbol_multiplication(terms[i])
        for i in range(len(terms)):
            if 'x' not in terms[i]:
                powers.append('0')
                continue
            if 'x^' not in terms[i]:
                powers.append('1')
                continue
            power = ''
            for j in range(len(terms[i])):
                if terms[i][j] == '^' and terms[i][j-1] == 'x':
                    for k in range(j+1, len(terms[i])):
                        if terms[i][k] in ['+', '-', '/', '*']:
                            break
                        power += terms[i][k]
                    break
            powers.append(power)
        # Знаходження коефіцієнтів
        coefs = []
        for i in range(len(terms)):
            terms[i] = terms[i].replace('+-', '-').replace('--', '+')

            if 'x' not in terms[i]:
                coefs.append(terms[i])
                continue
            if 'x^' not in terms[i]:
                terms[i] = terms[i].replace('x', 'x^1.0')
            buffer_term = terms[i].replace(f"x^{powers[i]}", '', 1).replace(f"x^{float(powers[i])}", '')
            if buffer_term == '':
                coefs.append(1)
                continue
            if buffer_term[0] in ['*', '/']:
                buffer_term = '1' + buffer_term
            if buffer_term[-1] in ['*', '/']:
                buffer_term = buffer_term + '1'
            if buffer_term[0] == '-' and buffer_term[1] in ['*', '/']:
                buffer_term = '-1' + buffer_term[1:]
            buffer_term = buffer_term.replace('*/', '/')
            try:
                coefs.append(eval(buffer_term))
            except:
                coefs.append(buffer_term)
        characteristics = []
        for i in range(len(powers)):
            if powers[i] == '':
                power = 0
            else:
                power = powers[i]
            characteristics.append((coefs[i], power))  # coef, power, monom
        result = ''
        for tup in characteristics:
            if float(tup[1]) == 0:
                continue
            if type(tup[0]) != str:
                new_coef = tup[0] * float(tup[1])
            else:
                new_coef = str(tup[0]) + '*' + str(tup[1])
            new_pow = float(tup[1]) - 1
            if str(new_coef) == '0' or str(new_coef) == '0.0':
                result += '+0'
            else:
                if new_pow != 0:
                    result += f"+{new_coef}*x^{new_pow}"
                else:
                    result += f"+{new_coef}"
        result = result[1:]
        result = result.replace('+-', '-').replace('--', '+')

        if result == '':
            return '0'
        else:
            return result
    else:  # Якщо дужки є, то це "складніша" функція
        result_arr = []
        is_complicated = False

        for func in terms:
            parenthesis_counter_sep = 0
            buffer = ''
            for char in func:
                if char == '(':
                    parenthesis_counter_sep += 1
            if parenthesis_counter_sep == 0:
                result_arr.append(derivative(func))
            else:  # Якщо в цьому доданку знайдено дужки
                semaphore_mul_div = 0
                buffer_mul_div = ''
                for char in func:  # Шукаємо знаки * та / поза дужками, щоб застосувати диференціювання добутку/частки
                    if char == '(':
                        semaphore_mul_div += 1
                    elif char == ')':
                        semaphore_mul_div -= 1
                    if char == '*' and semaphore_mul_div == 0:
                        # Маємо частину функції від початку до першої * не в дужках (не включаючи *)
                        is_complicated = True  # Це складна функція (...)*(...)
                        u = buffer_mul_div.replace('e+-', 'e-')
                        v = func[len(buffer_mul_div)+1:].replace('e+-', 'e-')  # TODO Зробити це ж в частині ділення
                        du = derivative(u)
                        dv = derivative(v)

                        if str(du) == '0' or str(v) == '0':
                            du_v = '0'
                        else:
                            if "+" in du:  # TODO Зробити це ж в частині ділення + перевірка семафором
                                du_v = '(' + du + ')*' + v
                            else:
                                du_v = du + '*' + v

                        if str(dv) == '0' or str(u) == '0':
                            u_dv = '0'
                        else:
                            if '+' in dv:
                                u_dv = u + '*(' + dv + ')'
                            else:
                                u_dv = u + '*' + dv

                        result_arr.append(du_v)
                        result_arr.append(u_dv)
                        break  # TODO Зробити це ж в частині ділення

                    elif char == '/' and semaphore_mul_div == 0:
                        # Маємо частину функції від початку до першої / не в дужках (не включаючи /)
                        is_complicated = True  # Це складна функція (...)/(...)
                        u = buffer_mul_div
                        v = func.replace(buffer_mul_div + '/', '')
                        du = derivative(u)
                        dv = derivative(v)
                        if str(du) == '0' or str(v) == '0':
                            du_v = '0'
                            result_arr.append(du_v)
                        else:
                            du_v = du + '*' + v
                            result_arr.append('(' + du_v + ')/(' + v + ')^2')

                        if str(dv) == '0' or str(u) == '0':
                            u_dv = '0'
                            result_arr.append(u_dv)
                        else:
                            u_dv = u + '*' + dv
                            result_arr.append('(-' + u_dv + ')/(' + v + ')^2')

                    buffer_mul_div += char

                semaphore = 0
                for char in func:
                    if char == '(':
                        semaphore += 1
                        continue
                    if char == ')':
                        semaphore -= 1
                        break

                    if semaphore >= 1:
                        buffer += char
            func = func.replace('('+buffer+')', '$').replace('e+-', 'e-')
            buffer = buffer.replace('e+-', 'e-')
            if derivative(buffer) != '0' and is_complicated is False:
                func = func.replace('sin$', f'cos({buffer})*'+'('+derivative(buffer)+')')
                func = func.replace('cos$', f'-sin({buffer})*' + '(' + derivative(buffer) + ')')
                func = func.replace('e^$', f'e^({buffer})*' + '(' + derivative(buffer) + ')')
                func = func.replace('log$', f'(1/({buffer}))*' + '(' + derivative(buffer) + ')')
                # TODO tg, ctg, a^x, hyperbolic...
                if '$^' in func:
                    inx = func.index('$')
                    powers = float(eval(func[inx+2:].replace('^', "**")))
                    coef = func[:inx]
                    if coef == '':
                        coef = '1'

                    if powers != 1:
                        func = str(float(coef)*powers)+'*('+buffer+f')^({powers-1})*(' + derivative(buffer) + ')'
                    else:
                        func = str(float(coef)*powers) + f'*(' + derivative(buffer) + ')'
                    # TODO тип x^x
                func = func.replace('$', '(' + derivative(buffer) + ')')
                result_arr.append(func)
            else:
                if '^$' in func:  # Тобто це вираз типу х^(2)
                    result_arr.append(derivative(func.replace('^$', f'^{eval(buffer)}')))  # TODO можуть бути помилки
                else:
                    result_arr.append('0')

        result = '0'
        while '0' in result_arr:
            result_arr.remove('0')

        if len(result_arr) == 1:
            result = result_arr[0]
        elif len(result_arr) >= 2:
            result = result_arr[0]
            for term_der in result_arr[1:]:
                if str(term_der) != '0':
                    result += '+'+term_der
        return result.replace('+-', '-').replace('--', '+')


def n_Derivative(func, n, x0):
    result = derivative(func)
    for i in range(n-1):
        result = derivative(result)
    return eval(result.replace('^', '**').replace('+-', '-').replace('x', str(x0)))


def find_y_partial_derivative(func):
    func = func.replace('x', '$').replace('y', 'x').replace('$', 'y')
    dy = derivative(func)
    dy = dy.replace('x', '$').replace('y', 'x').replace('$', 'y')
    return dy


def create_func_of_xy(func):
    def two_vars_func(arr):
        nonlocal func
        try:
            return eval(func.replace('x', str(arr[0])).replace('y', str(arr[1])))
        except OverflowError:
            return 10**10
    return two_vars_func


def RK4(func, _x0, _y0, _a, _b, h=0.001):
    func = func.replace('^', '**')
    f = create_func_of_xy(func)
    x_list = [_x0]
    y_list = [_y0]

    it = 1
    while (_a+it*h) <= _b:
        xn = x_list[-1]
        yn = y_list[-1]
        xnh2 = xn + h/2

        k1 = f([xn, yn])
        k2 = f([xnh2, yn+(h/2)*k1])
        k3 = f([xnh2, yn+(h/2)*k2])
        k4 = f([xn+h, yn+h*k3])

        new_y = yn + (h/6)*(k1+2*k2+2*k3+k4)
        x_list.append(xn+h)
        y_list.append(new_y)
        it += 1
    return x_list, y_list


def find_d_and_alpha(_a, _b, _x0, _y0, func, epsilon=10**-8):  # Але повертає інтервал інтегрування: по х і у та альфа
    counter_unsuccessful = 0
    func = func.replace('^', '**')
    if _x0 < _a or _x0 > _b:
        raise TypeError("C'est pas possible: x0 doit etre entre a et b")
    a_of_x0 = max(abs(_a-_x0), abs(_b-_x0))
    _q = 1
    end = False
    while end is False:
        b_of_y0 = _q*a_of_x0
        bonds_x = (_x0 - a_of_x0, _x0 + a_of_x0)
        bonds_y = (_y0 - b_of_y0, _y0 + b_of_y0)
        new_func = create_func_of_xy('-abs(' + func + ')')
        try:
            res = scipy.optimize.minimize(new_func, x0=[_x0, _y0], bounds=[bonds_x, bonds_y])
        except ZeroDivisionError:
            return "Метод розбіжний", "Метод розбіжний", "Метод розбіжний"
        K = -1*res.fun
        func_dy = '-abs(' + find_y_partial_derivative(func).replace('^', '**') + ')'
        new_func_dy = create_func_of_xy(func_dy)
        try:
            res2 = scipy.optimize.minimize(new_func_dy, x0=[_x0, _y0], bounds=[bonds_x, bonds_y])
        except ZeroDivisionError:
            return "Метод розбіжний", "Метод розбіжний", "Метод розбіжний"
        M = -1 * res2.fun
        if M > 100500:
            return "Метод розбіжний", "Метод розбіжний", "Метод розбіжний"
        try:
            one_over_M = 1/M
        except ZeroDivisionError:
            one_over_M = 100500
        if a_of_x0 < one_over_M:  # Значить можемо покращити ще трохи а або б
            try:
                qk = _q / K
            except:
                qk = 1000
            if qk < 1:
                _q = K
                continue
            else:
                d = a_of_x0
                alpha = M * d
                return (_x0 - d, _x0 + d), (_y0 - K * d, _y0 + K * d), alpha
        else:  # Значить а (і б) > 1/M - далі збільшувати q немає сенсу
            counter_unsuccessful += 1
            d = one_over_M * 0.95
            if 2*d < 0.5 and counter_unsuccessful < 20:
                a_of_x0 = a_of_x0/2
                continue
            alpha = M * d
            return (_x0 - d, _x0 + d), (_y0 - K*d, _y0 + K*d), alpha


def integrate_polynomial(poly_str):
    terms = preProc(poly_str)
    while '' in terms:
        terms.remove('')
    # Заміна (х-x0) на (х-x0)^1
    for i in range(len(terms)):  # TODO Це тимчасове рішення - зробити красивіше потім
        for j in range(len(terms[i])):
            try:
                if terms[i][j] == 'x' and terms[i][j+1] != '^':
                    terms[i] = terms[i].replace('x', 'x^1')
            except IndexError:
                terms[i] = terms[i].replace('x', 'x^1')
        for j in range(len(terms[i])):
            try:
                if terms[i][j] == ')' and terms[i][j+1] != '^':
                    terms[i] = terms[i].replace(')', ')^1')
            except IndexError:
                terms[i] = terms[i].replace(')', ')^1')

    # Знаходження степенів
    powers = []
    for i in range(len(terms)):
        power = ''
        semaphore_1 = 0
        for j in range(len(terms[i])):
            if terms[i][j] == '(':
                semaphore_1 += 1
            elif terms[i][j] == ')':
                semaphore_1 -= 1
            if terms[i][j] == '^' and semaphore_1 == 0:
                semaphore = 0
                for k in range(j+1, len(terms[i])):
                    if terms[i][k] == '(':
                        semaphore += 1
                    if terms[i][k] == ')':
                        semaphore -= 1
                    if terms[i][k] in ['+', '-', '/', '*'] and semaphore == 0:
                        break
                    power += terms[i][k]
                break
        powers.append(power)

    # Знаходження коефіцієнтів
    coefs = []
    monoms = []
    for i in range(len(terms)):
        terms[i] = terms[i].replace('+-', '-').replace('--', '+')
        # Шукаємо х0 в (х-х0)
        _x0 = ''
        buffer_str = terms[i].replace('(x', '$').replace(')', '#')
        start_copy = False
        for char in buffer_str:
            if char == '#':
                start_copy = False
            if start_copy is True:
                _x0 += char
            if char == '$':
                start_copy = True
        monoms.append(f'(x{_x0})')

        buffer_term = terms[i].replace(f"(x{_x0})^{powers[i]}", '').replace(f"x^{powers[i]}", '')
        if buffer_term == '':
            coefs.append(1)
            continue
        if buffer_term[0] in ['*', '/']:
            buffer_term = '1'+buffer_term
        if buffer_term[-1] in ['*', '/']:
            buffer_term = buffer_term+'1'
        if buffer_term[0] == '-' and buffer_term[1] in ['*', '/']:
            buffer_term = '-1'+buffer_term[1:]
        buffer_term = buffer_term.replace('*/', '/')
        coefs.append(eval(buffer_term))

    characteristics = []
    for i in range(len(powers)):
        if powers[i] == '':
            power = 0
            monoms[i] = 'x'
        else:
            power = eval(powers[i])
        characteristics.append((coefs[i], power, monoms[i]))  # coef, power, monom

    result = ''
    for tup in characteristics:
        new_pow = tup[1]+1
        new_coef = tup[0]/new_pow
        new_monom = tup[2]
        if new_coef == 1:
            result += f"+{new_monom}^{new_pow}"
        elif new_coef == 0:
            pass
        else:
            result += f"+{new_coef}*{new_monom}^{new_pow}"

    result = result[1:]
    result = result.replace('+-', '-').replace('(x)', 'x')
    if result == '':
        result = '0'
    return result


def newton_leibnitz(integr_str, _a):
    # Формула Ньютона-Лейбніца "від а до х"
    return (integr_str + '-' + str(eval(integr_str.replace('^', "**").replace('x', str(_a))))).replace('-+', '-')\
        .replace('--', '+')


def factorial(n):
    # Обчислення n!
    if n == 0:
        return 1
    return n*factorial(n-1)


def taylor(func_str, x0, n):
    # Розкладення функції func_str у ряд Тейлора до порядку n у точці х0
    func = custom_func(func_str)
    result = f'{func(x0)}'
    for i in range(1, n+1):
        coef = n_Derivative(func_str, i, x0)/factorial(i)  # round(zentral(func, x0,0.05+0.05*i,i) / factorial(i), 6)
        if coef != 0:
            result += f'+{coef}*(x-{x0})^{i}'

    return result.replace('+-', '-').replace('--', '+')


def plot_piecewise(func_arr, density=25, x_y_conv=None):
    x_list = []
    y_list = []
    fig = plt.figure()
    for tup in func_arr:
        step = (tup[1][1]-tup[1][0])/density
        tup_func = custom_func(tup[0])
        x_tup_list = [(tup[1][0]+i*step) for i in range(density)]
        y_tup_list = [tup_func(x) for x in x_tup_list]

        x_list += x_tup_list
        y_list += y_tup_list

    if x_y_conv != None:
        plt.axvline(x=x_y_conv[0], label='Область збіжності', color='pink', linestyle='-.')
        plt.axvline(x=x_y_conv[1], color='pink', linestyle='-.')

    plt.plot(x_list, y_list, label='Графік інтегральної кривої рівняння')
    plt.title("Застосування загального алгоритму")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    return fig


def picard(dydx, _x0, _y0, _a, _b, alpha, convergence=None):
    # Функція, що реалізовує метод Пікара для загальних функцій
    if convergence != None:
        _a = min(_a, convergence[0])
        _b = max(_b, convergence[1])

    step = round((_b-_a)/50, 2)
    result = []

    n_iter = 4

    previous_iteration_list2 = []

    x_taylor = _x0
    y0_approx = _y0
    it = 0
    while _x0 + step*it <= _b:
        to_integrate = dydx.replace('y', str(y0_approx))
        to_integrate = taylor(to_integrate, x_taylor, 8)  # Розкладення в ряд Тейлора підінтегральної функції (де у уже замінено)

        y_s_1 = y0_approx
        for i in range(n_iter):
            y_s = newton_leibnitz(integrate_polynomial(to_integrate), x_taylor) + '+' + str(y0_approx)
            to_integrate = taylor(dydx.replace('y', y_s), x_taylor, 8)
            if y_s == y_s_1 or i == n_iter:
                result.append((y_s, [_x0 + step*it, _x0 + step*(it+1)]))
                previous_iteration_list2.append((y_s_1, [_x0 + step*it, _x0 + step*(it+1)]))
                break
            else:
                y_s_1 = y_s
        it += 1
        x_taylor = _x0 + step*it
        y0_approx = eval(result[-1][0].replace('x', f"{x_taylor}").replace('^', '**'))

    res2 = []
    previous_iteration_list1 = []
    it = 0
    x_taylor = _x0
    y0_approx = _y0
    while _x0 - step*it >= _a:
        to_integrate = dydx.replace('y', str(y0_approx))
        to_integrate = taylor(to_integrate, x_taylor, 8)  # Розкладення в ряд Тейлора підінтегральної функції (де у уже замінено)

        y_s_1 = y0_approx
        for i in range(n_iter):
            y_s = newton_leibnitz(integrate_polynomial(to_integrate), x_taylor) + '+' + str(y0_approx)
            to_integrate = taylor(dydx.replace('y', y_s), x_taylor, 8)
            if y_s == y_s_1 or i == n_iter:
                res2.append((y_s, [_x0 - step*(it+1), _x0 - step*it]))
                previous_iteration_list1.append((y_s, [_x0 - step*(it+1), _x0 - step*it]))
                break
            else:
                y_s_1 = y_s
        x_taylor = _x0 - step*it
        it += 1
        y0_approx = eval(res2[-1][0].replace('x', f"{x_taylor}").replace('^', '**'))
    res2.reverse()
    result = res2 + result

    return result


def plot_list(a, b, polynomials, x_y_conv=None, precise_solution=None, title_user="Графіки наближених розв'язків рівняння"):
    if a > b:
        raise TypeError('a повинно бути менше b')
    if x_y_conv is not None:
        true_a = min(a, x_y_conv[0])
        true_b = max(b, x_y_conv[1])
    else:
        true_a = a
        true_b = b
    x_list = [true_a+0.01 * i for i in range(int((true_b-true_a)*100))]
    fig = plt.figure()
    for i in range(len(polynomials)):
        pic_f = custom_func(polynomials[i])
        picard_list = [pic_f(x) for x in x_list]
        if i == 0:
            plt.plot(x_list, picard_list, label=f'Ітерація №{i}', color='grey', linestyle='--')
        else:
            plt.plot(x_list, picard_list, label=f'Ітерація №{i}')

    if precise_solution is not None:
        precise_func = custom_func(precise_solution)
        precise_list = [precise_func(x) for x in x_list]
        plt.plot(x_list, precise_list, label="Точний розв'язок", color='black')
    if x_y_conv != None:
        plt.axvline(x=x_y_conv[0], label='Область збіжності', color='pink', linestyle='-.')
        plt.axvline(x=x_y_conv[1], color='pink', linestyle='-.')
    plt.title(f"{title_user}")
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    return fig


def picard_poly(dydx, x0, y0, _a, _b, alpha, convergence=None):
    if convergence == None:
        bond = [(_a, _b)]
    else:
        bond = [(convergence[0], convergence[1])]
    dydx = dydx.replace('y', '(y)')  # TODO Перевірка
    y_s = y0
    _y_list = [str(y_s)]
    f_xy = prepare_polynomial(dydx.replace('y', str(y0)))
    y_s = newton_leibnitz(integrate_polynomial(f_xy), x0) + '+' + str(y0)
    f_xy = dydx.replace('y', str(y_s))
    _y_list.append(prepare_polynomial(y_s))
    start_time = time.time()
    for i in range(4):
        if (time.time() - start_time) > 15:
            break
        f_xy = prepare_polynomial(f_xy)
        y_s = newton_leibnitz(integrate_polynomial(f_xy), x0) + '+' + str(y0)
        f_xy = dydx.replace('y', str(y_s))
        _y_list.append(prepare_polynomial(y_s))

    delta_y_i = custom_func('-abs((' + _y_list[1] + ')-(' + _y_list[0]+'))')  # Обернені знаки, бо мінімізація
    res = scipy.optimize.minimize(delta_y_i, x0=x0, bounds=bond)
    a_param = res.fun*(-1)
    if a_param != 0:
        n = int(log(0.01*(1-alpha)/a_param, alpha))
    else:
        n = 1000000
    return _y_list, a_param, n


def parce_helper(string):
    string = string.replace(' ', '')
    if 'від' in string:
        inx = string.index('від') + len('від')
        a_str = ''
        for char in string[inx:]:
            if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+']:
                break
            else:
                a_str += char
        _a = float(a_str)
        string = string.replace(f'від{a_str}', '')
    else:
        _a = -1

    if 'до' in string:
        b_str = ''
        inx = string.index('до') + len('до')
        for char in string[inx:]:
            if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+']:
                break
            else:
                b_str += char
        _b = float(b_str)
        string = string.replace(f'до{b_str}', '')
    else:
        _b = 1

    string = string.replace('х', 'x')
    x0_str = ''
    y0_str = ''
    if 'x0' in string and 'y0' in string:

        inx = string.index('x0=') + len('x0=')
        for char in string[inx:]:
            if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+']:
                break
            else:
                x0_str += char
        x0 = float(x0_str)


        inx = string.index('y0=') + len('y0=')
        for char in string[inx:]:
            if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+']:
                break
            else:
                y0_str += char
        y0 = float(y0_str)

    else:
        x0 = 0
        y0 = 1

    string = string.replace(f'x0={x0_str}', '').replace(f'y0={y0_str}', '')
    return _a, _b, string.replace(',', ''), x0, y0


def fuehrung(string):
    string = string.lower().replace(' ', '')
    query_type = 'error'
    if (('пікара' in string) or ("y'=" in string)) and ('рунге' not in string):
        string = string.replace('пікара', '').replace('метод', '')
        _a, _b, string, x0, y0 = parce_helper(string)
        query_type = 'difeq'
        return query_type, string.replace("y'=", ''), _a, _b, string, 'picard', x0, y0

    elif ('рунге' in string) and ('пікар' not in string):
        string = string.replace('рунгекутта', '').replace('рунге-кутта', '').replace('метод', '').replace('рунге', '')
        _a, _b, string, x0, y0 = parce_helper(string)
        query_type = 'difeq'
        return query_type, string.replace("y'=", ''), _a, _b, string, 'rk4', x0, y0

    elif ('рунге' in string) and ('пікар' in string):
        string = string.replace('рунгекутта', '').replace('рунге-кутта', '').replace('пікара', '').replace('метод', '')\
            .replace('та', '').replace('рунге', '')
        _a, _b, string, x0, y0 = parce_helper(string)
        query_type = 'difeq'
        return query_type, string.replace("y'=", ''), _a, _b, string, 'both', x0, y0

    elif 'похідна' in string and ('f=' in string or 'y=' in string):
        query_type = 'derivative'
        string = string.replace('похідна', '')
        _x0 = None
        if 'вточці' in string:
            x0_str = ''
            inx = string.index('вточці')+len('вточці')
            for char in string[inx:]:
                if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', ',', '-', '+']:
                    break
                else:
                    x0_str += char
            _x0 = tuple(eval(x0_str.replace(')', ',)')))
            string = string.replace(f'вточці{x0_str}', '')
        if 'поy' in string or 'поу' in string:
            res_precise = None
            string = string.replace('поy', '')
            try:
                res = find_y_partial_derivative(string.replace('f=', '').replace('y=', ''))
                if _x0 is not None:
                    if 'y' in res:
                        res_precise = eval(res.replace('^', '**').replace('x', '('+ str(_x0[0]) + ')').replace('y', '(' + str(_x0[1]) + ')'))
                    else:
                        res_precise = eval(res.replace('^', '**').replace('x', '(' + str(_x0[0]) + ')'))
                checker = eval(res.replace('x', '1.352422').replace('y', '2.12312').replace('^', '**'))
            except:
                res = 'На жаль, нам не вдалося знайти похідну цієї функції'
            if res_precise is not None:
                return query_type, string.replace('f=', '').replace('y=', ''), res, res_precise
            else:
                return query_type, string.replace('f=', '').replace('y=', ''), res, res
        else:
            res_precise = None
            string = string.replace('поx', '').replace('пох', '')
            try:
                res = derivative(string.replace('f=', '').replace('y=', ''))
                checker = eval(res.replace('x', '1.352422').replace('y', '2.12312').replace('^', '**'))
            except:
                res = 'На жаль, нам не вдалося знайти похідну цієї функції'

            if _x0 is not None:
                if 'y' in res:
                    res_precise = eval(res.replace('^', '**').replace('x', '(' + str(_x0[0]) + ')').replace('y', '('+ str(_x0[1]) + ')'))
                else:
                    res_precise = eval(res.replace('^', '**').replace('x', '(' + str(_x0[0]) + ')'))

            if res_precise is not None:
                return query_type, string.replace('f=', '').replace('y=', ''), res, res_precise
            else:
                return query_type, string.replace('f=', '').replace('y=', ''), res, res
    return query_type
