import base64
from io import BytesIO
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from math import sin, cos, e, pi
from Term_Paper_New import custom_func, plot_piecewise, fuehrung, find_d_and_alpha, plot_list, picard_poly, picard,\
    add_superscript, RK4

app = Flask(__name__)

def nonlinear_case_helper(dxdy, x0, y0, _a, _b, alpha, convergence):
    try:
        func_array = picard(dxdy, x0, y0, _a, _b, alpha, convergence)
        fig = plot_piecewise(func_array)
    except:
        message = ['Отакої! Нам не вдалося інтерпретувати Ваш запит, спробуйте ще раз!']
        return render_template('home.html', data=None, message=message)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = ['data:image/png;base64,' + str(base64.b64encode(buf.getbuffer()).decode("ascii"))]
    return data

@app.route("/")
def home():
    return render_template('home.html', data=None, message=None)


@app.route("/solve", methods=['POST'])
def solver():
    query = request.form['query']
    analysis = fuehrung(query)
    try:
        type = analysis[5]
    except:
        type = None
    if analysis[0] == 'difeq' and (type == 'picard' or type == 'both'):
        dxdy = analysis[1]
        _a = analysis[2]
        _b = analysis[3]
        x0 = float(analysis[6])
        y0 = float(analysis[7])

        if all(x not in dxdy for x in ['log', 'sin', 'cos', 'e']):
            try:
                x_range, y_range, alpha = find_d_and_alpha(_a, _b, x0, y0, dxdy)
                if x_range == 'Метод розбіжний':
                    message = 'Метод розбіжний: не виконується умова Ліпшиця'
                    return render_template('home.html', data=None, message=[message])
                try:
                    poly_list, a, n = picard_poly(dxdy, x0, y0, x_range[0], x_range[1], alpha, convergence=x_range)
                    n_made = len(poly_list)
                except TypeError:
                    data = nonlinear_case_helper(dxdy, x0, y0, _a, _b, alpha, x_range)
                    message = [f"<i>Рівняння</i>: <b>{add_superscript(analysis[4])}</b> за умови <b>y({x0})={y0}</b>",
                               f"<br> Використано загальний алгоритм, адже було виявлено дробну степінь. ",
                               f"<br> Розв'язок збіжний у області: <br><b>G: "
                               f"{{(x, y)| x є [{round(x_range[0], 3)}, {round(x_range[1], 3)}],  "
                               f"y є [{round(y_range[0], 3)}, {round(y_range[1], 3)}]}}</b> "
                               f"<br><br>З параметром: <br><b>α = {round(alpha, 9)}</b> "]
                    return render_template('home.html', data=data, message=message)

                x_y_convergence = x_range
                fig = plot_list(_a, _b, poly_list, x_y_conv=x_y_convergence, precise_solution=None)
            except:
                message = ['Отакої! Нам не вдалося інтерпретувати Ваш запит, спробуйте ще раз!']
                return render_template('home.html', data=None, message=message)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            data = ['data:image/png;base64,' + str(base64.b64encode(buf.getbuffer()).decode("ascii"))]

            if type == 'both':
                x_list, y_list = RK4(dxdy, x0, y0, _a, _b)
                fig = plt.figure()
                plt.plot(x_list, y_list, label='Наближення інтегральної кривої')
                plt.title("Розв'язок задачі Коші методом Рунге-Кутта")
                plt.xlabel('x')
                plt.ylabel('y')
                buf = BytesIO()
                fig.savefig(buf, format="png")
                data.append('data:image/png;base64,' + str(base64.b64encode(buf.getbuffer()).decode("ascii")))

            fig2 = plot_list(x_range[0], x_range[1], poly_list,
                             title_user="Графіки наближених розв'язків рівняння у області збіжності")  # Тільки на області зб.
            buf2 = BytesIO()
            fig2.savefig(buf2, format="png")
            data.append('data:image/png;base64,' + str(base64.b64encode(buf2.getbuffer()).decode("ascii")))

            analytical_start = add_superscript(poly_list[-1])
            analytical = ''
            counter = 0
            for char in analytical_start:
                if char == '+' or char == '-':
                    counter += 1
                    if counter % 3 == 0:
                        analytical += '<br>+'
                    else:
                        analytical += char
                else:
                    analytical += char
            current_precision = round((alpha**5)*a/(1-alpha), 5) + 0.00001
            message = list([f"<i>Рівняння</i>: <b>{add_superscript(analysis[4])}</b> за умови <b>y({x0})={y0}</b>",
                      f"Аналітичний запис розв'язку: <b>y = {analytical}</b>",
                      f"<br> Розв'язок збіжний у області: <br><b>G: "
                      f"{{(x, y)| x є [{round(x_range[0], 3)}, {round(x_range[1], 3)}],  "
                      f"y є [{round(y_range[0], 3)}, {round(y_range[1], 3)}]}}</b> "
                      f"<br><br>З параметрами: <br><b>α = {round(alpha,9)}</b> "
                      f"<br> <b> a = {a} </b>",
                      f"<br> Для досягнення точності у 0.01 всюди у <b>G</b> необхідно зробити <b>n = {n}</b> ітерацій" 
                      f"<br> Тим не менш, на даний момент, " 
                      f"кількість ітерацій завжди фіксована і не перевищує <b>5</b>"
                      f"<br>Цього разу зроблено {n_made-1} ітерацій"
                      f"<br> При чому похибка не перевищує {current_precision}"])
            for i in range(len(message)):
                message[i] = message[i].replace('**', '^').replace('*', '⋅')
            return render_template('home.html', data=data, message=message)
        else:
            x_range, y_range, alpha = find_d_and_alpha(_a, _b, x0, y0, dxdy)
            data = nonlinear_case_helper(dxdy, x0, y0, _a, _b, alpha, x_range)
            message = [f"<i>Рівняння</i>: <b>{add_superscript(analysis[4])}</b> за умови <b>y({x0})={y0}</b>",
                      f"<br> Використано загальний алгоритм, адже було виявлено нелінійний елемент.",
                       f"<br> Розв'язок збіжний у області: <br><b>G: "
                       f"{{(x, y)| x є [{round(x_range[0], 3)}, {round(x_range[1], 3)}],  "
                       f"y є [{round(y_range[0], 3)}, {round(y_range[1], 3)}]}}</b> "
                       f"<br><br>З параметром: <br><b>α = {round(alpha, 9)}</b> "]
            return render_template('home.html', data=data, message=message)

    elif analysis[0] == 'difeq' and type == 'rk4':
        dxdy = analysis[1]
        _a = analysis[2]
        _b = analysis[3]
        x0 = 0
        y0 = 1
        x_list, y_list = RK4(dxdy, x0, y0, _a, _b)
        fig = plt.figure()
        plt.plot(x_list, y_list, label='Наближення інтегральної кривої')
        plt.title("Розв'язок задачі Коші методом Рунге-Кутта")
        plt.xlabel('x')
        plt.ylabel('y')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = ['data:image/png;base64,' + str(base64.b64encode(buf.getbuffer()).decode("ascii"))]
        message = [f"<i>Рівняння</i>: <b>{add_superscript(analysis[4])}</b> за умови <b>y({x0})={y0}</b>",
                   f"<br>Використано метод Рунге-Кутта 4-го порядку точності, відносно кроку інтегрування"]
        return render_template('home.html', data=data, message=message)
    elif analysis[0] == 'derivative':
        if analysis[2] == 'На жаль, нам не вдалося знайти похідну цієї функції':
            return render_template('home.html', data=None, message=['На жаль, нам не вдалося знайти похідну цієї функції'])

        surface_func = analysis[1].replace('^', '**').replace('sin', 'np.sin').replace('cos', 'np.cos')
        X = np.linspace(-2, 2, 500)
        Y = np.linspace(-2, 2, 500)
        x, y = np.meshgrid(X, Y)
        z = eval(surface_func)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(x, y, z, cmap='jet', edgecolor='none')
        ax.set_title(f"Графік похідної функції f={analysis[1]}")
        ax.set_xlabel("x", fontweight='bold')
        ax.set_ylabel("y", fontweight='bold')
        ax.set_zlabel("f", fontweight='bold')


        if analysis[2] != analysis[3]:
            message = f"<i>Похідна функції </i>: <b>{add_superscript(analysis[1])}</b>" \
                      f"<br> <b>f' = {add_superscript(analysis[2])}</b>" \
                      f"<br><b>f'(x0, y0) = {analysis[3]}</b>".replace('**', '^').replace('*', '⋅')
        else:
            message = f"<i>Похідна функції </i>: <b>{add_superscript(analysis[1])}</b>" \
                      f"<br> <b>f' = {add_superscript(analysis[2])}</b>".replace('**', '^').replace('*', '⋅')

        buf2 = BytesIO()
        fig.savefig(buf2, format="png")
        data1 = ['data:image/png;base64,' + str(base64.b64encode(buf2.getbuffer()).decode("ascii"))]

        return render_template('home.html', data=data1, message=[message])
    else:
        message = 'Отакої! Нам не вдалося інтерпретувати Ваш запит, спробуйте ще раз!'
        return render_template('home.html', data=None, message=[message])


if __name__ == '__main__':
   app.run()