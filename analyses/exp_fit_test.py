"""
For exploring and comparing curve fitting methods.
"""
import os, pickle, time
import numpy as np
import sympy
import scipy.optimize, scipy.ndimage
import pyqtgraph as pg
import pyqtgraph.multiprocess as mp
from neuroanalysis.fitting import Psp
from neuroanalysis.data import TSeries
from keras.models import Sequential
from keras.layers import Dense, Dropout


class ExpFitMethod:
    params = ['yoffset', 'amp', 'tau']
    dtype = [
        ('fit', object),
        ('yoffset', float),
        ('amp', float),
        ('tau', float),
        ('init_yoffset', float),
        ('init_amp', float),
        ('init_tau', float),
        ('err', float),
        ('nfev', int),
        ('success', bool),
        ('fit_time', float),
    ]

    _jac_fns = None

    def __init__(self, name, use_jac=True, method=None):
        self.name = name
        self.method = method
        self.use_jac = use_jac

        if use_jac and self._jac_fns is None:
            self._compile_jac()

    @classmethod
    def _compile_jac(cls):
        yoff, amp, tau, i, n = sympy.symbols('yoff amp tau i n')
        t = sympy.IndexedBase('t')
        y = sympy.IndexedBase('y')
        err = sympy.sqrt(sympy.summation((y[i] - yoff + amp * sympy.exp(-t[i] / tau))**2, [i, 0, n]))
        cls._jac_fns = (
            sympy.lambdify([yoff, amp, tau, t, y, n], sympy.diff(err, yoff), modules=['sympy']),
            sympy.lambdify([yoff, amp, tau, t, y, n], sympy.diff(err, amp), modules=['sympy']),
            sympy.lambdify([yoff, amp, tau, t, y, n], sympy.diff(err, tau), modules=['sympy']),
        )
    
    def fit(self, data):
        y = data.data
        t = data.time_values
        start = time.time()
        yoff = y[-1]
        amp = y[0] - yoff
        tau = t[-1] - t[0]
        init = (yoff, amp, tau)
        args = (t, y)
        jac_fn = self.exp_jac_fn if self.use_jac else None
        fit = scipy.optimize.minimize(fun=self.exp_err_fn, x0=init, args=args, jac=jac_fn, method=self.method)
        return {
            'fit': fit,
            'yoffset': fit.x[0],
            'amp': fit.x[1],
            'tau': fit.x[2],
            'init_yoffset': init[0],
            'init_amp': init[1],
            'init_tau': init[2],
            'err': fit.fun,
            'nfev': fit.nfev,
            'success': fit.success,
            'fit_time': time.time() - start,
        }

    def eval(self, result, t):
        x = (result['yoffset'], result['amp'], result['tau'])
        return self.exp_fn(x, t)

    @staticmethod
    def exp_fn(params, t):
        (yoffset, amp, tau) = params
        return yoffset + amp * np.exp(-t / tau)

    @staticmethod
    def exp_err_fn(params, t, y):
        residual = y - ExpFitMethod.exp_fn(params, t)
        return np.linalg.norm(residual)

    @classmethod
    def exp_jac_fn(cls, params, t, y):

        yoff, amp, tau = params
        N = len(y)
        norm = np.sqrt(((yoff + amp * np.exp(-t/tau) - y) ** 2).sum())
        exp_t_tau = np.exp(-t / tau)
        dyoff = (N * yoff + (amp * exp_t_tau).sum() - y.sum()) / norm
        damp = ((yoff + amp * exp_t_tau - y) * exp_t_tau).sum() / norm
        dtau = (amp * (yoff + amp* exp_t_tau - y) * exp_t_tau * t / tau**2).sum() / norm

        return np.array([dyoff, damp, dtau])
        """
        t = sympy.Array(t)
        y = sympy.Array(y)
        return np.array([fn(yoff, amp, tau, t, y, N-1) for fn in cls._jac_fns])
        """


class ExpGenerator:
    params = ['yoffset', 'amp', 'tau']
    dtype = [
        ('x', object),
        ('y', object),
        ('t', object),
        ('true_y', object),
        ('yoffset', float),
        ('amp', float),
        ('tau', float),
    ]

    def __init__(self, duration=0.4, sample_rate=50000):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.t = np.arange(0, duration, self.dt)
        
    def make_example(self):
        yoffset = np.random.uniform(-80e-3, -60e-3)
        amp = np.random.uniform(-100e-3, 100e-3)
        tau = np.random.uniform(5e-3, 500e-3)

        x = yoffset, amp, tau
        true_y = ExpFitMethod.exp_fn(x, self.t)
        y = true_y + self.make_noise(self.t)

        return {
            'x': x, 
            'y': y,
            't': self.t,
            'true_y': true_y,
            'yoffset': x[0],
            'amp': x[1],
            'tau': x[2],
        }

    def make_noise(self, t):
        kernel_t = np.arange(0, 0.1, self.dt)
        kernel = 0.01 * np.exp(-kernel_t / 50e-3)

        # noise = 0.02 * np.random.normal(size=len(t))
        # noise = scipy.ndimage.gaussian_filter(noise, 4)
        noise_amp = np.random.uniform(0.001, 0.01)
        noise = noise_amp * np.random.normal(size=len(t) + len(kernel_t))
        noise = np.convolve(noise, kernel, 'valid')

        return noise[:len(t)]


class PspFitMethod:
    params = ['yoffset', 'amp', 'rise_time', 'decay_tau']
    dtype = [
        # ('fit', object),
        ('yoffset', float),
        ('amp', float),
        ('rise_time', float),
        ('decay_tau', float),
        ('init_yoffset', float),
        ('init_amp', float),
        ('init_rise_time', float),
        ('init_decay_tau', float),
        ('err', float),
        ('nfev', int),
        ('success', bool),
        ('fit_time', float),
    ]

    def __init__(self, name, method=None):
        self.name = name
        self.method = method

    def train_epoch(self, train_x, train_y, test_x, test_y):
        pass

    def plot_training_history(self, plt):
        pass

    def fit(self, data):
        y = data.data
        t = data.time_values
        start = time.time()
        yoff = y[0]
        pos_amp = y.max() - yoff
        neg_amp = y.min() - yoff
        amp = neg_amp if abs(neg_amp) > pos_amp else pos_amp
        decay_tau = 20e-3
        rise_time = 3e-3
        init = (yoff, amp, rise_time, decay_tau)
        bounds = [
            (-1, 1),
            (-1, 1),
            (0.1e-3, 20e-3),
            (1e-3, 200e-3),
        ]
        args = (t, y)
        fit = scipy.optimize.minimize(fun=self.psp_err_fn, x0=init, args=args, method=self.method, bounds=bounds)
        return {
            # 'fit': fit,
            'yoffset': fit.x[0],
            'amp': fit.x[1],
            'rise_time': fit.x[2],
            'decay_tau': fit.x[3],
            'init_yoffset': init[0],
            'init_amp': init[1],
            'init_rise_time': init[2],
            'init_decay_tau': init[3],
            'err': fit.fun,
            'nfev': fit.nfev,
            'success': fit.success,
            'fit_time': time.time() - start,
        }

    def eval(self, result, t):
        x = (result['yoffset'], result['amp'], result['rise_time'], result['decay_tau'])
        return self.psp_fn(x, t)

    @staticmethod
    def psp_fn(params, t):
        (yoffset, amp, rise_time, decay_tau) = params
        return Psp.psp_func(x=t, xoffset=0.01, yoffset=yoffset, amp=amp, rise_time=rise_time, decay_tau=decay_tau, rise_power=2)

    @staticmethod
    def psp_err_fn(params, t, y):
        residual = y - PspFitMethod.psp_fn(params, t)
        return np.linalg.norm(residual)


class PspMLMethod:
    params = ['yoffset', 'amp', 'rise_time', 'decay_tau']
    dtype = [
        ('fit', object),
        ('yoffset', float),
        ('amp', float),
        ('rise_time', float),
        ('decay_tau', float),
        ('init_yoffset', float),
        ('init_yscale', float),
        ('init_tscale', float),
        ('fit_time', float),
    ]

    def __init__(self, name, input_size):
        self.name = name

        model = Sequential()
        model.add(Dense(100, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        model.add(Dense(40, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(4, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        self.model = model
        self.train_history = []
        self.test_history = []
        self.epochs = []

    def train_epoch(self, train_input, train_output, test_input, test_output):
        assert np.all(np.isfinite(train_input))
        assert np.all(np.isfinite(train_output))
        
        yoffset = train_input[:, :1]
        yscale = 100.0
        tscale = 100.0

        train_input_scaled = (train_input - yoffset) * yscale

        train_output_scaled = np.empty_like(train_output)

        train_output_scaled[:, 0] = (train_output[:, 0] - yoffset[:,0]) * yscale
        train_output_scaled[:, 1] = train_output[:, 1] * yscale
        train_output_scaled[:, 2] = train_output[:, 2] * tscale
        train_output_scaled[:, 3] = train_output[:, 3] * tscale

        # print("fit:")
        # print(train_input_scaled.mean(), train_input_scaled.std())
        # for i in range(4):
        #     print(i)
        #     print(train_output[:, i].mean(), train_output[:, i].std())
        #     print(train_output_scaled[:, i].mean(), train_output_scaled[:, i].std())

        ep = self.model.fit(train_input_scaled, train_output_scaled, epochs=1)
        self.epochs.append(ep)
        # self.train_history.append(ep.history['loss'][0])

        for input, output, hist in [
            (train_input, train_output, self.train_history), 
            (test_input, test_output, self.test_history)]: 

            err = np.mean([np.linalg.norm(np.array(self.fit(input[i])['fit']) - output[i]) for i in range(len(input))])
            hist.append(err)

    def plot_training_history(self, plt):
        plt.plot(self.train_history, pen='g', name=self.name + ' train_loss')
        plt.plot(self.test_history, pen={'color': 'g', 'style': pg.QtCore.Qt.DashLine}, name=self.name + ' test_loss')

    def fit(self, data):
        if isinstance(data, TSeries):
            y = data.data
        else:
            y = data

        start = time.time()
        yoffset = y[0]
        yscale = 100.0
        tscale = 100.0

        y = (y - yoffset) * yscale

        fit = self.model.predict(y[None, :])[0]
        fit = [
            fit[0] / yscale + yoffset,
            fit[1] / yscale,
            fit[2] / tscale,
            fit[3] / tscale,
        ]
        assert np.all(np.isfinite(fit))
        return {
            'fit': fit,
            'yoffset': fit[0],
            'amp': fit[1],
            'rise_time': fit[2],
            'decay_tau': fit[3],
            'init_yoffset': yoffset,
            'init_yscale': yscale,
            'init_tscale': tscale,
            'fit_time': time.time() - start,
        }

    def eval(self, result, t):
        x = (result['yoffset'], result['amp'], result['rise_time'], result['decay_tau'])
        return self.psp_fn(x, t)

    @staticmethod
    def psp_fn(params, t):
        (yoffset, amp, rise_time, decay_tau) = params
        return Psp.psp_func(x=t, xoffset=0.01, yoffset=yoffset, amp=amp, rise_time=rise_time, decay_tau=decay_tau, rise_power=2)

    @staticmethod
    def psp_err_fn(params, t, y):
        residual = y - PspFitMethod.psp_fn(params, t)
        return np.linalg.norm(residual)


class PspGenerator:
    params = ['yoffset', 'amp', 'rise_time', 'decay_tau']
    dtype = [
        ('params', object),
        ('data', object),
        ('true_data', object),
        ('yoffset', float),
        ('amp', float),
        ('rise_time', float),
        ('decay_tau', float),
    ]

    def __init__(self, duration=0.4, sample_rate=50000):
        self.sample_rate = sample_rate
        self.duration = duration
        self.template = TSeries(np.zeros(int(duration*sample_rate)), sample_rate=sample_rate)

    def make_example(self):
        yoffset = np.random.uniform(-80e-3, -60e-3)
        amp = np.random.uniform(-10e-3, 10e-3)
        rise_t = np.random.uniform(0.5e-3, 10e-3)
        decay_tau = np.random.uniform(2, 10) * rise_t

        params = yoffset, amp, rise_t, decay_tau
        true_y = PspFitMethod.psp_fn(params, self.template.time_values)
        y = true_y + self.make_noise()

        return {
            'data': self.template.copy(data=y),
            'true_data': self.template.copy(data=true_y),
            'params': params,
            'yoffset': params[0],
            'amp': params[1],
            'rise_time': params[2],
            'decay_tau': params[3],
        }

    def make_noise(self):
        n_samp = len(self.template)
        kernel_t = np.arange(0, 0.1, self.template.dt)
        tau = np.random.uniform(10e-3, 100e-3)
        kernel = 0.01 * np.exp(-kernel_t / tau)

        # noise = 0.02 * np.random.normal(size=len(t))
        # noise = scipy.ndimage.gaussian_filter(noise, 4)
        noise_amp = np.random.uniform(0.001, 0.003)
        noise = noise_amp * np.random.normal(size=n_samp + len(kernel_t))
        noise = np.convolve(noise, kernel, 'valid')
        noise -= noise[:int(0.01/self.template.dt)].mean()

        return noise[:n_samp]


if __name__ == '__main__':
    app = pg.mkQApp()
    pg.dbg()

    sample_rate = 50000
    duration = 0.1
    n_samp = int(duration * sample_rate)
    n_train = 50000
    n_test = 2000
    n_epochs = 20

    # generator = ExpGenerator(duration=0.4, sample_rate=sample_rate)
    
    # methods = [
    #     ExpFitMethod(name='minimize_wo_jac', use_jac=False),
    #     ExpFitMethod(name='minimize_w_jac_slsqp', use_jac=True, method='SLSQP'),
    #     ExpFitMethod(name='minimize_w_jac_cobyla', use_jac=True, method='COBYLA'),
    #     ExpFitMethod(name='minimize_w_jac_bfgs', use_jac=True, method='BFGS'),
    #     ExpFitMethod(name='minimize_w_jac_l_bfgs_b', use_jac=True, method='L-BFGS-B'),
    #     # ExpFitMethod(name='minimize_w_jac_dogleg', use_jac=True, method='dogleg'),   # requires hessian
    # ]

    generator = PspGenerator(duration=0.1, sample_rate=sample_rate)

    methods = [
        PspFitMethod(name='minimize'),
        PspMLMethod(name='nn_predict', input_size=n_samp),
    ]

    dtype = generator.dtype.copy()
    dtype.append(('training', bool))

    for method in methods:
        pfx = method.name + '_'
        for field in method.dtype:
            dtype.append((pfx + field[0], field[1]))
        dtype.extend([
            (pfx+'fit_data', object),
            (pfx+'true_err', float),
        ])
        for par_name in method.params:
            dtype.append((pfx + par_name + '_err', float))

    examples = np.empty((n_train + n_test), dtype=dtype)

    example_cache = 'examples.pkl'
    if os.path.exists(example_cache):
        examples = pickle.load(open(example_cache, 'rb'))
    else:
        with pg.ProgressDialog("making some noise..", maximum=examples.shape[0]) as dlg:
            for i in range(examples.shape[0]):
                ex = generator.make_example()
                for k,v in ex.items():
                    examples[i][k] = v
                dlg += 1
                if dlg.wasCanceled():
                    raise Exception("User cancel")

            examples[:n_train]['training'] = True
            examples[n_train:]['training'] = False

        tmp = example_cache + '.tmp'
        pickle.dump(examples, open(tmp, 'wb'))
        os.rename(tmp, example_cache)

    n_params = len(generator.params)
    example_len = len(examples[0]['data'])
    train_params = np.empty((n_train, n_params))
    train_data = np.empty((n_train, example_len))
    test_params = np.empty((n_test, n_params))
    test_data = np.empty((n_test, example_len))
    for i in range(n_train):
        train_params[i] = examples[i]['params']
        train_data[i] = examples[i]['data'].data
    for i in range(n_test):
        test_params[i] = examples[n_train+i]['params']
        test_data[i] = examples[n_train+i]['data'].data        


    with pg.ProgressDialog("training hard..", maximum=n_epochs) as dlg:
        train_plt = pg.plot()
        train_plt.addLegend()

        for i in range(n_epochs):
            train_plt.clear()
            for method in methods:
                method.train_epoch(train_data, train_params, test_data, test_params)
                method.plot_training_history(train_plt)
            app.processEvents()

            dlg += 1
            if dlg.wasCanceled():
                raise Exception("User cancel")
        

    with pg.ProgressDialog("fitting, don't you think?", maximum=n_train+n_test) as dlg:
        for i in range(n_train+n_test):
            ex = examples[i]
            data = ex['data']
            t = data.time_values
            params = ex['params']
            true_data = ex['true_data']
            for method in methods:
                pfx = method.name + '_'
                result = method.fit(data)
                for k,v in result.items():
                    ex[pfx+k] = v
                try:
                    fit_y = method.eval(result, t)
                    ex[pfx+'fit_data'] = data.copy(data=fit_y)
                    ex[pfx+'true_err'] = np.linalg.norm(true_data.data - fit_y)
                except:
                    pg.debug.printExc()
                for pname in method.params:
                    ex[pfx+pname+'_err'] = ex[pname] - result[pname]
            dlg += 1
            if dlg.wasCanceled():
                raise Exception("User cancel")

    plt = pg.plot()
    plt.addLegend()

    sp = pg.ScatterPlotWidget()
    fields = []
    for typ in dtype:
        if typ[1] is object:
            continue
        if typ[1] is bool:
            fields.append((typ[0], {'mode': 'enum', 'values': [True, False]}))
        else:
            fields.append((typ[0], {'mode': 'range'}))
    sp.setFields(fields)
    sp.setData(examples)

    ch = sp.colorMap.addNew('training')
    ch['Values', 'True'] = 'g'
    ch['Values', 'False'] = 'y'

    ch = sp.filter.addNew('nn_predict_true_err')
    ch['Max'] = 100

    sp.show()


    def pointsClicked(sp, pts):
        global sel
        sel = [pt.data() for pt in pts]
        plt.clear()
        for pt in pts:
            d = pt.data()
            t = d['data'].time_values
            y = d['data'].data
            plt.plot(t, y, antialias=True, name='y')
            plt.plot(t, d['true_data'].data, pen={'color': 'w', 'style': pg.QtCore.Qt.DashLine}, antialias=True, name='true_y')
            for i,method in enumerate(methods):
                pfx = method.name + '_'
                data = d[pfx+'fit_data']
                if data is None:
                    continue
                plt.plot(t, data.data, pen=(i, 5), antialias=True, name=method.name)

            print("----------")
            for n in d.dtype.names:
                s = []
                for i,line in enumerate(str(d[n]).split('\n')):
                    if i > 0:
                        line = ' '*30 + line
                    s.append(line)
                s = '\n'.join(s)
                print("{:30s}{:s}".format(n, s))

        sp.setSelectedPoints(pts)

    sp.sigScatterPlotClicked.connect(pointsClicked)







"""
### Can we use sympy to generate functions for Jacobian / Hessian matrices?
###   -> Yes, but the functions are not optimized, and not numba/jit compilable.


import time
import sympy

yoff, amp, tau, i, n = sympy.symbols('yoff amp tau i n')
t = sympy.IndexedBase('t')
y = sympy.IndexedBase('y')
err = sympy.sqrt(sympy.summation((y[i] - yoff + amp * sympy.exp(-t[i] / tau))**2, [i, 0, n]))
derr_dtau = sympy.diff(err, tau)
derr_dtau_fn = sympy.lambdify([yoff, amp, tau, t, y, n], derr_dtau, modules=['sympy'])

start = time.time(); derr_dtau.evalf(subs={'amp': 1, 'yoff': 0, 'tau': 0.5, 'n': 2, 't': sympy.Array([0, 1, 2]), 'y': sympy.Array([5, 4, 3])}); print(time.time() - start)
start = time.time(); derr_dtau_fn(0, 1, 0.5, sympy.Array([0., 1, 2]), sympy.Array([5., 4, 3]), 2); print(time.time() - start)





import time
import sympy
import inspect
import numba


import sympy.printing.lambdarepr as SPL
class NumbaPrinter(SPL.NumPyPrinter):
    def _print_Sum(self, expr):
        code = ['tot_ = 0']
        indent = ''
        for i, a, b in expr.limits:
            code.append('{ind}for {i} in range({a}, {b}+1)'.format(ind=indent, i=self._print(i), a=self._print(a), b=self._print(b)))
            indent += '    '
        code.append('{ind}tot += {function}'.format(ind=indent, function=self._print(expr.function)))
        return '\n'.join(code)

i, n = sympy.symbols('i n')
y = sympy.IndexedBase('y')
expr = sympy.summation(y[i]+1, [i, 0, n-1])
fn = sympy.lambdify([y, n], expr, printer=NumbaPrinter, modules=['numpy'])

print(inspect.getsource(fn))

jit_fn = numba.jit(fn)

import numpy as np
a = np.arange(20000)

start = time.time(); fn(a, len(a)); print(time.time() - start)
start = time.time(); (a+1).sum(); print(time.time() - start)
start = time.time(); jit_fn(a, len(a)); print(time.time() - start)


"""
