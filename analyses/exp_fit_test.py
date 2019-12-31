"""
For exploring and comparing curve fitting methods.
"""
import time
import numpy as np
import scipy.optimize, scipy.ndimage
import pyqtgraph as pg
import pyqtgraph.multiprocess as mp


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

    def __init__(self, name, use_jac=True, method=None):
        self.name = name
        self.method = method
        self.use_jac = use_jac
    
    def fit(self, y, t):
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

    @staticmethod
    def exp_jac_fn(params, t, y):
        x0, x1, x2 = params
        N = len(y)

        norm = np.sqrt(((x0 + x1 * np.exp(-t/x2) - y) ** 2).sum())
        exp_t_tau = np.exp(-t / x2)
        dx0 = (N * x0 + (x1 * exp_t_tau).sum() - y.sum()) / norm
        dx1 = ((x0 + x1 * exp_t_tau - y) * exp_t_tau).sum() / norm
        dx2 = (x1 * (x0 + x1* exp_t_tau - y) * exp_t_tau * t / x2**2).sum() / norm

        return np.array([dx0, dx1, dx2])


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
        noise = 0.002 * np.random.normal(size=len(t) + len(kernel_t))
        noise = np.convolve(noise, kernel, 'valid')

        return noise[:len(t)]



if __name__ == '__main__':
    pg.mkQApp()
    pg.dbg()

    sample_rate = 50000
    N = 1000

    generator = ExpGenerator(duration=0.4, sample_rate=sample_rate)
    
    methods = [
        ExpFitMethod(name='minimize_wo_jac', use_jac=False),
        ExpFitMethod(name='minimize_w_jac_slsqp', use_jac=True, method='SLSQP'),
        ExpFitMethod(name='minimize_w_jac_cobyla', use_jac=True, method='COBYLA'),
        ExpFitMethod(name='minimize_w_jac_bfgs', use_jac=True, method='BFGS'),
        # ExpFitMethod(name='minimize_w_jac_dogleg', use_jac=True, method='dogleg'),   # !!!
        ExpFitMethod(name='minimize_w_jac_l_bfgs_b', use_jac=True, method='L-BFGS-B'),
    ]

    dtype = generator.dtype.copy()

    for method in methods:
        pfx = method.name + '_'
        for field in method.dtype:
            dtype.append((pfx + field[0], field[1]))
        dtype.extend([
            (pfx+'fit_y', object),
            (pfx+'true_err', float),
        ])
        for par_name in method.params:
            dtype.append((pfx + par_name + '_err', float))

    examples = np.empty(N, dtype=dtype)

    with pg.ProgressDialog("making some noise..", maximum=N) as dlg:
        for i in range(N):
            ex = generator.make_example()
            for k,v in ex.items():
                examples[i][k] = v
            dlg += 1

    with pg.ProgressDialog("fitting, don't you think?", maximum=N) as dlg:
        for i in range(N):
            ex = examples[i]
            y = ex['y']
            t = ex['t']
            x = ex['x']
            true_y = ex['true_y']
            for method in methods:
                pfx = method.name + '_'
                result = method.fit(y, t)
                for k,v in result.items():
                    ex[pfx+k] = v
                fit_y = method.eval(result, t)
                ex[pfx+'fit_y'] = fit_y
                ex[pfx+'true_err'] = np.linalg.norm(true_y - fit_y)
                for pname in method.params:
                    ex[pfx+pname+'_err'] = ex[pname] - result[pname]
            dlg += 1


    # with pg.ProgressDialog("quantifying life mistakes..", maximum=N) as dlg:
    #     for i,result in enumerate(results):
    #         for method_name, fit in result:
    #             ex = examples[i]


    #             for i,param in enumerate(method.params):
    #                 ex[pfx+param] = 

    #             ex['fit_yoffset'] = fit_x[0]
    #             ex['fit_amp'] = fit_x[1]
    #             ex['fit_tau'] = fit_x[2]
    #             ex['fit'] = fit
    #             ex['fit_y'] = fit_y
    #             ex['fit_err'] = fit_err
    #             ex['true_err'] = true_err
    #             ex['yoffset_err'] = x[0] - fit_x[0]
    #             ex['amp_err'] = x[1] - fit_x[1]
    #             ex['tau_err'] = x[2] - fit_x[2]
    #             ex['fit_success'] = fit_success
    #             ex['fit_nfev'] = fit_nfev

    #             dlg += 1

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

    # ch = sp.colorMap.addNew('fit_success')
    # ch['Values', 'True'] = 'g'
    # ch['Values', 'False'] = 'r'


    sp.show()


    def pointsClicked(sp, pts):
        global sel
        sel = [pt.data() for pt in pts]
        plt.clear()
        for pt in pts:
            d = pt.data()
            plt.plot(d['t'], d['y'], antialias=True, name='y')
            plt.plot(d['t'], d['true_y'], pen={'color': 'w', 'style': pg.QtCore.Qt.DashLine}, antialias=True, name='true_y')
            for i,method in enumerate(methods):
                pfx = method.name + '_'
                plt.plot(d['t'], d[pfx+'fit_y'], pen=(i, 5), antialias=True, name=method.name)

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

