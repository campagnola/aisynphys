# coding: utf8
from __future__ import print_function, division

import traceback, sys, logging
import numpy as np   

import neuroanalysis.fitting
from neuroanalysis.filter import bessel_filter
from neuroanalysis.event_detection import exp_deconvolve, threshold_events
from neuroanalysis.baseline import mode_filter
from neuroanalysis.stimuli import find_square_pulses
from neuroanalysis.util.array import append_columns

from .pipeline_module import MultipatchPipelineModule
from .dataset import DatasetPipelineModule


class SynapticEventsPipelineModule(MultipatchPipelineModule):
    
    name = 'synaptic_events'
    dependencies = [DatasetPipelineModule]
    table_group = ['synaptic_events']

    @classmethod
    def create_db_entries(cls, job, session):
        db = job['database']
        job_id = job['job_id']

        # Load experiment from DB
        expt = db.experiment_from_ext_id(job_id, session=session)
        try:
            assert expt.data is not None
            # this should catch corrupt NWBs
            assert expt.data.contents is not None
        except Exception:
            error = 'No NWB data for this experiment'
            return [error]
        
        errors = []
        for sync_rec in expt.data.contents:
            # collect all stimulus locations
            stim_regions = {}  # dev_id: (start_times, stop_times)
            all_stim_regions = [[], []]
            for dev_id in sync_rec.devices:
                rec = sync_rec[dev_id]
                pulses = find_square_pulses(rec['command'])
                starts = np.array([pulse.start_time for pulse in pulses])
                stops = np.array([pulse.start_time + pulse.duration for pulse in pulses])
                stim_regions[dev_id] = (starts, stops)
                all_stim_regions[0].append(starts)
                all_stim_regions[1].append(stops)
            all_stim_regions = [np.concatenate(r) for r in all_stim_regions]

            for cell in expt.cell_list:
                # get all recordings made on this cell's electrode
                dev_id = cell.electrode.device_id
                if dev_id not in sync_rec.devices:
                    continue
                events = cls.detect_events(sync_rec, dev_id, stim_regions, all_stim_regions)
                if events is None:
                    continue

            # Write new record to DB
            entry = db.SynapticEvents(cell_id=cell.id)
            session.add(entry)

        return errors

    def job_records(self, job_ids, session):
        """Return a list of records associated with a list of job IDs.
        
        This method is used by drop_jobs to delete records for specific job IDs.
        """
        db = self.database
        q = session.query(db.synaptic_events)
        q = q.filter(db.synaptic_events.cell_id==db.Cell.id)
        q = q.filter(db.Cell.experiment_id==db.Experiment.id)
        q = q.filter(db.Experiment.ext_id.in_(job_ids))
        return q.all()

    @classmethod
    def detect_events(cls, sync_rec, device_id, stim_regions, all_stim_regions):
        threshold = 3  # detect events 3 stdev from noise
        recording = sync_rec[device_id]

        if recording.clamp_mode == 'ic':
            lowpass1 = 5000
            tau = 20e-3
            lowpass2 = 2000
            baseline = recording.baseline_potential
        else:
            lowpass1 = 10000
            tau = 3.9e-3
            lowpass2 = 2000
            baseline = recording.baseline_current

        bsub = recording['primary'] - baseline

        filtered = bessel_filter(bsub, lowpass1, btype='low')
        # filtered = bessel_filter(filtered, highpass, btype='high')

        # Nonlinear highpass: removes low-frequency drift with minimal distortion of events
        mfilt = mode_filter(filtered.data)
        filtered = filtered.copy(data=filtered.data-mfilt)
        
        deconv = bessel_filter(exp_deconvolve(filtered, tau), lowpass2, btype='low')

        # get stdev of quiescent regions as noise threshold
        baseline_regions = sync_rec.baseline_regions()
        baseline_data = np.concatenate([deconv.time_slice(start, stop - 30e-3).data for start, stop in baseline_regions])

        if len(baseline_data) < 1000:
            print('skip %s' % recording)
            return None

        stdev = baseline_data.std()
        mean = baseline_data.mean()
        bins = np.linspace(mean-2*stdev, mean+2*stdev, 100)
        hist = np.histogram(baseline_data, bins=bins)
        hmax = hist[0].max()
        gauss = neuroanalysis.fitting.Gaussian()
        fit = gauss.fit(hist[0], x=(bins[1:] + bins[:-1])/2, params={
            'xoffset': (mean, mean - stdev, mean + stdev), 
            'yoffset': (0, 'fixed'), 
            'amp': (hmax, hmax * 0.5, hmax * 2), 
            'sigma': (stdev, stdev * 0.2, stdev * 5)
        })

        deconv = deconv - fit.best_values['xoffset']        

        # find all events that cross threshold
        events = threshold_events(deconv, fit.best_values['sigma']*threshold)
        events = append_columns(events, [('in_spont_region', bool), ('in_evoked_region', bool), ('in_async_region', bool)])

        # sort events by region
        stim_mask = cls.events_in_regions(events['time'], stim_regions[device_id][0] - 30e-3, stim_regions[device_id][1] + 30e-3)
        events['in_evoked_region'] = (~stim_mask) & cls.events_in_regions(events['time'], all_stim_regions[0], all_stim_regions[1] + 6e-3)
        events['in_async_region'] = (~stim_mask) & (~events['in_evoked_region']) & cls.events_in_regions(events['time'], all_stim_regions[1] + 6e-3, all_stim_regions[1] + 50e-3)
        events['in_spont_region'] = (~stim_mask) & ~(events['in_evoked_region'] | events['in_async_region'])

        print(sync_rec.key)
        if sync_rec.key == 40:
            plt1 = show(sync_rec, device_id, deconv, events, fit.best_values['sigma']*threshold)

            import pyqtgraph as pg
            plt = pg.plot()
            plt.plot((bins[1:] + bins[:-1])/2, hist[0])
            plt.plot((bins[1:] + bins[:-1])/2, fit.best_fit, pen='g')
            raise Exception()
        return events, deconv

    @staticmethod
    def events_in_regions(times, starts, stops):
        return ((times[:, None] >= starts[None, :]) & (times[:, None] < stops[None, :])).any(axis=1)


def show(sync_rec, device_id, deconv, events, threshold):
    import pyqtgraph as pg 

    rec = sync_rec[device_id]

    win = pg.GraphicsLayoutWidget()
    plt1 = win.addPlot(0, 0)
    plt2 = win.addPlot(1, 0)
    plt1.setXLink(plt2)
    
    plt1.addLine(y=threshold)
    plt1.addLine(y=-threshold)

    plt1.plot(deconv.time_values, deconv.data)
    plt2.plot(rec['primary'].time_values, rec['primary'].data)
    plt2.plot(rec['primary'].time_values, sync_rec.baseline_mask().astype(float)*0.5e-9, pen='b')


    spontaneous_events = events[events['in_spont_region']]
    b_vt = pg.VTickGroup(spontaneous_events['time'], pen='g', yrange=[0, 0.1])
    plt1.addItem(b_vt)

    async_events = events[events['in_async_region']]
    a_vt = pg.VTickGroup(async_events['time'], pen='y', yrange=[0, 0.15])
    a_vt.setZValue(10)
    plt1.addItem(a_vt)

    evoked_events = events[events['in_evoked_region']]
    e_vt = pg.VTickGroup(evoked_events['time'], pen='r', yrange=[0, 0.2])
    e_vt.setZValue(20)
    plt1.addItem(e_vt)

    win.show()    
    return win
